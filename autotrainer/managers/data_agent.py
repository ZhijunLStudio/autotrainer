"""DataAgent — orchestrates the complete data processing pipeline.

Flow per dataset:
  1. inspect()       — sample raw data, infer schema
  2. generate_script() — LLM writes a conversion script
  3. sandbox.run()   — execute the script
  4. validate()      — check output is valid erniekit JSONL
  5. fix_script()    — if failed, LLM repairs and retries (up to MAX_RETRY)
  6. clean()         — dedup, remove bad rows, normalize
  7. profile()       — statistics + sample preview
  8. split()         — train / val / test

Context is managed via ContextStore (budget-based).
All decisions are logged to experiment_index for reproducibility.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import click

from autotrainer.context.store import ContextStore
from autotrainer.managers.raw_inspector import RawInspector, InspectionResult
from autotrainer.managers.sandbox import Sandbox, SandboxResult
from autotrainer.pf_integration.dataset_validator import DatasetValidator
from autotrainer.utils.file_utils import atomic_write_json, safe_read_json


MAX_RETRY = 3       # how many times to try fixing the script
MAX_SCRIPT_TOKENS = 3000   # limit script size going into context


@dataclass
class DatasetResult:
    """Full result for one processed dataset."""

    source_path: str = ""
    dataset_name: str = ""
    format_identified: str = ""
    script_path: str = ""          # saved conversion script
    raw_jsonl_path: str = ""       # output of conversion script
    cleaned_path: str = ""
    split: dict = field(default_factory=dict)   # {train, val, test} -> {path, count}
    profile: dict = field(default_factory=dict)
    clean_stats: dict = field(default_factory=dict)
    attempts: int = 0
    status: str = "pending"        # pending / completed / failed
    errors: list[str] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    def to_record(self) -> dict:
        return {
            "source_path": self.source_path,
            "dataset_name": self.dataset_name,
            "format_identified": self.format_identified,
            "script_path": self.script_path,
            "raw_jsonl_path": self.raw_jsonl_path,
            "cleaned_path": self.cleaned_path,
            "split": self.split,
            "profile": self.profile,
            "clean_stats": self.clean_stats,
            "attempts": self.attempts,
            "status": self.status,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class DataAgent:
    """LLM-driven data processing agent."""

    def __init__(
        self,
        work_dir: str,
        llm_client=None,
        context_store: ContextStore | None = None,
        sandbox_timeout: int = 300,
    ):
        self.work_dir = work_dir
        self.llm = llm_client
        self.context = context_store or ContextStore()
        self.inspector = RawInspector()
        self.sandbox = Sandbox(timeout=sandbox_timeout)
        self.validator = DatasetValidator()

        # Lazy-import skill handlers (avoid circular import)
        self._inspect_handler = None
        self._fix_handler = None

        os.makedirs(work_dir, exist_ok=True)

    def _get_inspect_handler(self):
        if self._inspect_handler is None:
            from autotrainer.skills.data_inspect.handler import DataInspectHandler
            self._inspect_handler = DataInspectHandler(llm_client=self.llm)
        return self._inspect_handler

    def _get_fix_handler(self):
        if self._fix_handler is None:
            from autotrainer.skills.data_fix.handler import DataFixHandler
            self._fix_handler = DataFixHandler(llm_client=self.llm)
        return self._fix_handler

    # ══════════════════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════════════════

    def run(self, data_paths: list[str]) -> list[DatasetResult]:
        """Process a list of data paths. Returns results for all datasets."""
        results = []
        for path in data_paths:
            click.echo(f"\n{'═' * 65}")
            click.echo(f"  Processing: {path}")
            click.echo(f"{'═' * 65}")
            result = self._process_one(path)
            results.append(result)
            self._save_result(result)

        self._print_summary(results)
        return results

    def _process_one(self, source_path: str) -> DatasetResult:
        """Full pipeline for a single data source."""
        name = Path(source_path).stem
        ds_dir = os.path.join(self.work_dir, name)
        os.makedirs(ds_dir, exist_ok=True)

        result = DatasetResult(
            source_path=source_path,
            dataset_name=name,
            started_at=datetime.now().isoformat(),
        )

        # ── Step 1: Inspect ──────────────────────────────────
        click.echo(f"\n  [1/7] Inspecting data...")
        inspection = self.inspector.inspect(source_path)
        summary = inspection.to_llm_summary()
        click.echo(f"    Format: {inspection.format_hint}")
        click.echo(f"    Rows (estimated): {inspection.estimated_rows}")
        click.echo(f"    Schema fields: {list(inspection.schema.keys())}")
        if inspection.issues:
            for issue in inspection.issues:
                click.echo(f"    [WARN] {issue}")

        # Update context data zone
        self.context.set_data_profile({
            "dataset": name,
            "format": inspection.format_hint,
            "rows": inspection.estimated_rows,
            "schema": inspection.schema,
            "sample_count": len(inspection.samples),
        })

        # ── Step 2: Generate conversion script ───────────────
        click.echo(f"\n  [2/7] Generating conversion script...")
        script = self._generate_script(summary, source_path, inspection)
        if not script:
            result.status = "failed"
            result.errors.append("LLM script generation failed")
            result.completed_at = datetime.now().isoformat()
            return result

        result.format_identified = "auto-detected"
        script_path = os.path.join(ds_dir, "convert_script.py")
        with open(script_path, "w") as f:
            f.write(script)
        result.script_path = script_path
        click.echo(f"    Script saved: {script_path}")

        # Syntax check
        ok, syntax_err = self.sandbox.validate_script(script)
        if not ok:
            click.echo(f"    [WARN] Syntax error: {syntax_err}")
            # Try to fix syntax immediately
            script = asyncio.run(
                self._get_fix_handler().fix_script(
                    script, f"Syntax error:\n{syntax_err}",
                    inspection.samples[:3], attempt=0
                )
            ).get("script", script)

        # ── Step 3-5: Run → Validate → Fix loop ──────────────
        jsonl_path = os.path.join(ds_dir, f"raw_{name}.jsonl")
        script, sandbox_result = self._run_with_retry(
            script=script,
            source_path=source_path,
            output_path=jsonl_path,
            inspection=inspection,
        )
        result.attempts = MAX_RETRY  # actual count tracked inside

        # Save final script
        with open(script_path, "w") as f:
            f.write(script)

        if not sandbox_result.success or sandbox_result.output_rows == 0:
            result.status = "failed"
            result.errors.append("Script execution failed after retries")
            result.completed_at = datetime.now().isoformat()
            click.echo(f"    [FAIL] Could not convert data. Script saved at: {script_path}")
            click.echo(f"    Inspect and fix manually, then re-run with --script {script_path}")
            return result

        result.raw_jsonl_path = jsonl_path
        click.echo(f"    Converted {sandbox_result.output_rows} rows → {jsonl_path}")

        # ── Step 6: Clean ─────────────────────────────────────
        click.echo(f"\n  [5/7] Cleaning (dedup, bad rows, normalize)...")
        from autotrainer.managers.data_pipeline import DataPipeline
        dp = DataPipeline(cache_dir=self.work_dir)

        cleaned_path = os.path.join(ds_dir, f"cleaned_{name}.jsonl")
        stats = dp.clean(jsonl_path, cleaned_path)
        result.clean_stats = stats
        result.cleaned_path = cleaned_path
        click.echo(
            f"    {stats['input_lines']} in → "
            f"dupes={stats['duplicates']}, bad={stats['json_errors']}, "
            f"empty={stats['empty_content']} → {stats['output_lines']} out"
        )

        if stats["output_lines"] == 0:
            result.status = "failed"
            result.errors.append("Zero rows after cleaning")
            result.completed_at = datetime.now().isoformat()
            return result

        # ── Step 7: Profile ────────────────────────────────────
        click.echo(f"\n  [6/7] Profiling...")
        prof = dp.profile(cleaned_path)
        result.profile = prof
        tl = prof.get("text_lengths", {})
        click.echo(f"    {prof.get('num_samples', 0)} samples, {prof.get('size_mb', 0)}MB")
        if tl:
            click.echo(f"    text: avg={tl.get('avg', 0)} p95={tl.get('p95', 0)} max={tl.get('max', 0)}")
        if prof.get("sample_preview"):
            preview = json.dumps(prof["sample_preview"][0], ensure_ascii=False)
            click.echo(f"    preview: {preview[:250]}")

        # Update context with data profile
        self.context.set_data_profile(prof)

        # ── Step 8: Split ──────────────────────────────────────
        click.echo(f"\n  [7/7] Splitting 90/5/5...")
        split_r = dp.split(cleaned_path)
        result.split = split_r
        click.echo(
            f"    train={split_r['train']['count']} → {split_r['train']['path']}\n"
            f"    val  ={split_r['val']['count']}   → {split_r['val']['path']}\n"
            f"    test ={split_r['test']['count']}  → {split_r['test']['path']}"
        )

        result.status = "completed"
        result.completed_at = datetime.now().isoformat()
        return result

    # ══════════════════════════════════════════════════════════
    # Script generation
    # ══════════════════════════════════════════════════════════

    def _generate_script(
        self,
        inspection_summary: str,
        source_path: str,
        inspection: InspectionResult,
    ) -> str:
        """Call data-inspect skill to get a conversion script."""
        handler = self._get_inspect_handler()

        if not self.llm:
            # No LLM: return a passthrough script for already-erniekit data
            return self._fallback_passthrough_script()

        try:
            result = asyncio.run(handler.generate_script(inspection_summary, source_path))
            script = result.get("script", "")
            if script:
                fmt = result.get("format_identified", "")
                explanation = result.get("explanation", "")
                click.echo(f"    Format identified: {fmt}")
                click.echo(f"    Plan: {explanation[:150]}")
            return script
        except Exception as e:
            click.echo(f"    [WARN] LLM script generation error: {e}")
            return self._fallback_passthrough_script()

    def _fallback_passthrough_script(self) -> str:
        """Passthrough script for data already in erniekit/messages format."""
        return '''
import os, json, shutil

INPUT_PATH = os.environ["INPUT_PATH"]
OUTPUT_PATH = os.environ["OUTPUT_PATH"]

count = 0
with open(INPUT_PATH, "r", errors="replace") as fin, open(OUTPUT_PATH, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                fout.write(json.dumps(data, ensure_ascii=False) + "\\n")
                count += 1
        except json.JSONDecodeError:
            pass

print(f"Converted: {count} samples")
'''

    # ══════════════════════════════════════════════════════════
    # Run → Validate → Fix loop
    # ══════════════════════════════════════════════════════════

    def _run_with_retry(
        self,
        script: str,
        source_path: str,
        output_path: str,
        inspection: InspectionResult,
    ) -> tuple[str, SandboxResult]:
        """Execute script, validate output, fix and retry on failure."""
        handler = self._get_fix_handler()
        last_result = SandboxResult()

        for attempt in range(1, MAX_RETRY + 1):
            click.echo(f"\n  [3/7] Running script (attempt {attempt}/{MAX_RETRY})...")
            t0 = time.time()
            result = self.sandbox.run(script, source_path, output_path)
            elapsed = time.time() - t0

            if result.success and result.output_rows > 0:
                # Validate format
                click.echo(f"    Ran in {elapsed:.1f}s → {result.output_rows} rows")
                validation = self.validator.validate_file(output_path)
                if validation.valid:
                    click.echo(f"    Output format: {validation.format_detected} ✓")
                    return script, result
                else:
                    error = f"Output validation failed:\n" + "\n".join(validation.errors[:5])
                    click.echo(f"    Format validation failed: {validation.errors[:3]}")
            elif result.output_rows == 0 and result.success:
                error = "Script ran successfully but produced 0 output rows"
                click.echo(f"    [WARN] 0 rows output")
            else:
                error = result.error_summary
                click.echo(f"    [FAIL] {error[:200]}")

            last_result = result

            if attempt < MAX_RETRY:
                click.echo(f"  [4/7] Asking LLM to fix script...")
                if not self.llm:
                    click.echo("    No LLM configured, cannot auto-fix")
                    break

                # Trim script for context (avoid huge scripts in context)
                script_for_context = script
                if len(script) > MAX_SCRIPT_TOKENS * 4:
                    script_for_context = script[:MAX_SCRIPT_TOKENS * 4] + "\n# ... (truncated)"

                try:
                    fix_result = asyncio.run(
                        handler.fix_script(
                            original_script=script_for_context,
                            error_summary=error,
                            data_samples=inspection.samples[:5],
                            attempt=attempt,
                        )
                    )
                    new_script = fix_result.get("script", "")
                    if new_script and new_script != script:
                        click.echo(f"    Fix: {fix_result.get('fix_applied', '')[:100]}")
                        script = new_script
                    else:
                        click.echo("    LLM returned same script, stopping retries")
                        break
                except Exception as e:
                    click.echo(f"    Fix failed: {e}")
                    break

        return script, last_result

    # ══════════════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════════════

    def _save_result(self, result: DatasetResult):
        """Append result to the data processing index."""
        index_path = os.path.join(self.work_dir, "data_index.json")
        index = safe_read_json(index_path) or {"datasets": []}

        # Update or append
        existing_ids = {d["source_path"] for d in index.get("datasets", [])}
        if result.source_path in existing_ids:
            for i, d in enumerate(index["datasets"]):
                if d["source_path"] == result.source_path:
                    index["datasets"][i] = result.to_record()
                    break
        else:
            index["datasets"].append(result.to_record())

        atomic_write_json(index_path, index)

    def _print_summary(self, results: list[DatasetResult]):
        click.echo(f"\n{'═' * 65}")
        click.echo(f"  Data Processing Summary")
        click.echo(f"{'═' * 65}")
        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "failed"]
        click.echo(f"  Completed: {len(completed)} / {len(results)}")
        for r in completed:
            samples = r.profile.get("num_samples", 0)
            click.echo(f"    ✓ {r.dataset_name}: {samples} samples → {r.split.get('train', {}).get('path', '')}")
        for r in failed:
            click.echo(f"    ✗ {r.dataset_name}: {r.errors}")
