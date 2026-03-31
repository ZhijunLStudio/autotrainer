"""DataAgent — ReAct-style agent for data processing.

The agent works in a think → act → observe loop:
1. LLM thinks about what to do next
2. Executes a shell command or Python snippet
3. Observes the output
4. Repeats until it produces a final conversion script

This is fundamentally different from hardcoded inspection:
- The agent can handle any data format
- It uses the same tools a human would (ls, head, python, etc.)
- It adapts based on what it sees
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import click

from autotrainer.context.store import ContextStore
from autotrainer.managers.sandbox import Sandbox, SandboxResult
from autotrainer.pf_integration.dataset_validator import DatasetValidator
from autotrainer.utils.file_utils import atomic_write_json, safe_read_json


MAX_REACT_STEPS = 15     # max exploration steps per dataset
MAX_RETRY_SCRIPT = 3     # max times to retry fixing the script
MAX_CMD_OUTPUT = 3000    # truncate command output to this length for context


@dataclass
class DatasetResult:
    """Full result for one processed dataset."""

    source_path: str = ""
    dataset_name: str = ""
    script_path: str = ""
    raw_jsonl_path: str = ""
    cleaned_path: str = ""
    split: dict = field(default_factory=dict)
    profile: dict = field(default_factory=dict)
    clean_stats: dict = field(default_factory=dict)
    react_steps: int = 0
    script_attempts: int = 0
    status: str = "pending"
    errors: list[str] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    def to_record(self) -> dict:
        return {
            "source_path": self.source_path,
            "dataset_name": self.dataset_name,
            "script_path": self.script_path,
            "raw_jsonl_path": self.raw_jsonl_path,
            "cleaned_path": self.cleaned_path,
            "split": self.split,
            "profile": self.profile,
            "clean_stats": self.clean_stats,
            "react_steps": self.react_steps,
            "script_attempts": self.script_attempts,
            "status": self.status,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class DataAgent:
    """ReAct agent for data processing."""

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
        self.sandbox = Sandbox(timeout=sandbox_timeout)
        self.validator = DatasetValidator()
        self._skill_md: str | None = None
        self._user_cwd = os.getcwd()    # capture cwd at creation for shell commands
        os.makedirs(work_dir, exist_ok=True)

    def _load_skill(self) -> str:
        if not self._skill_md:
            skill_path = os.path.join(
                os.path.dirname(__file__), "..", "skills", "data_inspect", "SKILL.md"
            )
            with open(skill_path, "r") as f:
                self._skill_md = f.read()
        return self._skill_md

    # ══════════════════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════════════════

    def run(self, data_paths: list[str]) -> list[DatasetResult]:
        """Process a list of dataset paths."""
        results = []
        for path in data_paths:
            click.echo(f"\n{'═' * 65}")
            click.echo(f"  Dataset: {os.path.basename(path)}")
            click.echo(f"  Path:    {path}")
            click.echo(f"{'═' * 65}")
            result = self._process_one(path)
            results.append(result)
            self._save_result(result)
        self._print_summary(results)
        return results

    def _process_one(self, source_path: str) -> DatasetResult:
        """Run the full ReAct pipeline for one dataset."""
        name = Path(source_path).name
        ds_dir = os.path.join(self.work_dir, name)
        os.makedirs(ds_dir, exist_ok=True)

        result = DatasetResult(
            source_path=source_path,
            dataset_name=name,
            started_at=datetime.now().isoformat(),
        )

        # ── Phase 1: ReAct exploration + script generation ────
        click.echo(f"\n  [Phase 1] Exploring data and generating conversion script...")

        jsonl_path = os.path.join(ds_dir, f"raw_{name}.jsonl")
        script, react_steps = asyncio.run(
            self._react_loop(source_path, jsonl_path, ds_dir)
        )
        result.react_steps = react_steps

        if not script:
            result.status = "failed"
            result.errors.append("Could not generate a conversion script")
            result.completed_at = datetime.now().isoformat()
            return result

        # Save the final script
        script_path = os.path.join(ds_dir, "convert_script.py")
        with open(script_path, "w") as f:
            f.write(script)
        result.script_path = script_path

        # ── Phase 2: Execute + validate + fix loop ────────────
        click.echo(f"\n  [Phase 2] Running conversion script...")
        script, sandbox_result, attempts = self._run_with_retry(
            script, source_path, jsonl_path
        )
        result.script_attempts = attempts

        # Update final script
        with open(script_path, "w") as f:
            f.write(script)

        if not sandbox_result.success or sandbox_result.output_rows == 0:
            result.status = "failed"
            result.errors.append(f"Script failed after {attempts} attempts")
            result.completed_at = datetime.now().isoformat()
            click.echo(f"  [FAIL] Check script: {script_path}")
            click.echo(f"  [FAIL] Error: {sandbox_result.error_summary[:300]}")
            return result

        result.raw_jsonl_path = jsonl_path
        click.echo(f"  Converted {sandbox_result.output_rows} rows → {jsonl_path}")

        # ── Phase 3: Standard post-processing ─────────────────
        from autotrainer.managers.data_pipeline import DataPipeline
        dp = DataPipeline(cache_dir=self.work_dir)

        click.echo(f"\n  [Phase 3] Cleaning (dedup, bad rows)...")
        cleaned_path = os.path.join(ds_dir, f"cleaned_{name}.jsonl")
        stats = dp.clean(jsonl_path, cleaned_path)
        result.clean_stats = stats
        result.cleaned_path = cleaned_path
        click.echo(
            f"    {stats['input_lines']} → "
            f"dupes={stats['duplicates']} bad={stats['json_errors']} "
            f"empty={stats['empty_content']} → {stats['output_lines']} out"
        )

        if stats["output_lines"] == 0:
            result.status = "failed"
            result.errors.append("Zero rows after cleaning")
            result.completed_at = datetime.now().isoformat()
            return result

        click.echo(f"\n  [Phase 4] Profiling...")
        prof = dp.profile(cleaned_path)
        result.profile = prof
        tl = prof.get("text_lengths", {})
        click.echo(f"    {prof.get('num_samples', 0)} samples, {prof.get('size_mb', 0)}MB")
        if tl:
            click.echo(f"    text: avg={tl.get('avg', 0)} p95={tl.get('p95', 0)}")
        if prof.get("sample_preview"):
            preview = json.dumps(prof["sample_preview"][0], ensure_ascii=False)
            click.echo(f"    preview: {preview[:200]}")

        click.echo(f"\n  [Phase 5] Splitting 90/5/5...")
        split_r = dp.split(cleaned_path)
        result.split = split_r
        click.echo(
            f"    train={split_r['train']['count']} → {split_r['train']['path']}\n"
            f"    val  ={split_r['val']['count']}\n"
            f"    test ={split_r['test']['count']}"
        )

        result.status = "completed"
        result.completed_at = datetime.now().isoformat()
        return result

    # ══════════════════════════════════════════════════════════
    # ReAct loop — think → act → observe
    # ══════════════════════════════════════════════════════════

    async def _react_loop(
        self,
        source_path: str,
        output_path: str,
        work_dir: str,
    ) -> tuple[str, int]:
        """Run the ReAct exploration loop.

        Returns: (final_script, num_steps)
        """
        if not self.llm:
            click.echo("  No LLM configured, using passthrough script")
            return self._passthrough_script(), 0

        system_prompt = self._load_skill()

        # ── Pre-explore: give LLM the directory tree upfront ──────────────
        # This avoids wasting steps on "where is the data?" exploration.
        click.echo("    Pre-exploring directory structure...")
        pre_context = self._pre_explore(source_path)

        # Seed the conversation with real file information
        messages = [
            {
                "role": "user",
                "content": (
                    f"Dataset absolute path: {source_path}\n"
                    f"Output JSONL path: {output_path}\n\n"
                    f"=== Directory structure (pre-explored) ===\n"
                    f"{pre_context}\n"
                    f"===========================================\n\n"
                    f"Based on the above, write a Python script to convert this dataset "
                    f"to erniekit JSONL format.\n"
                    f"If you need more detail (e.g., to read a parquet schema or see "
                    f"first few rows), use shell/python actions. "
                    f"Otherwise, go directly to final_script."
                ),
            }
        ]

        for step in range(1, MAX_REACT_STEPS + 1):
            click.echo(f"    Step {step}/{MAX_REACT_STEPS}...", nl=False)

            # Ask LLM for next action
            try:
                response_text = await self.llm.complete_messages(
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    max_tokens=4096,
                    temperature=0.0,
                )
            except Exception as e:
                click.echo(f" LLM error: {e}")
                return "", step

            # Parse the action
            try:
                action = self._parse_action(response_text)
            except Exception as e:
                click.echo(f" parse error: {e}")
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Parse error: {e}. Please respond with valid JSON.",
                })
                continue

            # Append to history
            messages.append({"role": "assistant", "content": response_text})

            act_type = action.get("action", "")
            thought = action.get("thought", "")

            if thought:
                click.echo(f" {thought[:80]}")

            # ── final_script ──────────────────────────────────
            if act_type == "final_script":
                script = action.get("script", "")
                if script:
                    click.echo(f"    → Script ready ({len(script)} chars)")
                    return script, step
                else:
                    messages.append({
                        "role": "user",
                        "content": "The 'script' field was empty. Please provide the complete Python script.",
                    })
                    continue

            # ── shell ─────────────────────────────────────────
            elif act_type == "shell":
                cmd = action.get("command", "")
                click.echo(f" $ {cmd[:60]}")
                output = self._run_shell(cmd)
                truncated = output[:MAX_CMD_OUTPUT]
                if len(output) > MAX_CMD_OUTPUT:
                    truncated += f"\n...[truncated, {len(output)} total chars]"
                messages.append({
                    "role": "user",
                    "content": f"Shell output:\n```\n{truncated}\n```",
                })

            # ── python ────────────────────────────────────────
            elif act_type == "python":
                code = action.get("code", "")
                click.echo(f" python snippet")
                output = self._run_python_snippet(code, source_path)
                truncated = output[:MAX_CMD_OUTPUT]
                if len(output) > MAX_CMD_OUTPUT:
                    truncated += f"\n...[truncated, {len(output)} total chars]"
                messages.append({
                    "role": "user",
                    "content": f"Python output:\n```\n{truncated}\n```",
                })

            else:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Unknown action '{act_type}'. "
                        f"Use one of: shell, python, final_script."
                    ),
                })

        click.echo(f"    Reached max steps ({MAX_REACT_STEPS})")
        return "", MAX_REACT_STEPS

    def _parse_action(self, text: str) -> dict:
        """Parse LLM response as JSON action."""
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        return json.loads(text)

    def _pre_explore(self, source_path: str) -> str:
        """Pre-explore a dataset directory and return a text summary for LLM context.

        Runs real shell/python commands so the LLM starts with actual information.
        """
        lines = []

        # 1. Directory listing with sizes
        ls_out = self._run_shell(f"ls -lhR {source_path} | head -80")
        lines.append(f"$ ls -lhR {source_path} | head -80\n{ls_out}")

        # 2. File type summary
        find_out = self._run_shell(
            f"find {source_path} -type f | sed 's|.*\\.||' | sort | uniq -c | sort -rn | head -20"
        )
        lines.append(f"\nFile extensions summary:\n{find_out}")

        # 3. Sample first file of each interesting type
        for ext in ("parquet", "jsonl", "json", "csv", "tsv", "xml"):
            find_file = self._run_shell(
                f"find {source_path} -name '*.{ext}' -type f | head -1"
            ).strip()
            if find_file and find_file != "(no output)":
                if ext == "parquet":
                    sample = self._run_python_snippet(
                        f"import pandas as pd\ndf = pd.read_parquet('{find_file}')\n"
                        f"print('Shape:', df.shape)\nprint('Columns:', list(df.columns))\n"
                        f"print('Dtypes:\\n', df.dtypes)\nprint('\\nFirst 2 rows:')\n"
                        f"print(df.head(2).to_string(max_colwidth=80))",
                        source_path,
                    )
                    lines.append(f"\nParquet sample ({find_file}):\n{sample}")
                elif ext in ("jsonl", "json"):
                    head = self._run_shell(f"head -3 {find_file}")
                    lines.append(f"\n{ext.upper()} sample ({find_file}):\n{head}")
                elif ext in ("csv", "tsv"):
                    head = self._run_shell(f"head -5 {find_file}")
                    lines.append(f"\n{ext.upper()} sample ({find_file}):\n{head}")
                elif ext == "xml":
                    head = self._run_shell(f"head -30 {find_file}")
                    lines.append(f"\nXML sample ({find_file}):\n{head}")

        summary = "\n".join(lines)
        # Truncate to avoid huge context
        if len(summary) > 8000:
            summary = summary[:8000] + "\n...[pre-explore truncated]"
        return summary

    def _run_shell(self, cmd: str) -> str:
        """Run a shell command safely and return output."""
        blocked = ["rm -rf", "rm -f", "mkfs", "dd if=", "> /dev/", "shutdown", "reboot"]
        for b in blocked:
            if b in cmd:
                return f"[BLOCKED] Command contains '{b}' which is not allowed"

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self._user_cwd,        # run from user's original cwd
            )
            out = result.stdout
            if result.stderr and not result.stdout:
                out = f"[stderr]: {result.stderr[:500]}"
            return out or "(no output)"
        except subprocess.TimeoutExpired:
            return "[TIMEOUT] Command took too long (>60s)"
        except Exception as e:
            return f"[ERROR] {e}"

    def _run_python_snippet(self, code: str, source_path: str) -> str:
        """Run a Python code snippet and return stdout."""
        # Inject INPUT_PATH for convenience
        preamble = f"import os\nos.environ['INPUT_PATH'] = {repr(source_path)}\n"
        full_code = preamble + code

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_code)
            tmp = f.name

        try:
            result = subprocess.run(
                ["python", tmp],
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = result.stdout
            if result.stderr:
                out += f"\n[stderr]: {result.stderr[:800]}"
            return out or "(no output)"
        except subprocess.TimeoutExpired:
            return "[TIMEOUT] Python snippet took too long (>60s)"
        except Exception as e:
            return f"[ERROR] {e}"
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # ══════════════════════════════════════════════════════════
    # Script execution + fix loop
    # ══════════════════════════════════════════════════════════

    def _run_with_retry(
        self,
        script: str,
        source_path: str,
        output_path: str,
    ) -> tuple[str, SandboxResult, int]:
        """Execute script, validate, fix and retry on failure."""
        last_result = SandboxResult()

        for attempt in range(1, MAX_RETRY_SCRIPT + 1):
            click.echo(f"  Attempt {attempt}/{MAX_RETRY_SCRIPT}...", nl=False)

            ok, syntax_err = self.sandbox.validate_script(script)
            if not ok:
                click.echo(f" syntax error")
                if self.llm and attempt < MAX_RETRY_SCRIPT:
                    script = self._fix_script_sync(script, f"Syntax error:\n{syntax_err}")
                    continue
                break

            result = self.sandbox.run(script, source_path, output_path)
            last_result = result

            if result.success and result.output_rows > 0:
                validation = self.validator.validate_file(output_path)
                if validation.valid:
                    click.echo(f" OK ({result.output_rows} rows, {result.elapsed_seconds:.1f}s)")
                    return script, result, attempt
                error = "Output format invalid: " + "; ".join(validation.errors[:3])
                click.echo(f" format invalid")
            elif result.output_rows == 0 and result.success:
                error = "Script ran but produced 0 rows"
                click.echo(f" 0 rows")
            else:
                error = result.error_summary
                click.echo(f" failed")

            if self.llm and attempt < MAX_RETRY_SCRIPT:
                script = self._fix_script_sync(script, error)
            else:
                break

        return script, last_result, MAX_RETRY_SCRIPT

    def _fix_script_sync(self, script: str, error: str) -> str:
        """Synchronous wrapper for LLM script fix."""
        if not self.llm:
            return script

        click.echo(f"  Asking LLM to fix...")

        fix_skill_path = os.path.join(
            os.path.dirname(__file__), "..", "skills", "data_fix", "SKILL.md"
        )
        try:
            with open(fix_skill_path) as f:
                fix_system = f.read()
        except FileNotFoundError:
            fix_system = "Fix the Python script based on the error."

        user_msg = f"Error:\n{error}\n\nScript:\n```python\n{script[:6000]}\n```\n\nFix it."

        try:
            result = asyncio.run(
                self.llm.complete_json(system=fix_system, user=user_msg, max_tokens=4096)
            )
            new_script = result.get("script", "")
            if new_script and new_script != script:
                click.echo(f"    fix: {result.get('fix_applied', '')[:80]}")
                return new_script
        except Exception as e:
            click.echo(f"    fix failed: {e}")

        return script

    def _passthrough_script(self) -> str:
        """Passthrough for already-erniekit data or no-LLM fallback."""
        return '''import os, json

INPUT_PATH = os.environ["INPUT_PATH"]
OUTPUT_PATH = os.environ["OUTPUT_PATH"]

import os as _os
# If INPUT_PATH is a directory, find the first JSONL/JSON file
if _os.path.isdir(INPUT_PATH):
    for root, dirs, files in _os.walk(INPUT_PATH):
        for fn in files:
            ext = fn.lower().rsplit(".", 1)[-1]
            if ext in ("jsonl", "json"):
                INPUT_PATH = _os.path.join(root, fn)
                break
        if not _os.path.isdir(INPUT_PATH):
            break

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
    # Persistence and summary
    # ══════════════════════════════════════════════════════════

    def _save_result(self, result: DatasetResult):
        index_path = os.path.join(self.work_dir, "data_index.json")
        index = safe_read_json(index_path) or {"datasets": []}
        existing = {d["source_path"] for d in index.get("datasets", [])}
        if result.source_path in existing:
            for i, d in enumerate(index["datasets"]):
                if d["source_path"] == result.source_path:
                    index["datasets"][i] = result.to_record()
                    break
        else:
            index["datasets"].append(result.to_record())
        atomic_write_json(index_path, index)

    def _print_summary(self, results: list[DatasetResult]):
        click.echo(f"\n{'═' * 65}")
        click.echo("  Summary")
        click.echo(f"{'═' * 65}")
        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "failed"]
        click.echo(f"  Completed: {len(completed)} / {len(results)}")
        for r in completed:
            n = r.profile.get("num_samples", 0)
            train_path = r.split.get("train", {}).get("path", "")
            click.echo(f"    ✓ {r.dataset_name}: {n} samples → {train_path}")
        for r in failed:
            click.echo(f"    ✗ {r.dataset_name}: {r.errors}")
        if completed:
            click.echo(f"\n  Ready for training:")
            for r in completed:
                click.echo(f"    train: {r.split.get('train', {}).get('path', '')}")
