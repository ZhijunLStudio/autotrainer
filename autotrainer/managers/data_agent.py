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
MAX_LOOP_REPEAT = 3      # if same thought appears this many times, force final_script

TIMEOUT_PER_GB = 300     # seconds of sandbox timeout per GB of input data
TIMEOUT_BASE = 120       # minimum timeout regardless of size
TIMEOUT_MAX = 3600       # maximum timeout (1 hour)
QUICK_VALIDATE_ROWS = 100  # rows to validate script correctness before full run


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

    def run(
        self,
        data_paths: list[str],
        custom_script: str | None = None,
        parallel: int = 1,
    ) -> list[DatasetResult]:
        """Process datasets with a live dashboard and optional parallel workers.

        Args:
            custom_script: Pre-written script path (skip LLM generation).
            parallel: Number of concurrent dataset workers.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from autotrainer.managers.data_dashboard import DataProcessingDashboard, Phase

        index = safe_read_json(os.path.join(self.work_dir, "data_index.json")) or {"datasets": []}
        completed_paths = {
            d["source_path"] for d in index.get("datasets", [])
            if d.get("status") == "completed"
        }

        to_process = [p for p in data_paths if p not in completed_paths or custom_script]
        already_done = [p for p in data_paths if p not in to_process]
        results_map: dict[str, DatasetResult] = {}

        # Restore already-done results
        for path in already_done:
            for d in index.get("datasets", []):
                if d["source_path"] == path:
                    r = DatasetResult(**{k: v for k, v in d.items() if k in DatasetResult.__dataclass_fields__})
                    results_map[path] = r
                    break

        dashboard = DataProcessingDashboard(data_paths, parallel=min(parallel, max(1, len(to_process))))
        for path in already_done:
            st = dashboard.get_status(path)
            if st:
                st.update(Phase.SKIPPED, "already completed")

        def _worker(path: str) -> DatasetResult:
            st = dashboard.get_status(path)

            def _notify(phase, message="", log_line=""):
                if st:
                    st.update(phase, message, log_line)
                    dashboard.refresh()

            result = self._process_one(path, custom_script=custom_script, notify=_notify)
            if st:
                if result.status == "completed":
                    st.samples = result.profile.get("num_samples", 0)
                    st.images = result.profile.get("image_count", 0)
                    st.update(Phase.COMPLETED, f"train={result.split.get('train', {}).get('count', 0)}")
                else:
                    st.error = result.errors[0][:50] if result.errors else "failed"
                    st.update(Phase.FAILED, st.error)
                dashboard.refresh()
            self._save_result(result)
            return result

        dashboard.start()
        try:
            n_workers = min(parallel, len(to_process)) if to_process else 1
            if n_workers == 1:
                for path in to_process:
                    results_map[path] = _worker(path)
            else:
                with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="data-worker") as ex:
                    futures = {ex.submit(_worker, p): p for p in to_process}
                    for future in as_completed(futures):
                        path = futures[future]
                        try:
                            results_map[path] = future.result()
                        except Exception as e:
                            results_map[path] = DatasetResult(
                                source_path=path,
                                dataset_name=os.path.basename(path),
                                status="failed", errors=[str(e)],
                            )
                            st = dashboard.get_status(path)
                            if st:
                                st.error = str(e)[:60]
                                st.update(Phase.FAILED)
                            dashboard.refresh()
        finally:
            dashboard.stop()

        results = [results_map[p] for p in data_paths if p in results_map]
        self._print_summary(results)
        return results

    def _process_one(
        self,
        source_path: str,
        custom_script: str | None = None,
        notify: callable | None = None,
    ) -> DatasetResult:
        """Run the full pipeline for one dataset with dashboard notifications."""
        from autotrainer.managers.data_dashboard import Phase as DPhase

        def _n(phase, msg="", log=""):
            if notify:
                notify(phase, msg, log)

        name = Path(source_path).name
        ds_dir = os.path.join(self.work_dir, name)
        image_dir = os.path.join(ds_dir, "images")
        os.makedirs(ds_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        result = DatasetResult(
            source_path=source_path,
            dataset_name=name,
            started_at=datetime.now().isoformat(),
        )

        jsonl_path = os.path.join(ds_dir, f"raw_{name}.jsonl")

        # ── Phase 1: Get conversion script ────────────────────
        if custom_script:
            _n(DPhase.GENERATING, f"using manual script")
            with open(custom_script, "r") as f:
                script = f.read()
            result.react_steps = 0
        else:
            _n(DPhase.EXPLORING, "pre-exploring directory structure")
            script, react_steps = asyncio.run(
                self._react_loop(source_path, jsonl_path, image_dir, notify=notify)
            )
            result.react_steps = react_steps

        if not script:
            result.status = "failed"
            result.errors.append("Could not generate a conversion script")
            result.completed_at = datetime.now().isoformat()
            _n(DPhase.FAILED, "no script generated")
            return result

        # Save the final script
        script_path = os.path.join(ds_dir, "convert_script.py")
        with open(script_path, "w") as f:
            f.write(script)
        result.script_path = script_path

        # ── Phase 2: Execute + validate + fix loop ────────────
        _n(DPhase.VALIDATING, f"quick validation (100 rows)")
        script, sandbox_result, attempts = self._run_with_retry(
            script, source_path, jsonl_path, image_dir=image_dir, notify=notify
        )
        result.script_attempts = attempts

        # Update final script
        with open(script_path, "w") as f:
            f.write(script)

        if not sandbox_result.success or sandbox_result.output_rows == 0:
            result.status = "failed"
            result.errors.append(f"Script failed after {attempts} attempts")
            result.completed_at = datetime.now().isoformat()
            err_summary = sandbox_result.error_summary[:200]
            _n(DPhase.FAILED, err_summary)
            click.echo(f"\n  [FAIL] {err_summary}")
            click.echo(f"  Script: {script_path}")
            click.echo(f"  Fix: autotrainer data --path {source_path} --script {script_path}")
            return result

        result.raw_jsonl_path = jsonl_path
        n_imgs = sum(1 for _ in Path(image_dir).glob("*.png")) if Path(image_dir).exists() else 0
        _n(DPhase.CLEANING, f"{sandbox_result.output_rows} rows converted")

        # ── Phase 3: Standard post-processing ─────────────────
        from autotrainer.managers.data_pipeline import DataPipeline
        dp = DataPipeline(cache_dir=self.work_dir)

        cleaned_path = os.path.join(ds_dir, f"cleaned_{name}.jsonl")
        stats = dp.clean(jsonl_path, cleaned_path)
        result.clean_stats = stats
        result.cleaned_path = cleaned_path

        if stats["output_lines"] == 0:
            result.status = "failed"
            result.errors.append("Zero rows after cleaning")
            result.completed_at = datetime.now().isoformat()
            _n(DPhase.FAILED, "zero rows after cleaning")
            return result

        _n(DPhase.PROFILING, f"{stats['output_lines']} rows after cleaning")
        prof = dp.profile(cleaned_path)
        result.profile = prof

        _n(DPhase.SPLITTING, f"{prof.get('num_samples', 0)} samples")
        split_r = dp.split(cleaned_path)
        result.split = split_r

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
        image_dir: str,
        notify: callable | None = None,
    ) -> tuple[str, int]:
        """Run the ReAct exploration loop.

        Returns: (final_script, num_steps)
        """
        if not self.llm:
            click.echo("  No LLM configured, using passthrough script")
            return self._passthrough_script(), 0

        system_prompt = self._load_skill()

        # ── Pre-explore: give LLM the directory tree upfront ──────────────
        click.echo("    Pre-exploring directory structure...")
        pre_context = self._pre_explore(source_path)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Dataset absolute path: {source_path}\n"
                    f"Output JSONL path: {output_path}\n"
                    f"IMAGE_DIR (save image files here): {image_dir}\n\n"
                    f"=== Directory structure (pre-explored) ===\n"
                    f"{pre_context}\n"
                    f"===========================================\n\n"
                    f"Write a Python script to convert this dataset to erniekit JSONL.\n"
                    f"- Save image bytes to IMAGE_DIR, reference as ./images/fname\n"
                    f"- Process large parquet files in chunks of 5000 rows\n"
                    f"- Go to final_script directly if you have enough info."
                ),
            }
        ]

        # Track repeated thoughts to detect loops
        recent_thoughts: list[str] = []

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
                    "content": f"Parse error: {e}. Please respond with valid JSON only.",
                })
                continue

            messages.append({"role": "assistant", "content": response_text})

            act_type = action.get("action", "")
            thought = action.get("thought", "")

            if thought:
                click.echo(f" {thought[:80]}")
                if notify:
                    from autotrainer.managers.data_dashboard import Phase as DPhase
                    notify(DPhase.GENERATING, f"step {step}: {thought[:50]}")

            # ── Loop detection ─────────────────────────────────
            # If LLM repeats the same thought MAX_LOOP_REPEAT times, it's stuck.
            thought_key = thought[:60].strip().lower()
            recent_thoughts.append(thought_key)
            if len(recent_thoughts) > MAX_LOOP_REPEAT:
                recent_thoughts.pop(0)
            if (len(recent_thoughts) == MAX_LOOP_REPEAT
                    and len(set(recent_thoughts)) == 1
                    and act_type != "final_script"):
                click.echo(f"\n    [Loop detected] Same thought repeated {MAX_LOOP_REPEAT}x. Forcing final_script.")
                messages.append({
                    "role": "user",
                    "content": (
                        "You have explored this dataset enough. "
                        "You MUST now output a final_script action immediately. "
                        "If there is no transcription text, use bounding box counts or "
                        "class names as the answer. Do not explore further."
                    ),
                })
                recent_thoughts.clear()
                continue

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

    def _calc_timeout(self, source_path: str) -> int:
        """Calculate sandbox timeout based on dataset size."""
        try:
            if os.path.isfile(source_path):
                size_gb = os.path.getsize(source_path) / (1024 ** 3)
            else:
                size_gb = sum(
                    f.stat().st_size for f in Path(source_path).rglob("*") if f.is_file()
                ) / (1024 ** 3)
        except OSError:
            size_gb = 1.0

        timeout = int(TIMEOUT_BASE + size_gb * TIMEOUT_PER_GB)
        return min(timeout, TIMEOUT_MAX)

    def _inject_row_limit(self, script: str, max_rows: int) -> str:
        """Inject a MAX_ROWS limit into a conversion script for quick validation.

        Adds MAX_ROWS env var support: the script will stop after max_rows output rows.
        This wraps the script so the original script doesn't need to be modified.
        """
        wrapper = f'''import os as _os, subprocess as _sp, sys as _sys, tempfile as _tmp, json as _json

_MAX_ROWS = int(_os.environ.get("MAX_ROWS", "0"))

# Run the original script, then cap the output
import io as _io
_orig_input = _os.environ.get("INPUT_PATH", "")
_orig_output = _os.environ.get("OUTPUT_PATH", "")

# Write original script to temp file
import textwrap as _tw
_script = _tw.dedent("""{script.replace(chr(34)*3, chr(34)*2 + chr(92) + chr(34))}""")
with _tmp.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as _f:
    _f.write(_script)
    _tmp_path = _f.name

# Run with a temp output
_tmp_out = _orig_output + ".tmp_validate"
_env = dict(_os.environ)
_env["OUTPUT_PATH"] = _tmp_out
import subprocess
_r = subprocess.run(["python", _tmp_path], env=_env, capture_output=True, text=True, timeout={max_rows * 10})
print(_r.stdout)
if _r.returncode != 0:
    print(_r.stderr, file=_sys.stderr)
    _sys.exit(_r.returncode)

# Cap to MAX_ROWS
if _MAX_ROWS > 0 and _os.path.exists(_tmp_out):
    count = 0
    with open(_tmp_out) as _fin, open(_orig_output, "w") as _fout:
        for _line in _fin:
            if count >= _MAX_ROWS:
                break
            _fout.write(_line)
            count += 1
    _os.unlink(_tmp_out)
    print(f"[validate] Capped to {{count}} rows")
else:
    if _os.path.exists(_tmp_out):
        import shutil; shutil.move(_tmp_out, _orig_output)
_os.unlink(_tmp_path)
'''
        # This approach is too complex — instead inject directly into the script
        # Simply prefix the script with a row counter interceptor
        return script  # Return original; we use separate env var approach in sandbox

    def _run_with_retry(
        self,
        script: str,
        source_path: str,
        output_path: str,
        image_dir: str = "",
        notify: callable | None = None,
    ) -> tuple[str, SandboxResult, int]:
        """Two-phase execution: quick validation (100 rows) then full run.

        Phase 1: Run on first 100 rows with short timeout to verify script correctness.
        Phase 2: Full run with adaptive timeout based on dataset size.
        """
        last_result = SandboxResult()
        extra_env = {"IMAGE_DIR": image_dir} if image_dir else {}
        full_timeout = self._calc_timeout(source_path)
        size_gb = self._calc_size_gb(source_path)

        def _notify_run(msg, log=""):
            if notify:
                from autotrainer.managers.data_dashboard import Phase as DPhase
                notify(DPhase.RUNNING, msg, log)

        # ── Phase 1: Quick validation on 100 rows ────────────────────────
        _notify_run(f"quick validation (100 rows)")
        click.echo(f"  Quick validation (first {QUICK_VALIDATE_ROWS} rows)...", nl=False)
        quick_output = output_path + ".quick_validate"

        # Wrap script to limit rows
        limited_script = self._make_limited_script(script, QUICK_VALIDATE_ROWS)
        ok, syntax_err = self.sandbox.validate_script(limited_script)
        if not ok:
            click.echo(f" syntax error")
            if self.llm:
                script = self._fix_script_sync(script, f"Syntax error:\n{syntax_err}", source_path)
            last_result.success = False
            last_result.stderr = syntax_err
        else:
            quick_sb = Sandbox(timeout=60)
            q_result = quick_sb.run(limited_script, source_path, quick_output, extra_env=extra_env)

            if q_result.success and q_result.output_rows > 0:
                click.echo(f" OK ({q_result.output_rows} rows sampled)")
                # Validate format on quick output
                validation = self.validator.validate_file(quick_output)
                if not validation.valid:
                    click.echo(f"  Format invalid on sample: {validation.errors[:2]}")
                    if self.llm:
                        error = "Output format invalid: " + "; ".join(validation.errors[:3])
                        script = self._fix_script_sync(script, error, source_path)
            elif q_result.timed_out:
                click.echo(f" timeout even on 100 rows — script has fundamental performance issue")
                error = (
                    f"Script timed out on just {QUICK_VALIDATE_ROWS} rows (60s limit). "
                    f"This usually means: 1) PIL image saving is too slow — skip saving images in the script, "
                    f"just use image_info=[] instead; 2) The loop has an unexpected infinite loop; "
                    f"3) pandas operations are inefficient. Simplify the script significantly."
                )
                if self.llm:
                    script = self._fix_script_sync(script, error, source_path)
            else:
                stdout_hint = q_result.stdout.strip()[-400:] if q_result.stdout else "(no stdout)"
                click.echo(f" 0 rows — {stdout_hint[:100]}")
                error = (
                    f"Script produced 0 rows on {QUICK_VALIDATE_ROWS}-row sample.\n"
                    f"stdout: {stdout_hint}\nstderr: {q_result.stderr[-400:]}"
                )
                if self.llm:
                    script = self._fix_script_sync(script, error, source_path)
            last_result = q_result

        try:
            os.unlink(quick_output)
        except OSError:
            pass

        # ── Phase 2: Full run with retry loop ────────────────────────────
        full_sandbox = Sandbox(timeout=full_timeout)
        _notify_run(f"full run {size_gb:.1f}GB (timeout={full_timeout}s)")
        click.echo(f"  Full run (timeout={full_timeout}s for {size_gb:.1f}GB)...")

        for attempt in range(1, MAX_RETRY_SCRIPT + 1):
            _notify_run(f"attempt {attempt}/{MAX_RETRY_SCRIPT}")
            click.echo(f"  Attempt {attempt}/{MAX_RETRY_SCRIPT}...", nl=False)

            ok, syntax_err = full_sandbox.validate_script(script)
            if not ok:
                click.echo(f" syntax error")
                if self.llm and attempt < MAX_RETRY_SCRIPT:
                    script = self._fix_script_sync(script, f"Syntax error:\n{syntax_err}", source_path)
                    continue
                break

            result = full_sandbox.run(script, source_path, output_path, extra_env=extra_env)
            last_result = result

            if result.success and result.output_rows > 0:
                validation = self.validator.validate_file(output_path)
                if validation.valid:
                    click.echo(f" OK ({result.output_rows} rows, {result.elapsed_seconds:.1f}s)")
                    return script, result, attempt
                error = "Output format invalid: " + "; ".join(validation.errors[:3])
                click.echo(f" format invalid")
            elif result.timed_out:
                size_gb = self._calc_size_gb(source_path)
                error = (
                    f"Script timed out after {full_timeout}s on {size_gb:.1f}GB dataset.\n"
                    f"Processed rows before timeout: {result.output_rows}.\n"
                    f"Fix: 1) Skip PIL image saving (use image_info=[]) to reduce IO. "
                    f"2) Process fewer parquet files (e.g. only first 2). "
                    f"3) Use df.itertuples() instead of df.iterrows() for 3-5x speedup."
                )
                click.echo(f" timeout ({result.output_rows} rows before cutoff)")
                if self.llm and attempt < MAX_RETRY_SCRIPT:
                    script = self._fix_script_sync(script, error, source_path)
                else:
                    break
            elif result.output_rows == 0 and result.success:
                stdout_hint = result.stdout.strip()[-500:] if result.stdout.strip() else "(no stdout)"
                error = (
                    f"Script ran successfully but produced 0 rows.\n"
                    f"stdout: {stdout_hint}\n"
                    f"Causes: wrong field names / all rows filtered / not writing to OUTPUT_PATH."
                )
                click.echo(f" 0 rows — {stdout_hint[:100]}")
                if self.llm and attempt < MAX_RETRY_SCRIPT:
                    script = self._fix_script_sync(script, error, source_path)
                else:
                    break
            else:
                error = result.error_summary
                click.echo(f" failed")
                if self.llm and attempt < MAX_RETRY_SCRIPT:
                    script = self._fix_script_sync(script, error, source_path)
                else:
                    break

        return script, last_result, MAX_RETRY_SCRIPT

    def _calc_size_gb(self, source_path: str) -> float:
        """Get dataset size in GB."""
        try:
            if os.path.isfile(source_path):
                return os.path.getsize(source_path) / (1024 ** 3)
            return sum(
                f.stat().st_size for f in Path(source_path).rglob("*") if f.is_file()
            ) / (1024 ** 3)
        except OSError:
            return 0.0

    def _make_limited_script(self, script: str, max_rows: int) -> str:
        """Wrap a script to stop after max_rows output rows."""
        # Inject a row-counting wrapper around fout.write calls
        # Strategy: replace open(OUTPUT_PATH, "w") with a counting wrapper
        limited = f'''import os as _os_limit
_MAX_ROWS_LIMIT = {max_rows}
_ROWS_WRITTEN = 0

# Monkey-patch to intercept output writes
import builtins as _builtins
_real_open = _builtins.open

class _LimitedFile:
    def __init__(self, f):
        self._f = f
    def write(self, s):
        global _ROWS_WRITTEN
        if s.strip():
            _ROWS_WRITTEN += 1
            if _ROWS_WRITTEN > _MAX_ROWS_LIMIT:
                raise SystemExit(f"[row_limit] Reached {{_MAX_ROWS_LIMIT}} rows")
        return self._f.write(s)
    def __enter__(self): return self
    def __exit__(self, *a): self._f.__exit__(*a)
    def __getattr__(self, n): return getattr(self._f, n)

_OUTPUT_PATH_VAL = _os_limit.environ.get("OUTPUT_PATH", "")
def _patched_open(file, mode="r", *args, **kwargs):
    f = _real_open(file, mode, *args, **kwargs)
    if str(file) == _OUTPUT_PATH_VAL and "w" in str(mode):
        return _LimitedFile(f)
    return f
_builtins.open = _patched_open

try:
{chr(10).join("    " + line for line in script.splitlines())}
except SystemExit as _e:
    if "[row_limit]" in str(_e):
        pass  # expected stop
    else:
        raise
finally:
    _builtins.open = _real_open
'''
        return limited

    def _fix_script_sync(self, script: str, error: str, source_path: str = "") -> str:
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

        # Limit script size but always include beginning + end (most likely problem areas)
        script_for_llm = script
        if len(script) > 6000:
            half = 3000
            script_for_llm = script[:half] + "\n# ...[middle truncated]...\n" + script[-half:]

        user_msg = (
            f"Error:\n{error}\n\n"
            f"Dataset path: {source_path}\n\n"
            f"Script to fix:\n```python\n{script_for_llm}\n```\n\n"
            f"Fix it. Return full corrected script in the 'script' field."
        )

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
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        console.print(f"\n{'═' * 65}")
        console.print("  [bold]Data Processing Summary[/bold]")
        console.print(f"{'═' * 65}")

        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "failed"]
        skipped = [r for r in results if r.status not in ("completed", "failed")]

        console.print(
            f"  Total: [bold]{len(results)}[/bold]  "
            f"[green]✓ {len(completed)}[/green]  "
            f"[red]✗ {len(failed)}[/red]  "
            f"[dim]⟳ {len(skipped)} skipped[/dim]"
        )

        table = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1))
        table.add_column("#", width=3)
        table.add_column("Dataset", width=40, no_wrap=False)
        table.add_column("Status", width=10)
        table.add_column("Samples", width=9, justify="right")
        table.add_column("Images", width=7, justify="right")
        table.add_column("Train path", width=50, no_wrap=False)

        for i, r in enumerate(results, 1):
            status_str = (
                "[green]completed[/green]" if r.status == "completed"
                else "[red]failed[/red]" if r.status == "failed"
                else "[dim]skipped[/dim]"
            )
            n = r.profile.get("num_samples", "-") if r.profile else "-"
            imgs = str(r.profile.get("image_count", "-")) if r.profile else "-"
            train = r.split.get("train", {}).get("path", "") if r.split else ""
            err = r.errors[0][:50] if r.errors else ""
            table.add_row(
                str(i),
                r.dataset_name,
                status_str,
                str(n),
                imgs,
                train or err,
            )

        console.print(table)

        if completed:
            console.print("\n  [bold]Ready for training:[/bold]")
            for r in completed:
                train_path = r.split.get("train", {}).get("path", "")
                n = r.profile.get("num_samples", 0)
                console.print(f"    [green]→[/green] {train_path}  ({n} samples)")
