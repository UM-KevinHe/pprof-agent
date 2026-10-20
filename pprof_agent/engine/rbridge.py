"""The R dispatch layer: find Rscript, hand it a JSON payload, read JSON back.

Adapted from chapter II's `bregsurv_agent/rbridge.py`. Three choices carried over because each was
paid for there: `--no-save --no-restore --no-init-file` rather than `--vanilla` (which drops
`R_LIBS_USER`, so R cannot find the user library on Great Lakes); `stdin=DEVNULL` (an inherited stdin
makes R wait for input and presents as a hang); files rather than pipes (R's output can exceed pipe
buffers, and the files are what `repro.R` replays). One change: a failure RAISES `RError` here, with
the script's JSON error object attached, because every caller is the harness and a refused step must
stop the loop rather than be answered.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

_HERE = Path(__file__).resolve().parent
R_SCRIPTS = Path(os.environ.get("PPROF_AGENT_R_SCRIPTS", str(_HERE.parent.parent / "r_scripts")))


class RError(RuntimeError):
    """An R script failed; `info` is its JSON error object (or what could be recovered)."""

    def __init__(self, info: Dict[str, Any]):
        self.info = info
        super().__init__(f"{info.get('where', 'R')}: {info.get('message', 'unknown error')}")


def find_rscript() -> str:
    override = os.environ.get("PPROF_AGENT_RSCRIPT")
    if override and Path(override).exists():
        return override
    on_path = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if on_path:
        return on_path
    for cand in ("/usr/local/bin/Rscript", "/usr/bin/Rscript", "/opt/homebrew/bin/Rscript",
                 "/Library/Frameworks/R.framework/Resources/bin/Rscript"):
        if Path(cand).exists():
            return cand
    raise FileNotFoundError("Rscript not found; load the R module or set PPROF_AGENT_RSCRIPT")


def run_r(script: str, payload: Dict[str, Any], payload_path: Path, out_path: Path,
          timeout_s: int = 3600, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run `r_scripts/<script>` with `payload` written to `payload_path`; return the parsed output."""
    script_path = R_SCRIPTS / script
    if not script_path.exists():
        raise RError({"where": script, "message": f"R script not found: {script_path}", "class": "FileNotFoundError"})
    payload_path = Path(payload_path)
    out_path = Path(out_path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if out_path.exists():
        out_path.unlink()
    env = os.environ.copy()
    if env.get("SLURM_CPUS_PER_TASK"):
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(k, env["SLURM_CPUS_PER_TASK"])
    if extra_env:
        env.update(extra_env)
    cmd = [find_rscript(), "--no-save", "--no-restore", "--no-init-file", str(script_path),
           str(payload_path), str(out_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, encoding="utf-8",
                              errors="replace", stdin=subprocess.DEVNULL, env=env)
    except subprocess.TimeoutExpired as exc:
        raise RError({"where": script, "message": f"Rscript timed out after {timeout_s}s", "class": "TimeoutExpired"}) from exc
    stderr_tail = (proc.stderr or "").strip()[-3000:]
    if out_path.exists():
        try:
            out = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RError({"where": script, "message": f"R output was not valid JSON: {exc}", "class": "JSONDecodeError",
                          "stderr": stderr_tail}) from exc
        if isinstance(out, dict) and out.get("status") == "error":
            out.setdefault("where", script)
            out["stderr"] = stderr_tail
            raise RError(out)
        if isinstance(out, dict):
            out["_stderr"] = stderr_tail
        return out
    raise RError({"where": script, "message": stderr_tail or "Rscript exited without producing output",
                  "class": "RscriptCrash", "returncode": proc.returncode, "stderr": stderr_tail})


def r_available() -> bool:
    try:
        find_rscript()
        return True
    except FileNotFoundError:
        return False
