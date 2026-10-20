"""Export a finished run as one JSON bundle for the web app, and build the app's data directory.

    python -m pprof_agent.export run <run_dir> [--out bundle.json] [--subset-flagged]
    python -m pprof_agent.export site <examples_dir> <site_dir> [--subset-flagged]

A bundle holds everything the app shows: the measure as declared with its sources, the data card, the
admissible set with the rules that fired, every table the engine wrote (records, floats rounded), the
figures as SVG text, the ledger with each step's source and reason, the planner log with the refusals,
the model calls (sizes and timing, never the prompt), the report prose with its references unresolved
plus the reference values, the conversation as it was shown, the flag stability across the computed
alternatives, and -- for a synthetic example -- the data dictionary, the published specification and
the planted truth. `--subset-flagged` breaks down every provider flagged under the reported
configuration (an R call each) so the app can drill into any of them offline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import actions, compare, report, rules
from .state import Session


def _records(path: Path, ndigits: int = 5) -> List[Dict[str, Any]]:
    df = pd.read_csv(path, dtype={"provider": str})
    out: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rec: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, float):
                rec[k] = None if math.isnan(v) else (round(v, ndigits) if abs(v) < 1e6 else v)
            elif hasattr(v, "item"):
                rec[k] = v.item()
            else:
                rec[k] = v
        out.append(rec)
    return out


def _read(path: Path) -> Optional[str]:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _json(path: Path) -> Optional[Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def subset_flagged(session: Session, log=print) -> int:
    """Break down every provider flagged under the reported configuration that has no breakdown yet."""
    from .engine import executor
    from .engine.rbridge import RError
    rec = session.recommended or {}
    df = compare.table_of(session, rec.get("test"), rec.get("reference", "theoretical")) if rec.get("test") else None
    if df is None:
        return 0
    n = 0
    for pid in df.loc[df["status"].isin(["worse", "better"]), "provider"].astype(str):
        if session.has(f"subset:{pid}"):
            continue
        try:
            executor.execute(session, "subset", {"provider": pid}, source="harness", reason="breakdown for the explorer")
            n += 1
        except RError as exc:
            log(f"subset {pid} failed: {exc.info.get('message', '')}")
    if n:
        session.save("trace.json")
    return n


def bundle(run_dir: str, dataset_dir: Optional[str] = None, with_figures: bool = True) -> Dict[str, Any]:
    """The bundle of a finished run on disk."""
    return bundle_of(Session.load(str(run_dir)), dataset_dir, with_figures)


def bundle_of(s: Session, dataset_dir: Optional[str] = None, with_figures: bool = True) -> Dict[str, Any]:
    """The bundle of a session as it stands (a live session included: whatever is in its ledger)."""
    run = Path(s.run_dir)
    spec = s.spec
    tables: Dict[str, List[Dict[str, Any]]] = {}
    for p in sorted((run / "tables").glob("*.csv")) if (run / "tables").exists() else []:
        try:
            tables[p.stem] = _records(p)
        except Exception as exc:  # noqa: BLE001
            tables[p.stem] = [{"error": str(exc)}]
    figures: Dict[str, Dict[str, Any]] = {}
    if with_figures:
        captions = {Path(r.summary.get("path", "")).name: r.summary.get("caption", "") for r in s.results if r.action == "figure" and r.summary.get("path")}
        for p in sorted((run / "figures").glob("*.svg")) if (run / "figures").exists() else []:
            figures[p.name] = {"svg": p.read_text(encoding="utf-8"), "caption": captions.get(p.name, "")}
    refs = report.build_references(s) if spec and s.verification and s.verification.ok else {}
    stab = compare.stability(s) if s.recommended else {}
    ds = Path(dataset_dir) if dataset_dir else run.parent.parent
    dataset: Dict[str, Any] = {"name": ds.name, "data_file": Path(s.data_path).name}
    for key, fname in (("dictionary", "dictionary.md"), ("readme", "README.md")):
        v = _read(ds / fname)
        if v:
            dataset[key] = v
    for key, fname in (("published_specification", "specification.json"), ("truth", "truth.json")):
        v = _json(ds / fname)
        if v is not None:
            dataset[key] = v
    scenario = _read(ds / "scenarios" / f"{run.name}.txt")
    out = {
        "format": "pprof-agent-bundle/1",
        "run": run.name, "run_dir": str(run), "phase": s.phase, "config_hash": s.config_hash,
        "dataset": dataset,
        "scenario": {"name": run.name, "script": scenario},
        "specification": spec.to_dict() if spec else None,
        "card": s.card, "profile": s.profile,
        "admissible": s.admissible.to_dict() if s.admissible else None,
        "thresholds": dict(rules.THRESHOLDS),
        "recommended": s.recommended,
        "actions": [{"name": c.name, "purpose": c.purpose, "cost": c.cost, "needs": list(c.needs),
                     "args": {k: ({"enum": r["enum"]} if "enum" in r else {k2: v2 for k2, v2 in r.items() if k2 != "doc"}) for k, r in c.args.items()}}
                    for c in actions.CARDS.values()],
        "results": [{"id": r.id, "action": r.action, "args": r.args, "source": r.source, "reason": r.reason, "step": r.step,
                     "seconds": r.seconds, "summary": _light(r.summary), "tables": [Path(v).name for v in (r.tables or {}).values()]}
                    for r in s.results],
        "plan_log": s.plan_log,
        "model_calls": [{k: v for k, v in c.items() if k not in ("raw_text",)} | {"raw_text": (c.get("raw_text") or "")[:4000]} for c in s.model_calls],
        "refusals": s.refusals, "steps": s.steps, "budget": s.budget, "cost_used": s.cost_used,
        "references": refs, "prose": s.prose, "report_md": _read(run / "report.md"), "repro_r": _read(run / "repro.R"),
        "transcript": s.transcript, "turns": s.turns,
        "stability": stab,
        "tables": tables, "figures": figures,
    }
    return _clean(out)


def _clean(obj: Any) -> Any:
    """JSON-safe: NaN and infinities become null (browsers' JSON.parse refuses them), numpy scalars become Python ones."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, "item"):
        try:
            return _clean(obj.item())
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)


def _light(summary: Dict[str, Any]) -> Dict[str, Any]:
    """A result summary without the bulky parts (tables are exported on their own)."""
    out = {}
    for k, v in (summary or {}).items():
        if k in ("tables", "_stderr", "stderr"):
            continue
        if isinstance(v, list) and len(v) > 200:
            out[k] = f"list[{len(v)}]"
        else:
            out[k] = v
    return out


def write_bundle(run_dir: str, out: Optional[str] = None, dataset_dir: Optional[str] = None) -> Path:
    b = bundle(run_dir, dataset_dir)
    p = Path(out) if out else Path(run_dir) / "bundle.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean(b), allow_nan=False), encoding="utf-8")
    return p


def build_site_data(examples_dir: str, site_dir: str, do_subset: bool = False, log=print) -> Path:
    """`<site_dir>/data/index.json` and one bundle per finished run under `<examples_dir>/<name>/runs/<scenario>/`."""
    ex = Path(examples_dir)
    data = Path(site_dir) / "data"
    data.mkdir(parents=True, exist_ok=True)
    index: Dict[str, Any] = {"datasets": []}
    for d in sorted(ex.iterdir()):
        if not (d / "data.csv").exists():
            continue
        entry = {"name": d.name, "title": _title(d), "runs": []}
        for r in sorted((d / "runs").glob("*")) if (d / "runs").exists() else []:
            if not (r / "trace.json").exists():
                continue
            try:
                if do_subset:
                    s = Session.load(str(r))
                    if s.phase == "DONE":
                        n = subset_flagged(s, log=log)
                        log(f"{d.name}/{r.name}: {n} breakdowns added")
                p = write_bundle(str(r), str(data / d.name / f"{r.name}.json"), dataset_dir=str(d))
                b = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 -- a run still being written, or an old trace
                log(f"{d.name}/{r.name}: not exported ({exc})")
                continue
            entry["runs"].append({"name": r.name, "title": _run_title(d, r.name), "file": f"data/{d.name}/{r.name}.json", "phase": b["phase"],
                                  "with_model": any(c.get("name") == "next_step" for c in b["model_calls"]),
                                  "size_bytes": p.stat().st_size, "messages": len(b["transcript"])})
            log(f"{d.name}/{r.name}: {p.stat().st_size // 1024} KB")
        index["datasets"].append(entry)
    (data / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return data / "index.json"


def _run_title(d: Path, run: str) -> str:
    """The scenario's first comment line names the run; the harness-only run and an edit re-run get fixed names."""
    if run == "cli":
        return "Harness only (no language model)"
    base = run[:-5] if run.endswith("_edit") else run
    p = d / "scenarios" / f"{base}.txt"
    first = next((l.strip().lstrip("#").strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip().startswith("#")), "") if p.exists() else ""
    if run.endswith("_edit"):
        return (first or base) + " (the re-run after the edit)"
    return first or run


def _title(d: Path) -> str:
    md = _read(d / "dictionary.md") or ""
    first = md.splitlines()[0] if md else d.name
    return first.lstrip("# ").split(" — ")[0] if first else d.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pprof_agent.export")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run_dir")
    r.add_argument("--out")
    r.add_argument("--dataset-dir")
    r.add_argument("--subset-flagged", action="store_true")
    w = sub.add_parser("site")
    w.add_argument("examples_dir")
    w.add_argument("site_dir")
    w.add_argument("--subset-flagged", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if a.subset_flagged:
            s = Session.load(a.run_dir)
            print("breakdowns added:", subset_flagged(s))
        p = write_bundle(a.run_dir, a.out, a.dataset_dir)
        print(p, p.stat().st_size // 1024, "KB")
    else:
        p = build_site_data(a.examples_dir, a.site_dir, do_subset=a.subset_flagged)
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
