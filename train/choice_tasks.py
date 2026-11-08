"""Training tasks for choosing the reported procedure (the act in pprof_agent/choose.py).

For every design, a few data sets are simulated (write_task_data.R, seeds apart from the operating
characteristics' repetitions) and each is analysed by the harness itself without a model
(pipeline.run_analysis with the declared question), plus the empirical null for every computed test, so the
act sees exactly the diagnostics it would see in deployment. A task records the act's prompt (policy
`choose_procedure` + choose.observation), the allowed answers, and the rule-based choice of V1 (the baseline);
its reward per allowed answer is joined later from the design's operating characteristics (summarize_oc.py).

    python choice_tasks.py --designs designs.jsonl --per-design 4 --work <scratch> --out choice_tasks.jsonl
Runs the tasks in parallel over SLURM_CPUS_PER_TASK processes, each R step on one thread.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
COVARIATES = ["AGE", "FEMALE", "DIABETES", "CHF", "COPD", "BMI", "YRS", "NH", "ALB", "HGB"]


def spec_for(design: Dict[str, Any]) -> Dict[str, Any]:
    """The analyst's declaration for a design's data: what the task takes as given (the question)."""
    binom = design["kind"] == "binomial_days"
    return {"outcome": "Y", "trials": "N" if binom else None, "provider": "PROV", "patient": "PAT",
            "adjusters": COVARIATES, "unit": "index_discharge" if design["kind"] == "discharge" else "patient_month",
            "event_direction": design["event_direction"], "eligibility": int(design["eligibility"]),
            "standardization": design.get("standardization", "indirect"), "norm": "median", "consequential_tail": "worse"}


def one_task(design: Dict[str, Any], j: int, designs_path: str, work: str, keep: bool) -> Dict[str, Any]:
    from pprof_agent import choose, pipeline, policy
    from pprof_agent.engine import executor
    tid = f"{design['id']}_t{j}"
    seed = int(design["seed"]) * 1000 + 500 + j
    tdir = Path(work) / tid
    try:
        subprocess.run(["Rscript", str(HERE / "write_task_data.R"), f"designs={designs_path}", f"id={design['id']}",
                        f"seed={seed}", f"out={tdir}"], check=True, capture_output=True, text=True)
        truth = json.load(open(tdir / "truth.json"))
        session = pipeline.run_analysis(str(tdir / "data.csv"), str(tdir), spec_dict=spec_for(design), figures=False,
                                        name="run", log=lambda *_: None)
        if session.phase in ("REFUSED", "DECLARING") or session.admissible is None:
            return {"id": tid, "error": f"phase {session.phase}",
                    "refusals": getattr(session.verification, "refusals", None)}
        enough = int(session.card.get("eligible_providers") or 0) >= 40
        for t in session.admissible.tests_computed:
            if enough and session.get(f"en:{t}") is None and session.get(f"test:{t}") is not None:
                try:
                    executor.execute(session, "empirical_null", {"test": t, "n_groups": 4, "size_by": "patients"},
                                     source="harness", reason="training diagnostics")
                except Exception as exc:  # noqa: BLE001
                    session.refusals.append(f"en {t}: {exc}")
        rec = session.recommended or {}
        fit = session.get("fit").summary
        diag = {"within_patient_corr": fit.get("within_patient_corr"),
                "median_patients": (session.card.get("patients_per_provider") or {}).get("median"),
                "invariant": int((session.card.get("invariant_providers") or {}).get("total", 0) or 0),
                "eligible": session.card.get("eligible_providers"), "tests": {}}
        for t in session.admissible.tests_computed:
            r, en = session.get(f"test:{t}"), session.get(f"en:{t}")
            if r is None:
                continue
            g = (en.summary.get("groups") or []) if en is not None else []
            diag["tests"][t] = {"z_beyond_pct": r.summary.get("z_beyond_pct"), "z_scale_by_size": r.summary.get("z_scale_by_size"),
                                "en_scales": [x.get("scale") for x in g] if isinstance(g, list) else None,
                                "en_location_max": en.summary.get("location_max_abs") if en is not None else None}
        return {"id": tid, "design_id": design["id"], "family": design["family"], "seed": seed, "act": "choose_procedure",
                "diag": diag,
                "messages": [{"role": "system", "content": policy.load("choose_procedure")},
                             {"role": "user", "content": choose.observation(session)}],
                "allowed": choose.allowed(session),
                "rule_choice": f"{rec.get('test')}|{rec.get('reference')}",
                "truth": {"n_up": len(truth["up"]), "n_down": len(truth["down"]), "n_invariant": len(truth["invariant"]),
                          "shift": truth["shift"], "n_eligible": truth["n_eligible"]}}
    except Exception as exc:  # noqa: BLE001
        return {"id": tid, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}
    finally:
        if not keep:
            shutil.rmtree(tdir, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--designs", required=True)
    ap.add_argument("--per-design", type=int, default=4)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true", help="keep each task's run directory")
    ap.add_argument("--limit", type=int, default=0, help="only the first N designs")
    a = ap.parse_args(argv)
    designs = [json.loads(l) for l in open(a.designs) if l.strip()]
    if a.limit:
        designs = designs[:a.limit]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    os.environ["SLURM_CPUS_PER_TASK"] = "1"      # each R step on one thread; the parallelism is over tasks
    os.makedirs(a.work, exist_ok=True)
    jobs = [(d, j) for d in designs for j in range(1, a.per_design + 1)]
    ok = bad = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as ex, open(a.out, "w") as fh:
        futs = [ex.submit(one_task, d, j, os.path.abspath(a.designs), a.work, a.keep) for d, j in jobs]
        for f in cf.as_completed(futs):
            r = f.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if "error" in r:
                bad += 1
                print(f"FAILED {r['id']}: {r['error']}", file=sys.stderr)
            else:
                ok += 1
    print(f"{ok} tasks written, {bad} failed -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
