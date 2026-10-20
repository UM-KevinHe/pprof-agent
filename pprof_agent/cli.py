"""Command line: run the agent without a model.

    python -m pprof_agent.cli run --data data.csv --out runs/ [--spec spec.json] [--reply "2) higher_is_better 3) month"]

Prints the numbered question when something is open, the refusal when the gate refuses, and the
report path when the run finishes.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pprof_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="profile, declare, verify, run the loop, report")
    r.add_argument("--data", required=True)
    r.add_argument("--out", required=True, help="directory under which the run directory is created")
    r.add_argument("--spec", help="JSON file with the declared fields")
    r.add_argument("--reply", help="a numbered reply to the open items")
    r.add_argument("--budget", type=int, default=30)
    r.add_argument("--name", help="run directory name")
    r.add_argument("--no-figures", action="store_true")
    r.add_argument("--calibrate", type=int, default=0, help="run the calibration simulation with this many data sets after the loop")
    r.add_argument("--subset", help="a provider id to break down after the loop")
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec, encoding="utf-8")) if a.spec else None
    extras = []
    if a.calibrate:
        extras.append(("calibrate_tests", {"n_sim": a.calibrate}))
    if a.subset:
        extras.append(("subset", {"provider": a.subset}))
    session = pipeline.run_analysis(a.data, a.out, spec_dict=spec, reply=a.reply, budget=a.budget, name=a.name,
                                    figures=not a.no_figures, extras=extras)
    print(f"phase: {session.phase}")
    if session.phase == "REFUSED":
        for x in session.verification.refusals:
            print("refused:", x)
        return 2
    if session.phase == "DECLARING":
        return 3
    print(json.dumps(session.recommended, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
