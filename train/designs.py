"""Training designs: the structures the simulator (simulate.R) draws data sets from.

Each design is one line of designs.jsonl. The categorical factors are balanced (every level appears about
equally often, in a shuffled order) and the continuous ones are drawn log-uniformly, so a few hundred designs
cover the space without a full factorial. The factors are the situations the agent's choice must react to:

  kind              patient-month rows (SFR-like), index discharges (SRR-like), binomial days at risk (AOH-like)
  n_prov            number of providers
  median_patients   provider size, with `size_sdlog` its spread (log-normal), so tiny and large providers mix
  sigma_patient     the patient random effect: the within-patient correlation the naive variance ignores
  target_rate       the event rate
  tau               the ordinary provider-to-provider spread (null under definition B)
  planted_frac      the share of eligible providers planted as outliers, half per tail, with `effect_or`
  casemix           how much the case mix differs by provider
  n_invariant       providers whose outcomes are all zero (Wald-type tests inadmissible)
  eligibility       providers with fewer patients are not profiled
  event_direction   which tail is the worse one (the consequential tail)

`family` groups designs for held-out evaluation (whole families can be left out of training).

    python designs.py --n 30 --seed 7 --max-prov 400 --out designs.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
from typing import Any, Dict, List, Sequence

KINDS = ("patient_month", "discharge", "binomial_days")
RATE = {"patient_month": (0.05, 0.6), "discharge": (0.08, 0.35), "binomial_days": (0.01, 0.08)}
N_PROV = (100, 200, 400, 800)
SIGMA = (0.0, 0.5, 1.0, 1.5)
TAU = (0.0, 0.1, 0.2, 0.35)
PLANTED = (0.0, 0.02, 0.05, 0.10)
CASEMIX = (0.0, 0.5, 1.0)
ELIGIBILITY = (5, 11, 20)


def balanced(levels: Sequence[Any], n: int, rng: random.Random) -> List[Any]:
    """`n` values cycling through `levels`, shuffled: each level appears floor or ceil n / len(levels) times."""
    out = [levels[i % len(levels)] for i in range(n)]
    rng.shuffle(out)
    return out


def loguniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def family(d: Dict[str, Any]) -> str:
    size = "small" if d["median_patients"] < 15 else ("medium" if d["median_patients"] < 35 else "large")
    return "|".join([d["kind"], "tau0" if d["tau"] == 0 else "tau+", size, "corr0" if d["sigma_patient"] == 0 else "corr+"])


def draw(n: int, seed: int, max_prov: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    n_prov_levels = [p for p in N_PROV if p <= max_prov]
    cols = {"kind": balanced(KINDS, n, rng), "n_prov": balanced(n_prov_levels, n, rng),
            "sigma_patient": balanced(SIGMA, n, rng), "tau": balanced(TAU, n, rng),
            "planted_frac": balanced(PLANTED, n, rng), "casemix": balanced(CASEMIX, n, rng),
            "eligibility": balanced(ELIGIBILITY, n, rng), "n_invariant": balanced((0, 0, 2), n, rng),
            "event_direction": balanced(("higher_is_worse", "higher_is_worse", "higher_is_better"), n, rng)}
    designs = []
    for i in range(n):
        d: Dict[str, Any] = {k: v[i] for k, v in cols.items()}
        lo, hi = RATE[d["kind"]]
        d.update(median_patients=round(loguniform(rng, 8, 60), 1), size_sdlog=round(rng.uniform(0.5, 1.2), 2),
                 target_rate=round(loguniform(rng, lo, hi), 4), effect_or=round(loguniform(rng, 1.3, 2.5), 3),
                 standardization="indirect")
        if d["kind"] == "binomial_days":
            d["n_invariant"] = 0          # zero hospital days over a whole facility year is not a plausible facility
        d["id"] = f"d{seed:03d}_{i + 1:04d}"
        d["seed"] = seed * 10000 + i + 1
        d["family"] = family(d)
        designs.append(d)
    return designs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-prov", type=int, default=800)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    designs = draw(a.n, a.seed, a.max_prov)
    with open(a.out, "w") as fh:
        for d in designs:
            fh.write(json.dumps(d, sort_keys=True) + "\n")
    print(f"{len(designs)} designs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
