#!/bin/bash
# smoke.sh <data_dir> <run_dir> -- run every R script once, in order, without Python. Prints each
# script's JSON. Usage inside an sbatch job (never on the login node).
set -u
DATA=$1; RUN=$2; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$RUN/payloads" "$RUN/outputs"
OUTCOME=$(python3 -c "import json,sys; print(json.load(open('$DATA/truth.json'))['outcome'])")
run() { # name script payload-json
  echo "$3" > "$RUN/payloads/$1.json"
  if Rscript "$HERE/$2" "$RUN/payloads/$1.json" "$RUN/outputs/$1.json"; then echo "--- $1: ok"; else echo "--- $1: FAILED (exit $?)"; fi
  head -c 1500 "$RUN/outputs/$1.json"; echo
}
run profile profile_data.R "{\"data_path\": \"$DATA/data.csv\"}"
SPEC="{\"outcome\": \"$OUTCOME\", \"provider\": \"PROVFS\", \"patient\": \"KECC_ID\", \"adjusters\": [\"AGE\",\"FEMALE\",\"DIABETES\",\"CHF\",\"COPD\",\"BMI\",\"YRS_ESRD\",\"NH\",\"ALB\",\"HGB\"], \"eligibility\": 2}"
run verify verify_spec.R "{\"data_path\": \"$DATA/data.csv\", \"spec\": $SPEC}"
run fit fit_fe.R "{\"data_path\": \"$DATA/data.csv\", \"run_dir\": \"$RUN\", \"spec\": $SPEC, \"tol\": 1e-6}"
run standardize standardize.R "{\"run_dir\": \"$RUN\", \"stdz\": [\"indirect\",\"direct\"], \"norm\": \"median\"}"
for t in exact score wald robust_wald robust_score stabilized_wald stabilized_score; do
  run test_$t test_providers.R "{\"run_dir\": \"$RUN\", \"test\": \"$t\", \"null\": \"median\", \"key\": \"$t\"}"
done
run en empirical_null.R "{\"run_dir\": \"$RUN\", \"key\": \"robust_score\", \"n_groups\": 4}"
run measure test_measure.R "{\"run_dir\": \"$RUN\", \"measure\": \"direct_rate\", \"se_source\": \"robust_wald\"}"
run evaluate evaluate_model.R "{\"run_dir\": \"$RUN\"}"
run select select_adjusters.R "{\"data_path\": \"$DATA/data.csv\", \"run_dir\": \"$RUN\", \"spec\": $SPEC, \"candidates\": [\"AGE\",\"FEMALE\",\"DIABETES\",\"CHF\",\"COPD\",\"BMI\",\"YRS_ESRD\",\"NH\",\"ALB\",\"HGB\"], \"nfolds\": 5}"
ls "$RUN/tables"
