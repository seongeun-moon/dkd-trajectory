#!/usr/bin/env bash
# The baseline sweep: four models x three seeds x three balancing runs.
#
# Point the driver at the preprocessed data and a writable result directory,
# either through these variables or with --data_path / --result_path.
#
# TARGET_TYPE picks the endpoint. It defaults to the one this sweep has always
# run; set it to ckd-stage for the other. The two are not interchangeable.
set -euo pipefail

: "${CKD_DATA_ROOT:?set CKD_DATA_ROOT to the preprocessed dataset directory}"
: "${CKD_RESULT_ROOT:=./result}"
: "${TARGET_TYPE:=egfr-decline}"

for model in lr svm lightgbm xgb; do
  for seed in 2022 2023 2024; do
    for run in 26 27 28; do
      python main_pycaret.py \
        --data_path "$CKD_DATA_ROOT" --result_path "$CKD_RESULT_ROOT" \
        --agg average --concat_time --concat_cdt --balanced \
        --binary --target_type "$TARGET_TYPE" \
        --model "$model" --seed "$seed" --balance_run "$run"
    done
  done
done
