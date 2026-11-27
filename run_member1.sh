#!/usr/bin/env bash
# runs the eda and the three baseline models
# Base.csv (or Base.parquet) has to be in data/
set -e
python scripts/01_eda.py
for m in majority logreg rf; do
    python scripts/02_tune_cv.py --model $m
    python scripts/03_final_test.py --model $m
done
python scripts/04_results_table.py
