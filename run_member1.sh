#!/usr/bin/env bash
# Reproduces everything for Member 1 (EDA + majority, logistic regression, random forest).
# Put Base.csv (or Base.parquet) in data/ first.
set -e
python scripts/01_eda.py
for m in majority logreg rf; do
    python scripts/02_tune_cv.py --model $m
    python scripts/03_final_test.py --model $m
done
python scripts/04_results_table.py
