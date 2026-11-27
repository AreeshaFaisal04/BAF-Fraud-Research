# Bank Account Fraud Detection that Stays Fair and Stable over Time

ML term project (Topic 9), Kinnaird College for Women University, Fall 2026.
Dataset: Bank Account Fraud (BAF) suite, Base variant (Jesus et al., NeurIPS 2022).

> Group members: _add names and roll numbers_ · Instructor: Dr. Sidra Zafar

## What this repository does

- Time-based evaluation exactly as in the BAF paper: **months 0–5 for development, months 6–7 for testing**. Months are never shuffled.
- **5 forward-chaining folds** on months 0–5 (train on 0..k-1, validate on k) for tuning and for mean ± std.
- **Final run:** refit on months 0–4, fix the threshold at **5% FPR on month 5**, then score months 6 and 7 once.
- Main metric: **recall at 5% false positive rate**, plus ROC-AUC, PR-AUC and F1.
- Same folds, same preprocessing, same 20-trial random-search budget and same metric code for every model.

## Quickest way: Google Colab

Open `notebooks/Member1_BAF_pipeline_colab.ipynb` in Colab and click **Runtime → Run all**.
It mounts Google Drive, downloads Base.csv from Kaggle with `kagglehub`, and runs every step below
(about an hour on the free CPU runtime). Results are saved to `MyDrive/BAF_ML_Project/outputs/`.

## Setup (local)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download **Base.csv** (or Base.parquet) from Kaggle:
https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
and put it in `data/`. The data is not committed (licence CC BY-NC-ND 4.0, and it is ~200 MB).

## Reproduce the results table

```bash
bash run_member1.sh          # EDA + majority, logistic regression, random forest
python scripts/04_results_table.py
python scripts/05_paper_tables.py
```

or step by step:

```bash
python scripts/01_eda.py                       # stats + dataset figures
python scripts/02_tune_cv.py --model logreg    # 20-trial random search on the 5 folds
python scripts/03_final_test.py --model logreg # months 0-4 fit, month 5 threshold, months 6-7 test
python scripts/04_results_table.py             # collects every model that has been run
python scripts/05_paper_tables.py              # LaTeX tables for the paper
python scripts/pipeline_diagram.py             # pipeline figure
```

The results table is written to `outputs/tables/results_table.md` (and `.csv`, `.tex`).
Random forest tuning is the slowest step (100 fits); `02_tune_cv.py` prints the time of every trial.

## Results (Member 1 models)

_Filled in automatically from `outputs/tables/results_table.md` after running the scripts._

## Repository layout

```
src/
  config.py      seed, months, FPR target, tuning budget, sampling fraction (shared by everyone)
  data.py        loading, cleaning (coded missing values -> NaN), feature groups
  folds.py       forward-chaining folds and the final split
  preprocess.py  imputation + missing flags, one-hot, scaling; negative sampling of training rows
  metrics.py     recall at 5% FPR, threshold at FPR, ROC-AUC, PR-AUC, F1
  models.py      model factories + search spaces (add new models here)
  experiment.py  CV, random search, final test
  plotstyle.py   shared figure style
scripts/
  01_eda.py, 02_tune_cv.py, 03_final_test.py, 04_results_table.py, 05_paper_tables.py, pipeline_diagram.py
outputs/
  tables/        CSV/JSON results (committed, small)
  figures/       PDF/PNG figures
paper/           LaTeX sections, tables and references for Overleaf
```

## For Members 2 and 3: adding a model

1. In `src/models.py` write `make_<name>(params, seed)` returning an estimator with `predict_proba`, add a search space dict, and register it in `MODELS` (set the third value to `True` if it needs scaling).
2. Run `python scripts/02_tune_cv.py --model <name>` then `python scripts/03_final_test.py --model <name>`.
3. Re-run `python scripts/04_results_table.py`. The fold scores in `outputs/tables/cv_folds_<name>.csv` are what the box plot needs, and `test_scores_<name>.csv` has every test-month score for the fairness and error analysis (`row` = position in the cleaned dataframe).

Do **not** change `src/config.py`, `src/folds.py`, `src/preprocess.py` or `src/metrics.py` without telling the group, otherwise the comparison is no longer fair.

## Notes on the data

- `-1` means missing in `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, `device_distinct_emails_8w`; any negative value means missing in `intended_balcon_amount` (BAF datasheet).
- `month` is used only for splitting, never as a feature. `device_fraud_count` is constant in the Base file and is dropped.
- Age groups for fairness: `customer_age > 50` vs `<= 50`, as in the BAF paper.

## Licence

Code: MIT. Data: BAF suite, CC BY-NC-ND 4.0 (not redistributed here).
