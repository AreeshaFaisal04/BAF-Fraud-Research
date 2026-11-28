# Bank Account Fraud Detection that Stays Fair and Stable over Time

ML term project (Topic 9), Kinnaird College for Women University, Fall 2026.
Dataset: Bank Account Fraud (BAF) suite, Base variant (Jesus et al., NeurIPS 2022).

Group members: Sidra Jahangir, Areesha Faisal, Rayha Azim · Instructor: Dr. Sidra Zafar

## What this repository does

- Time-based evaluation exactly as in the BAF paper: months 0–5 for development, months 6–7 for testing. Months are never shuffled.
- 5 forward-chaining folds on months 0–5 (train on 0..k-1, validate on k) for tuning and for mean ± std.
- Final run: refit on months 0–4, fix the threshold at 5% FPR on month 5, then score months 6 and 7 once.
- Main metric: recall at 5% false positive rate, plus ROC-AUC, PR-AUC and F1.
- Same folds, same preprocessing, same 20-trial random-search budget and same metric code for every model.

## Quickest way: Google Colab

Open `notebooks/Member1_BAF_pipeline_colab.ipynb` in Colab and click Runtime → Run all. First upload `Base.csv` to `MyDrive/BAF_ML_Project/data/` (only `Base.csv` is needed from the Kaggle zip, not the Variant files). The notebook mounts Google Drive, reads `Base.csv` from there and runs every step below (about an hour on the free CPU runtime). Results are saved to `MyDrive/BAF_ML_Project/outputs/`.

## Setup (local)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `Base.csv` (or `Base.parquet`) from Kaggle: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 and put it in `data/`. The data is not committed (licence CC BY-NC-ND 4.0, and it is ~200 MB).

`requirements.txt` includes `lightgbm` and `catboost` (needed for the Member 2 models).

## Reproduce the results table

```bash
bash run_member1.sh          # EDA + majority, logistic regression, random forest

# Member 2 models (LightGBM, CatBoost)
python scripts/02_tune_cv.py --model lightgbm
python scripts/03_final_test.py --model lightgbm
python scripts/02_tune_cv.py --model catboost
python scripts/03_final_test.py --model catboost

python scripts/04_results_table.py
python scripts/05_paper_tables.py
```

or step by step for one model:

```bash
python scripts/01_eda.py                       # stats + dataset figures
python scripts/02_tune_cv.py --model logreg    # 20-trial random search on the 5 folds
python scripts/03_final_test.py --model logreg # months 0-4 fit, month 5 threshold, months 6-7 test
python scripts/04_results_table.py             # collects every model that has been run
python scripts/05_paper_tables.py              # LaTeX tables for the paper
python scripts/pipeline_diagram.py             # pipeline figure
```

Valid model names: `majority`, `logreg`, `rf`, `lightgbm`, `catboost` (and `mlp` once Member 3 has added it).

The results table is written to `outputs/tables/results_table.md` (and `.csv`, `.tex`). Random forest tuning is the slowest step (100 fits); `02_tune_cv.py` prints the time of every trial.

## Results (Member 1 models)

Run on the free Google Colab CPU runtime (2 cores, 12.7 GB RAM; versions in `outputs/tables/environment.json`). CV columns are mean ± std over the 5 forward-chaining folds (validation months 1-5, threshold set at exactly 5% FPR inside each month). Test columns use the model refit on months 0-4 with the threshold fixed on month 5.

| model | CV Recall@5%FPR | CV ROC-AUC | CV PR-AUC | CV F1 | Test M6 Recall@5%FPR | Test M7 Recall@5%FPR | Test M6-7 FPR | Test M6-7 ROC-AUC |
|:--|:--|:--|:--|:--|--:|--:|--:|--:|
| Majority class | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.010 ± 0.001 | 0.000 ± 0.000 | 0 | 0 | 0 | 0.5 |
| Logistic regression | 0.516 ± 0.020 | 0.887 ± 0.014 | 0.134 ± 0.018 | 0.161 ± 0.018 | 0.522 | 0.441 | 0.046 | 0.881 |
| Random forest | 0.503 ± 0.022 | 0.881 ± 0.018 | 0.125 ± 0.018 | 0.157 ± 0.020 | 0.521 | 0.496 | 0.048 | 0.88 |

Notes:

- The majority baseline is 98.5-99.1% accurate on every month but catches no fraud, which is why accuracy is not used.
- Logistic regression leads random forest by 0.013 in mean CV recall, which is smaller than the fold std (~0.02), so neither is clearly better on the development months. On the test months the forest holds up better in month 7 (0.496 vs 0.441).
- The threshold fixed on month 5 gives a 5.6% FPR in month 6 and 3.4-3.9% in month 7, i.e. the operating point drifts.
- Score files for every test application (`outputs/tables/test_scores_<model>.csv`, used for the fairness and error analysis) are too big for GitHub; they are re-created by `scripts/03_final_test.py` (also kept in the group's Drive folder).

## Results (Member 2 models: LightGBM, CatBoost)

Same folds, preprocessing, 20-trial budget and metric as the other models. Both models use balanced class weights (`class_weight="balanced"` for LightGBM, `auto_class_weights="Balanced"` for CatBoost), up to 1000 trees, and early stopping with patience 50 on validation AUC. In the final fit the stopping set is month 5; inside the CV folds it is a random 10% of that fold's training rows, so the validation month never influences training. CatBoost uses the same one-hot encoded inputs as the other models. Run on the free Colab CPU runtime (Python 3.13, LightGBM 4.6.0, CatBoost 1.2.10, scikit-learn 1.6.1, pandas 2.2.3, NumPy 2.4.6).

| model | CV Recall@5%FPR | CV ROC-AUC | CV PR-AUC | CV F1 | Test M6 Recall@5%FPR | Test M7 Recall@5%FPR | Test M6-7 FPR | Test M6-7 ROC-AUC |
|:--|:--|:--|:--|:--|--:|--:|--:|--:|
| LightGBM | 0.550 ± 0.026 | 0.896 ± 0.017 | 0.160 ± 0.030 | 0.170 ± 0.021 | 0.568 | 0.575 | 0.055 | 0.895 |
| CatBoost | 0.555 ± 0.026 | 0.898 ± 0.015 | 0.163 ± 0.029 | 0.172 ± 0.020 | 0.568 | 0.561 | 0.055 | 0.895 |

Notes:

- Both boosted models lead logistic regression and the random forest in mean CV recall by more than the fold std (leads of 0.034-0.052). CatBoost vs LightGBM differs by 0.005, far below the std, so neither is clearly better.
- On the test months the boosted models keep their recall in month 7 (0.575 / 0.561), while logistic regression and the random forest drop. ROC-AUC does not fall in month 7 for any model, so the drop of the baselines mostly reflects the month-5 threshold drifting (their FPR falls to 3.4-3.9%), not worse ranking.
- The boosted models reach an FPR of 6.1-6.2% in month 6 and 4.8% in month 7. The slightly higher FPR in month 6 may explain part of their recall gap there.

<!-- Results (Member 3 models: MLP, fairness, ablation) to be added by Member 3 -->

## Repository layout

```
src/
  config.py      seed, months, FPR target, tuning budget, sampling fraction (shared by everyone)
  data.py        loading, cleaning (coded missing values -> NaN), feature groups
  folds.py       forward-chaining folds and the final split
  preprocess.py  imputation + missing flags, one-hot, scaling; negative sampling of training rows
  metrics.py     recall at 5% FPR, threshold at FPR, ROC-AUC, PR-AUC, F1
  models.py      model factories + search spaces (add new models here); early-stopping fit functions
  experiment.py  CV, random search, final test (supports early stopping on month 5)
  plotstyle.py   shared figure style
scripts/
  01_eda.py, 02_tune_cv.py, 03_final_test.py, 04_results_table.py, 05_paper_tables.py, pipeline_diagram.py
outputs/
  tables/        CSV/JSON results (committed, small)
  figures/       PDF/PNG figures
paper/           LaTeX sections, tables and references for Overleaf
```
