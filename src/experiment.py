# cross validation on the 5 folds, random search and the final test
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline

from .config import SEED, TARGET, NEG_SAMPLE_FRAC, N_TRIALS
from .data import ALL_FEATURES
from .folds import forward_chaining_folds, final_split
from .metrics import evaluate, threshold_at_fpr
from .models import MODELS, EARLY_STOP_FITTERS, ES_HOLDOUT_FRAC
from .preprocess import build_preprocessor, downsample_negatives


def make_pipeline(model_name, params, features, missing_indicators=True, seed=SEED):
    factory, _, needs_scaling = MODELS[model_name]
    pre = build_preprocessor(features, scale=needs_scaling, missing_indicators=missing_indicators)
    return Pipeline([("pre", pre), ("model", factory(params, seed))])


def fit_predict(model_name, params, df, train_idx, pred_idx, features,
                neg_frac=NEG_SAMPLE_FRAC, missing_indicators=True, seed=SEED, es_idx=None):
    # es_idx: row positions of the early-stopping set.
    #   final fit -> month 5 (passed by final_test)
    #   CV folds  -> None: a random 10% of the training rows is held out instead,
    #                so the validation month never influences training (no leakage)
    X_tr = df.iloc[train_idx][features]
    y_tr = df.iloc[train_idx][TARGET].to_numpy()
    X_tr, y_tr = downsample_negatives(X_tr, y_tr, neg_frac, seed)
    pipe = make_pipeline(model_name, params, features, missing_indicators, seed)
    X_pred = df.iloc[pred_idx][features]

    if model_name not in EARLY_STOP_FITTERS:
        pipe.fit(X_tr, y_tr)
        return pipe, pipe.predict_proba(X_pred)[:, 1]

    if es_idx is not None:
        X_es = df.iloc[es_idx][features]
        y_es = df.iloc[es_idx][TARGET].to_numpy()
    else:
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_tr, y_tr, test_size=ES_HOLDOUT_FRAC, stratify=y_tr, random_state=seed)

    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    Z_tr = pre.fit_transform(X_tr, y_tr)      # preprocessor is fitted on the training rows only
    Z_es = pre.transform(X_es)
    EARLY_STOP_FITTERS[model_name](model, Z_tr, y_tr, Z_es, y_es)
    scores = model.predict_proba(pre.transform(X_pred))[:, 1]
    return pipe, scores


def run_cv(model_name, params, df, features=ALL_FEATURES, **kw):
    # one row of metrics for each fold
    rows = []
    for k, tr, va in forward_chaining_folds(df):
        t0 = time.time()
        _, scores = fit_predict(model_name, params, df, tr, va, features, **kw)
        m = evaluate(df.iloc[va][TARGET].to_numpy(), scores)
        m.update({"fold": k, "seconds": round(time.time() - t0, 1)})
        rows.append(m)
    return pd.DataFrame(rows)


def random_search(model_name, df, n_trials=N_TRIALS, seed=SEED, log=print):
    # random search, best trial = highest mean recall at 5% FPR over the 5 folds
    _, space, _ = MODELS[model_name]
    if space is None:  # nothing to tune (majority class)
        candidates = [{}]
    else:
        candidates = list(ParameterSampler(space, n_iter=n_trials, random_state=seed))

    trial_rows, fold_tables = [], []
    for i, params in enumerate(candidates, start=1):
        folds = run_cv(model_name, params, df)
        folds["trial"] = i
        fold_tables.append(folds)
        summary = {"trial": i, **{f"param_{k}": v for k, v in params.items()}}
        for col in ["recall_at_5fpr", "roc_auc", "pr_auc", "f1"]:
            summary[f"{col}_mean"] = folds[col].mean()
            summary[f"{col}_std"] = folds[col].std(ddof=1)
        summary["seconds"] = folds["seconds"].sum()
        trial_rows.append(summary)
        log(f"[{model_name}] trial {i}/{len(candidates)} {params} -> "
            f"recall@5%FPR {summary['recall_at_5fpr_mean']:.4f} ± {summary['recall_at_5fpr_std']:.4f} "
            f"({summary['seconds']:.0f}s)")

    trials = pd.DataFrame(trial_rows)
    best_trial = int(trials.sort_values("recall_at_5fpr_mean", ascending=False).iloc[0]["trial"])
    best_params = candidates[best_trial - 1]
    all_folds = pd.concat(fold_tables, ignore_index=True)
    best_folds = all_folds[all_folds["trial"] == best_trial].copy()
    return trials, best_params, best_folds


def final_test(model_name, params, df, features=ALL_FEATURES, **kw):
    # train on months 0-4, pick the threshold on month 5, then test on months 6 and 7
    # early stopping (lightgbm, catboost, mlp) uses month 5 as the stopping set
    idx = final_split(df)
    pred_idx = np.concatenate([idx["threshold"], idx["test_m6_7"]])
    pipe, scores = fit_predict(model_name, params, df, idx["fit"], pred_idx, features,
                               es_idx=idx["threshold"], **kw)
    n_thr = len(idx["threshold"])
    s_thr, s_test = scores[:n_thr], scores[n_thr:]

    y = df[TARGET].to_numpy()
    thr = threshold_at_fpr(y[idx["threshold"]], s_thr)

    score_lookup = pd.Series(s_test, index=idx["test_m6_7"])
    rows = []
    for name in ["threshold", "test_m6", "test_m7", "test_m6_7"]:
        s = s_thr if name == "threshold" else score_lookup.loc[idx[name]].to_numpy()
        m = evaluate(y[idx[name]], s, thr=thr)
        m["split"] = {"threshold": "month 5 (threshold)", "test_m6": "month 6",
                      "test_m7": "month 7", "test_m6_7": "months 6-7"}[name]
        rows.append(m)

    test_scores = pd.DataFrame({
        "row": idx["test_m6_7"],
        "month": df.iloc[idx["test_m6_7"]]["month"].to_numpy(),
        "y_true": y[idx["test_m6_7"]],
        "score": s_test,
        "flagged": (s_test > thr).astype(int),
    })
    return pd.DataFrame(rows), test_scores, pipe