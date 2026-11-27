"""
Model definitions + random-search spaces.

To add a model (Members 2 and 3): write a make_<name>(params, seed) function that
returns an sklearn-style estimator with predict_proba, add its search space,
and register both in MODELS at the bottom. Nothing else needs to change -
scripts/02_tune_cv.py and scripts/03_final_test.py will then work for it.
"""
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ---------------- baseline 1: majority class ----------------
def make_majority(params, seed):
    # always predicts "not fraud" -> every row gets the same score
    return DummyClassifier(strategy="most_frequent")


# ---------------- baseline 2: logistic regression scorecard ----------------
def make_logreg(params, seed):
    return LogisticRegression(
        C=params["C"],
        class_weight="balanced",   # class weights fixed on (removed only in the ablation)
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    )


LOGREG_SPACE = {
    "C": loguniform(1e-3, 1e2),
}


# ---------------- random forest ----------------
def make_rf(params, seed):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        max_samples=params["max_samples"],
        bootstrap=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


RF_SPACE = {
    "n_estimators": randint(100, 301),
    "max_depth": [8, 12, 16, 20, 24],
    "min_samples_leaf": [1, 5, 10, 25, 50, 100],
    "max_features": ["sqrt", 0.2, 0.33],
    "max_samples": uniform(0.05, 0.25),  # 5% - 30% of the training rows per tree (keeps 100 fits affordable on 2 cores)
}


# name -> (factory, search space or None, needs scaling?)
MODELS = {
    "majority": (make_majority, None, False),
    "logreg": (make_logreg, LOGREG_SPACE, True),
    "rf": (make_rf, RF_SPACE, False),
    # Member 2: "lightgbm": (make_lightgbm, LGBM_SPACE, False), "catboost": (...)
    # Member 3: "mlp": (make_mlp, MLP_SPACE, True)
}
