# models and their random search spaces
import lightgbm as lgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# early stopping: stop when the validation AUC has not improved for this many rounds
ES_ROUNDS = 50
# inside the CV folds: share of the (sampled) training rows held out for early stopping
ES_HOLDOUT_FRAC = 0.10


# ---------------- baseline 1: majority class ----------------
def make_majority(params, seed):
    # always predicts "not fraud" -> every row gets the same score
    return DummyClassifier(strategy="most_frequent")


# ---------------- baseline 2: logistic regression scorecard ----------------
def make_logreg(params, seed):
    return LogisticRegression(
        C=params["C"],
        class_weight="balanced",   # fraud class gets more weight
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
    "max_samples": uniform(0.05, 0.25),  # each tree uses 5-30% of the rows (otherwise too slow on colab)
}


# ---------------- LightGBM (Member 2) ----------------
def make_lightgbm(params, seed):
    return LGBMClassifier(
        n_estimators=1000,                      # upper limit, early stopping picks the real number
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        subsample_freq=1,                       # needed, otherwise subsample is ignored
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


LIGHTGBM_SPACE = {
    "learning_rate": loguniform(0.02, 0.3),
    "num_leaves": randint(15, 256),
    "max_depth": [-1, 6, 8, 12],
    "min_child_samples": randint(20, 200),
    "subsample": uniform(0.6, 0.4),             # 0.6 - 1.0
    "colsample_bytree": uniform(0.5, 0.5),      # 0.5 - 1.0
    "reg_lambda": loguniform(1e-3, 10),
}


def fit_lightgbm_es(model, Z_tr, y_tr, Z_es, y_es):
    model.fit(Z_tr, y_tr, eval_set=[(Z_es, y_es)], eval_metric="auc",
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])


# ---------------- CatBoost (Member 2) ----------------
def make_catboost(params, seed):
    return CatBoostClassifier(
        iterations=1000,                        # upper limit, early stopping picks the real number
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        bagging_temperature=params["bagging_temperature"],
        auto_class_weights="Balanced",
        eval_metric="AUC",
        random_seed=seed,
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
    )


CATBOOST_SPACE = {
    "learning_rate": loguniform(0.05, 0.3),
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": loguniform(1, 30),
    "bagging_temperature": uniform(0, 1),
}


def fit_catboost_es(model, Z_tr, y_tr, Z_es, y_es):
    model.fit(Z_tr, y_tr, eval_set=(Z_es, y_es),
              early_stopping_rounds=ES_ROUNDS, use_best_model=True)


# name: (function, search space, needs scaling)
MODELS = {
    "majority": (make_majority, None, False),
    "logreg": (make_logreg, LOGREG_SPACE, True),
    "rf": (make_rf, RF_SPACE, False),
    "lightgbm": (make_lightgbm, LIGHTGBM_SPACE, False),
    "catboost": (make_catboost, CATBOOST_SPACE, False),
}

# models that use early stopping: name -> function(model, Z_train, y_train, Z_stop, y_stop)
# (Member 3 adds "mlp" here)
EARLY_STOP_FITTERS = {
    "lightgbm": fit_lightgbm_es,
    "catboost": fit_catboost_es,
}