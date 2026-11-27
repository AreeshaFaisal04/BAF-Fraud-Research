# preprocessing (sklearn pipeline, so it is only fitted on the training rows of each fold)
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import numeric_features, categorical_features


def build_preprocessor(features, scale=True, missing_indicators=True):
    # numbers: fill missing values with the median + add a 0/1 missing column, scale if needed
    # categories: one-hot encoding
    # scaling is only needed for logistic regression and the MLP
    num = numeric_features(features)
    cat = categorical_features(features)

    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=missing_indicators))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    transformers = [("num", Pipeline(num_steps), num)]
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def downsample_negatives(X, y, frac, seed):
    # keep all fraud rows and a random part of the non-fraud rows (only used on training data)
    if frac >= 1.0:
        return X, y
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    keep_neg = rng.choice(neg, size=int(round(len(neg) * frac)), replace=False)
    keep = np.sort(np.concatenate([pos, keep_neg]))
    return X.iloc[keep], y[keep]
