# metrics
# main metric is recall at 5% false positive rate (same as the BAF paper)
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from .config import FPR_TARGET


def threshold_at_fpr(y_true, scores, fpr=FPR_TARGET):
    # threshold where only fpr (5%) of the non-fraud rows have a higher score
    y_true = np.asarray(y_true)
    neg_scores = np.asarray(scores)[y_true == 0]
    return float(np.quantile(neg_scores, 1 - fpr, method="higher"))


def confusion_at(y_true, scores, thr):
    y_true = np.asarray(y_true)
    pred = np.asarray(scores) > thr
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    return tp, fp, fn, tn


def evaluate(y_true, scores, thr=None, fpr=FPR_TARGET):
    # all the metrics
    # if thr is None the threshold is set on the same rows (used for the cv folds)
    # for the test months the threshold from month 5 is passed in
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if thr is None:
        thr = threshold_at_fpr(y_true, scores, fpr)
    tp, fp, fn, tn = confusion_at(y_true, scores, thr)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "recall_at_5fpr": recall,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "accuracy": (tp + tn) / len(y_true),
        "threshold": thr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
