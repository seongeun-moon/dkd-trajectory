"""Evaluation metrics shared by the reproduction scripts.

Every script here computes discrimination, calibration and
threshold-dependent metrics through these functions, so the numbers reported for
the transformer, the tree baselines and the stress tests are produced by exactly
the same code path.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)

#: Horizon condition code in ``targets``/``conditions`` -> horizon label.
HORIZONS = {2: "1y", 3: "3y", 4: "5y", 6: "10y"}

#: Horizon label -> filename suffix used by the archived prediction CSVs.
HORIZON_SUFFIX = {"1y": "12m", "3y": "36m", "5y": "60m", "10y": "120m"}

#: Every condition code the extract carries -> months. Codes 0/1/5 exist but
#: are not reported (6 mo, 6-12 mo pooled, and 7 y respectively).
CONDITION_MONTHS = {0: None, 1: 6, 2: 12, 3: 36, 4: 60, 5: 84, 6: 120}

#: The reported horizons, in months.
REPORTED_MONTHS = (12, 36, 60, 120)

DEFAULT_BOOTSTRAP = 300


def bootstrap_ci(y, p, fn, n_boot: int = DEFAULT_BOOTSTRAP, seed: int = 0,
                 alpha: float = 0.05):
    """Percentile bootstrap CI for a metric ``fn(y, p)``.

    Resamples with replacement; draws in which the outcome is constant are
    skipped because AUROC/AUPRC are undefined there.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        try:
            stats.append(fn(yb, pb))
        except ValueError:
            continue
    if not stats:
        return float("nan"), float("nan")
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


def confusion_at(y, p, threshold: float = 0.5):
    """Return ``(tp, fp, tn, fn)`` for a hard threshold on the risk score."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def threshold_metrics(y, p, threshold: float = 0.5) -> dict:
    """Recall/precision/specificity/NPV/F1 at a fixed threshold."""
    tp, fp, tn, fn = confusion_at(y, p, threshold)
    div = lambda num, den: num / den if den else float("nan")  # noqa: E731
    return dict(
        recall=div(tp, tp + fn),
        precision=div(tp, tp + fp),
        specificity=div(tn, tn + fp),
        npv=div(tn, tn + fn),
        f1=div(2 * tp, 2 * tp + fp + fn),
    )


#: Recall the matched operating point is read at, for the F1 column of
#: the reported tables ("F1 at recall 0.8").
MATCHED_RECALL = 0.8


def f1_at_recall(y, p, target: float = MATCHED_RECALL) -> dict:
    """F1 at the operating point where recall equals ``target``.

    The reported tables give F1 twice: at the fixed 0.5 cutoff, and at a matched
    operating point, so that models are compared at equal sensitivity rather
    than at a cutoff their calibration may never reach. Precision at ``target``
    is read off the precision-recall curve by linear interpolation, and F1 is
    the harmonic mean of that precision and ``target``.

    Returns NaNs when ``target`` lies outside the recall range the model's
    scores can produce.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    nan = dict(f1_at_recall=float("nan"), precision_at_recall=float("nan"),
               threshold_at_recall=float("nan"))
    if not 0 < y.sum() < len(y):
        return nan
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    if len(recall) == 0 or not recall.min() <= target <= recall.max():
        return nan
    order = np.argsort(recall)
    prec_at = float(np.interp(target, recall[order], precision[order]))
    thr_at = float(np.interp(target, recall[order], thresholds[order]))
    denom = prec_at + target
    return dict(
        f1_at_recall=float(2 * prec_at * target / denom) if denom else 0.0,
        precision_at_recall=prec_at,
        threshold_at_recall=thr_at,
    )


def evaluate(y, p, threshold: float = 0.5, n_boot: int = DEFAULT_BOOTSTRAP,
             seed: int = 0) -> dict:
    """Full metric block for one model x horizon.

    Counts, AUROC/AUPRC with bootstrap CIs, Brier score, the operating
    characteristics at ``threshold``, and F1 at the matched recall-0.8 operating
    point -- together the columns of the reported tables."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = dict(n=int(len(y)), pos=int(y.sum()), prev=float(y.mean()))
    if 0 < y.sum() < len(y):
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
        out["auroc_lo"], out["auroc_hi"] = bootstrap_ci(
            y, p, roc_auc_score, n_boot=n_boot, seed=seed)
        out["auprc_lo"], out["auprc_hi"] = bootstrap_ci(
            y, p, average_precision_score, n_boot=n_boot, seed=seed)
    else:
        for key in ("auroc", "auprc", "auroc_lo", "auroc_hi",
                    "auprc_lo", "auprc_hi"):
            out[key] = float("nan")
    out["brier"] = float(brier_score_loss(y, p))
    out.update(threshold_metrics(y, p, threshold))
    out.update(f1_at_recall(y, p))
    return out


def per_horizon(preds, targets, conditions, threshold: float = 0.5,
                n_boot: int = DEFAULT_BOOTSTRAP, min_positives: int = 3):
    """Evaluate a prediction vector split by horizon condition code.

    ``preds`` may be a 1-D risk vector or a 2-D softmax output (column 1 used).
    """
    preds = np.asarray(preds)
    risk = preds[:, 1] if preds.ndim == 2 else preds
    targets = np.asarray(targets).astype(int)
    conditions = np.asarray(conditions)

    rows = []
    for code, label in HORIZONS.items():
        idx = conditions == code
        if idx.sum() == 0 or targets[idx].sum() < min_positives:
            continue
        row = dict(horizon=label)
        row.update(evaluate(targets[idx], risk[idx], threshold, n_boot))
        rows.append(row)
    return rows


def auroc_by_horizon(preds, targets, conditions) -> dict:
    """AUROC per horizon plus a pooled ``overall`` value (permutation tests)."""
    preds = np.asarray(preds)
    risk = preds[:, 1] if preds.ndim == 2 else preds
    targets = np.asarray(targets).astype(int)
    conditions = np.asarray(conditions)

    out = {}
    for code, label in HORIZONS.items():
        idx = conditions == code
        if 0 < targets[idx].sum() < idx.sum():
            out[label] = float(roc_auc_score(targets[idx], risk[idx]))
    out["overall"] = (float(roc_auc_score(targets, risk))
                      if 0 < targets.sum() < len(targets) else float("nan"))
    return out
