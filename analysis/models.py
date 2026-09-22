"""Fitting the tabular baselines on the mean-aggregated feature matrix.

Shared by the eGFR-decline retraining and by both stress tests, so a "baseline
model" means the same thing in each of them. Hyperparameters come from
``configs/baselines.yaml``.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import xgboost as xgb


def resolve_scale_pos_weight(params: dict, y) -> dict:
    """Replace ``scale_pos_weight: auto`` with the observed neg/pos ratio.

    Event prevalence ranges from ~1.2% at 1 year to ~8.4% at 10 years, so the
    ratio is computed per horizon rather than fixed.
    """
    params = dict(params)
    if params.get("scale_pos_weight") == "auto":
        positives = max(int(np.asarray(y).sum()), 1)
        params["scale_pos_weight"] = (len(y) - positives) / positives
    return params


def make_lightgbm(config: dict):
    return lgb.LGBMClassifier(**config["lightgbm"])


def make_xgboost(config: dict, y):
    return xgb.XGBClassifier(**resolve_scale_pos_weight(config["xgboost"], y))


def train_tree_baselines(X, y, t, horizons: dict, config: dict,
                         min_positives: int = 3) -> dict:
    """Fit LightGBM and XGBoost per horizon.

    Returns ``{(model_name, horizon_label): fitted_estimator}``; horizons with
    fewer than ``min_positives`` events are skipped.
    """
    models = {}
    for code, horizon in horizons.items():
        idx = t[:, 1] == code
        y_h = y[idx]
        if y_h.sum() < min_positives:
            continue
        models[("lgbm", horizon)] = make_lightgbm(config).fit(X[idx], y_h)
        models[("xgb", horizon)] = make_xgboost(config, y_h).fit(X[idx], y_h)
    return models
