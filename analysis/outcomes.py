"""The two study endpoints, defined once.

Both are derived from the ``targets`` array of the preprocessed extract, whose
columns are ``[patient_id, index_date, observation_months, baseline_eGFR,
horizon_months, horizon_eGFR, horizon_CKD_stage]``.

Both endpoints are evaluated at the prediction date, Delta months after the
index date, for Delta in {12, 36, 60, 120}.

CKD-stage progression
    Positive when the CKD stage at the horizon is above the baseline band,
    stages following the KDIGO GFR categories. Encoded as ``stage > 0`` in the
    target column.

eGFR decline
    Positive when eGFR at the horizon has fallen to <= 70% of baseline, i.e. a
    relative decline of at least 30%.

``horizon_eGFR`` is the monthly-aggregated endpoint value built in
preprocessing: same-day duplicates are median-aggregated, and where several
daily values fall in the same relative month bin the last one is kept.
"""

from __future__ import annotations

import numpy as np

#: Column indices into the preprocessed ``targets`` array.
COL_BASELINE_EGFR = 3
COL_HORIZON_EGFR = 5
COL_STAGE = -1

EGFR_DECLINE_RATIO = 0.7


def ckd_progression(targets) -> np.ndarray:
    """Binary CKD-stage-progression label (1 = progressed)."""
    stage = np.asarray(targets)[:, COL_STAGE].astype(float)
    return (stage > 0).astype(int)


def egfr_decline(targets, ratio: float = EGFR_DECLINE_RATIO) -> np.ndarray:
    """Binary sustained-eGFR-decline label (1 = >= 30% relative decline)."""
    targets = np.asarray(targets)
    horizon = targets[:, COL_HORIZON_EGFR].astype(float)
    baseline = targets[:, COL_BASELINE_EGFR].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_obs = np.divide(horizon, baseline,
                              out=np.full_like(horizon, np.nan),
                              where=baseline != 0)
    return (ratio_obs <= ratio).astype(int)


#: Endpoint name -> labelling function, for scripts that accept ``--endpoint``.
ENDPOINTS = {
    "ckd_progression": ckd_progression,
    "egfr_decline": egfr_decline,
}


def label(targets, endpoint: str) -> np.ndarray:
    """Dispatch to the requested endpoint's labelling function."""
    if endpoint not in ENDPOINTS:
        raise SystemExit(
            f"Unknown endpoint {endpoint!r}; choose from {sorted(ENDPOINTS)}.")
    return ENDPOINTS[endpoint](targets)
