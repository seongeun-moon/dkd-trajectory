"""Cohort definition shared by the segmentation and tokenisation steps.

These values define the dataset the pipeline builds. ``DATA_TYPE`` is derived
from them rather than written out separately, so the directory name always
describes the data inside it.

Every patient contributes one observation window per index date, and the
outcome is evaluated at a prediction date Delta months later. Windows whose eGFR
falls below MIN_BASELINE_EGFR, or that hold fewer than MIN_EGFR_MEASUREMENTS
eGFR values, are excluded.
"""

#: Observation-window lengths in months.
OBSERVATION_WINDOWS = (12,)

#: Prediction horizons Delta in months, measured from the index date.
PREDICTION_INTERVALS = (12, 36, 60, 120)

#: Minimum number of non-missing eGFR measurements an observation window must
#: contain.
MIN_EGFR_MEASUREMENTS = 3

#: Slices whose eGFR drops below this at any point in the observation window are
#: excluded. The retained cohort is named ``normal``.
MIN_BASELINE_EGFR = 60
COHORT = 'normal'

#: Name of the dataset, and of the directory holding it. Derived so it cannot
#: disagree with the parameters above: (12,) + 'normal' -> '12_normal'.
DATA_TYPE = '_'.join(str(w) for w in OBSERVATION_WINDOWS) + '_' + COHORT
