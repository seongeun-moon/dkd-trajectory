"""Locating and loading the archived per-run prediction files."""

import json
import os

import joblib

from paths import RESULT_ROOT

SEEDS = ("2022", "2023", "2024")

#: run directories to read, by the template their sessions are named with
RUNS = {
    'transformer': 'performer_l1h4e64lr0.0001p20_{seed}',
    'lightgbm': 'lightgbm_average_time_cdt_{seed}',
    'xgboost': 'xgb_average_time_cdt_{seed}',
}

#: a run directory stores its predictions under whichever of these it has
RESULT_FILES = ('result.json', 'best_valid_loss_result.pkl', 'test_result.pkl')


def run_dir(template, seed):
    """Absolute path of one run's session directory."""
    return os.path.join(RESULT_ROOT, template.format(seed=seed))


def load_result(path):
    """Read one run's predictions, whichever of RESULT_FILES it carries."""
    for name in RESULT_FILES:
        full = os.path.join(path, name)
        if not os.path.exists(full):
            continue
        if name.endswith('.json'):
            with open(full) as fh:
                return json.load(fh)
        return joblib.load(full)
    raise SystemExit(f'No result file in {path}; expected one of {RESULT_FILES}.')
