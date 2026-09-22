"""Per-horizon AUROC / recall / F1 for a run directory, averaged over seeds."""

import os

import joblib
import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from paths import DATA_ROOT
from metrics import CONDITION_MONTHS, REPORTED_MONTHS
from results import RUNS, SEEDS, load_result, run_dir

THRESHOLD = 0.5


def score(results, idx):
    """AUROC / recall / F1 at one horizon, one value per seed."""
    aucs, recalls, f1s = [], [], []
    for result in results:
        prediction = np.array(result['prediction'])
        prediction = prediction[idx, 1] if prediction.ndim == 2 else prediction[idx]
        label = np.array(result['target'])[idx]

        aucs.append(roc_auc_score(label, prediction))
        predicted_class = [1 if p >= THRESHOLD else 0 for p in prediction]
        recalls.append(recall_score(label, predicted_class))
        f1s.append(f1_score(label, predicted_class))
    return np.array(aucs), np.array(recalls), np.array(f1s)


def report(name, template, cdt):
    """Print the per-horizon table for one run, averaged over SEEDS."""
    results = [load_result(run_dir(template, seed)) for seed in SEEDS]

    print(f'\n{name}')
    all_recalls, all_f1s = [], []
    for time_cdt in np.unique(cdt[:, 1]):
        aucs, recalls, f1s = score(results, cdt[:, 1] == time_cdt)
        if CONDITION_MONTHS[time_cdt] in REPORTED_MONTHS:
            print(f'  {CONDITION_MONTHS[time_cdt] :>4}m: AUC {aucs.mean():.2f} '
                  f'Recall {recalls.mean():.2f} F1 {f1s.mean():.2f}')
        all_recalls.append(recalls.mean())
        all_f1s.append(f1s.mean())
    print(f'  mean over horizons: recall {np.mean(all_recalls):.3f} '
          f'F1 {np.mean(all_f1s):.3f}')


def main():
    cdt = joblib.load(os.path.join(DATA_ROOT, 'reduced24_frequency_test.pkl'))['times']
    for name, template in RUNS.items():
        report(name, template, cdt)


if __name__ == '__main__':
    main()
