import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, precision_recall_curve,
                             roc_curve)

from paths import DATA_ROOT, OUT_ROOT
from metrics import CONDITION_MONTHS, REPORTED_MONTHS
from results import RUNS, SEEDS, load_result, run_dir


def export(path, cdt):
    """Write the per-horizon prediction CSVs and ROC / PR curves for one run."""
    result = load_result(path)

    for time_cdt in np.unique(cdt[:,1]):

        if CONDITION_MONTHS[time_cdt] in REPORTED_MONTHS:
            df = pd.DataFrame(columns=["prediction", "target"])

            idx = cdt[:, 1] == time_cdt
            prediction = np.array(result["prediction"])
            if len(prediction.shape) == 2:
                prediction = prediction[idx, 1]
            else:
                prediction = prediction[idx]

            label = np.array(result["target"])[idx]

            df["prediction"] = prediction
            df["target"] = label

            filename = path.strip("/").split("/")[-1]
            modelname = filename.split("_")[0]

            os.makedirs(os.path.join(OUT_ROOT, 'graphs', modelname), exist_ok=True)

            # auroc
            fpr, tpr, thresholds = roc_curve(df["target"], df["prediction"])

            roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
            roc_display.plot(color='black')
            plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend().remove()
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_ROOT, 'graphs', modelname, f"{filename}_{CONDITION_MONTHS[time_cdt]}m_roc.png"))

            # auprc
            precision, recall, thresholds = precision_recall_curve(df["target"], df["prediction"])
            pr_auc = average_precision_score(df["target"], df["prediction"])

            pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc)
            pr_display.plot(color='black')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend().remove()
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_ROOT, 'graphs', modelname, f"{filename}_{CONDITION_MONTHS[time_cdt]}m_pr.png"))

            # raw result
            os.makedirs(os.path.join(OUT_ROOT, 'raw_results', modelname), exist_ok=True)
            df.to_csv(os.path.join(OUT_ROOT, 'raw_results', modelname, f"{filename}_{CONDITION_MONTHS[time_cdt]}m.csv"), index=False, encoding="utf-8-sig")


def main():
    cdt = joblib.load(os.path.join(DATA_ROOT, 'reduced24_frequency_test.pkl'))['times']
    for template in RUNS.values():
        for seed in SEEDS:
            path = run_dir(template, seed)
            print(path)
            export(path, cdt)


if __name__ == '__main__':
    main()
