"""TreeSHAP feature attribution for the tree baselines.

Feature attribution for the two boosting
baselines is TreeSHAP with the tree-path-dependent estimator: exact Shapley
values for tree ensembles, in log-odds units, summarised as the mean absolute
attribution per variable at each horizon. The estimator uses the trained trees'
own leaf-sample distributions as its reference, so no background matrix is
drawn.

Produces, per model:
  ``shap_<model>_meanabs.csv``  variable x horizon mean |SHAP|, plus the mean
                               across horizons used to rank variables
  ``shap_<model>_metrics.csv``  the discrimination of the model each attribution
                                was computed from, so the ranking can be read
                                against the model's actual performance
  ``shap_<model>_by_horizon.png``

Usage
-----
    python build_feature_matrix.py --data-root "$CKD_DATA_ROOT"
    python shap_importance.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

import paths
from models import make_lightgbm, make_xgboost

warnings.filterwarnings("ignore")

CONFIG = Path(__file__).resolve().parent / "configs" / "baselines.yaml"
TOP_N_PLOT = 15


def mean_abs_shap(model, X, feature_names, background=None):
    """Mean |SHAP| per feature for the positive class.

    ``background=None`` uses TreeSHAP's tree_path_dependent perturbation, the
    estimator the study specifies: it takes its reference from the
    trained trees' own leaf-sample distributions and needs no reference data.
    Passing a background matrix switches to the interventional estimator, kept
    as a robustness check.
    """
    import shap

    explainer = (shap.TreeExplainer(model) if background is None
                 else shap.TreeExplainer(model, data=background,
                                         feature_perturbation="interventional"))
    values = explainer.shap_values(X)
    # LightGBM's binary classifier returns a list (one array per class) on some
    # versions and a single array on others; XGBoost returns a single array.
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:  # (n, features, classes)
        values = values[..., -1]
    return pd.Series(np.abs(values).mean(axis=0), index=feature_names)


def plot(table, model_label, out_path: Path, horizons):
    ranked = table.sort_values("mean_across_horizons", ascending=False)
    top = ranked.head(TOP_N_PLOT)[list(horizons)][::-1]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(top) + 2))
    top.plot(kind="barh", ax=ax, edgecolor="black", width=0.8)
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title(f"{model_label}: TreeSHAP attribution by horizon "
                 f"(top {TOP_N_PLOT})")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(title="horizon", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    paths.add_data_arguments(parser)
    parser.add_argument("--features", type=Path, default=None,
                        help="features_tree.npz (default: <out-dir>/features_tree.npz).")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--models", nargs="+", default=["lgbm", "xgb"],
                        choices=["lgbm", "xgb"])
    parser.add_argument("--min-positives", type=int, default=3)
    parser.add_argument("--background", choices=["none", "train"], default="none",
                        help="TreeSHAP reference. 'none' (default) is the "
                             "tree_path_dependent estimator the study "
                             "specifies; 'train' switches to an "
                             "interventional background sampled from the "
                             "training split, as a robustness check.")
    parser.add_argument("--background-size", type=int, default=1000,
                        help="Rows sampled for --background train.")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    try:
        import shap  # noqa: F401
    except ImportError:
        raise SystemExit("This script needs SHAP (`pip install shap`).")

    config = yaml.safe_load(args.config.read_text())
    horizons = {int(k): v for k, v in config["horizons"].items()}

    out_dir = paths.work_dir(args.out_dir)
    features = args.features or (out_dir / "features_tree.npz")
    if not Path(features).exists():
        raise SystemExit(f"Feature matrix not found: {features}\n"
                         f"Build it with build_feature_matrix.py first.")

    cached = np.load(features, allow_pickle=True)
    Xtr, ttr = cached["Xtr"], cached["ttr"]
    Xte, tte = cached["Xte"], cached["tte"]
    ytr = (cached["ytr"] > 0).astype(int)
    yte = (cached["yte"] > 0).astype(int)
    names = list(cached["feature_names"])

    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    labels = {"lgbm": "LightGBM", "xgb": "XGBoost"}
    for key in args.models:
        columns, metric_rows = {}, []
        for code, horizon in horizons.items():
            idx_tr, idx_te = ttr[:, 1] == code, tte[:, 1] == code
            y_tr, y_te = ytr[idx_tr], yte[idx_te]
            if y_tr.sum() < args.min_positives or y_te.sum() < args.min_positives:
                continue
            model = (make_lightgbm(config) if key == "lgbm"
                     else make_xgboost(config, y_tr))
            model.fit(Xtr[idx_tr], y_tr)

            p = model.predict_proba(Xte[idx_te])[:, 1]
            metric_rows.append(dict(
                horizon=horizon, n_train=int(idx_tr.sum()),
                n_test=int(idx_te.sum()), prev_te=float(y_te.mean()),
                auroc=float(roc_auc_score(y_te, p)),
                auprc=float(average_precision_score(y_te, p))))

            background = None
            if args.background == "train":
                pool = Xtr[idx_tr]
                take = min(args.background_size, len(pool))
                rng = np.random.default_rng(0)
                background = pool[rng.choice(len(pool), take, replace=False)]
            columns[horizon] = mean_abs_shap(model, Xte[idx_te], names,
                                             background)
            print(f"  {labels[key]} {horizon}: SHAP over {idx_te.sum()} slices")

        table = pd.DataFrame(columns)
        table["mean_across_horizons"] = table.mean(axis=1)
        table = table.sort_values("mean_across_horizons", ascending=False)
        table.to_csv(table_dir / f"shap_{key}_meanabs.csv")
        pd.DataFrame(metric_rows).to_csv(
            table_dir / f"shap_{key}_metrics.csv", index=False)
        print(f"  wrote {table_dir / f'shap_{key}_meanabs.csv'}")
        print(f"  wrote {table_dir / f'shap_{key}_metrics.csv'}")

        if not args.no_figures:
            plot(table, labels[key],
                 out_dir / "figures" / f"shap_{key}_by_horizon.png",
                 columns.keys())

        print(f"\n=== {labels[key]}: top 10 by mean |SHAP| ===")
        print(table.head(10).round(4).to_string())
        print()


if __name__ == "__main__":
    main()
