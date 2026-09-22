"""Recompute the reported evaluation metrics from archived prediction files.

Consumes the per-model per-seed prediction CSVs produced by the original
training runs (5 models x 3 seeds x 4 horizons) and regenerates:

* the seed-averaged metric table behind the reported table (CKD-stage
  progression, test set) -- AUROC, AUPRC, Brier, recall, precision,
  specificity, NPV, F1 at the 0.5 cutoff and F1 at the matched recall-0.8
  operating point, with bootstrap CIs;
* the same metrics per seed, which the table's seed averages are taken over;
* the calibration curves of Figure 6A and the decision curves of Figure 6B
  (threshold probabilities 0.01-0.5, the method), plus precision-recall curves.

The prediction CSVs are individual-level records derived from the restricted
SNUBH extract, so they are **not** distributed with this repository. Point
``--raw-results`` (or ``$CKD_RAW_RESULTS``) at your local copy; each file is
expected at ``<raw_results>/<dir>/<prefix>_<seed>_<horizon>.csv`` with
``target`` and ``prediction`` columns.

Usage
-----
    python compute_metrics.py --raw-results /path/to/raw_results \
                                      --out-dir work/reproduction
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import paths
import plots
from metrics import (HORIZON_SUFFIX, evaluate,
                              threshold_metrics)

CONFIG = Path(__file__).resolve().parent / "configs" / "baselines.yaml"
TRANSFORMER_CONFIG = Path(__file__).resolve().parent / "configs" / "transformer_performer.yaml"
HORIZON_LABELS = list(HORIZON_SUFFIX)


def load_predictions(root: Path, model: dict, seed: int, horizon: str) -> pd.DataFrame:
    """Read one archived prediction CSV."""
    path = (root / model["dir"] /
            f"{model['prefix']}_{seed}_{HORIZON_SUFFIX[horizon]}.csv")
    if not path.exists():
        raise SystemExit(
            f"Missing archived predictions: {path}\n"
            f"Check --raw-results / $CKD_RAW_RESULTS (see DATA_ACCESS.md).")
    df = pd.read_csv(path)
    missing = {"target", "prediction"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing column(s) {sorted(missing)}.")
    return df


def collect(root: Path, models: list[dict], seeds: list[int], threshold: float,
            n_boot: int):
    """Evaluate every model x horizon, per seed and pooled across seeds.

    Point estimates are the mean over seeds, matching how the study reports
    them; the confidence intervals bootstrap the seed-pooled predictions.
    """
    pooled_rows, per_seed_rows, pooled_predictions = [], [], {}

    for model in models:
        label = model["label"]
        for horizon in HORIZON_LABELS:
            ys, ps, seed_rows = [], [], []
            for seed in seeds:
                df = load_predictions(root, model, seed, horizon)
                y = df["target"].astype(int).to_numpy()
                p = df["prediction"].to_numpy(dtype=float)
                ys.append(y)
                ps.append(p)
                row = dict(model=label, seed=seed, horizon=horizon)
                row.update(evaluate(y, p, threshold, n_boot=0))
                seed_rows.append(row)
            per_seed_rows.extend(seed_rows)

            y_all = np.concatenate(ys)
            p_all = np.concatenate(ps)
            pooled_predictions[(label, horizon)] = (y_all, p_all)
            pooled = evaluate(y_all, p_all, threshold, n_boot=n_boot)

            row = dict(
                model=label, horizon=horizon,
                n=int(len(ys[0])), pos=int(ys[0].sum()), prev=float(ys[0].mean()),
                auroc=float(np.mean([r["auroc"] for r in seed_rows])),
                auroc_std=float(np.std([r["auroc"] for r in seed_rows])),
                auroc_lo=pooled["auroc_lo"], auroc_hi=pooled["auroc_hi"],
                auprc=float(np.mean([r["auprc"] for r in seed_rows])),
                auprc_std=float(np.std([r["auprc"] for r in seed_rows])),
                auprc_lo=pooled["auprc_lo"], auprc_hi=pooled["auprc_hi"],
                brier=float(np.mean([r["brier"] for r in seed_rows])),
            )
            row.update({f"{k}_t{threshold:g}": v for k, v in
                        threshold_metrics(y_all, p_all, threshold).items()})
            pooled_rows.append(row)

    return pd.DataFrame(pooled_rows), pd.DataFrame(per_seed_rows), pooled_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    paths.add_data_arguments(parser, raw_results=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Operating threshold (default: from config).")
    parser.add_argument("--bootstrap", type=int, default=None,
                        help="Bootstrap resamples for CIs (default: from config).")
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    threshold = (args.threshold if args.threshold is not None
                 else config["evaluation"]["threshold"])
    n_boot = (args.bootstrap if args.bootstrap is not None
              else config["evaluation"]["bootstrap_resamples"])
    seeds = yaml.safe_load(TRANSFORMER_CONFIG.read_text())["seeds"]
    models = config["archived_models"]

    root = paths.raw_results_root(args.raw_results)
    out_dir = paths.work_dir(args.out_dir)
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading archived predictions from {root}")
    pooled, per_seed, predictions = collect(root, models, seeds, threshold, n_boot)

    pooled.to_csv(table_dir / "metrics_aggregated.csv", index=False)
    per_seed.to_csv(table_dir / "metrics_per_seed.csv", index=False)
    print(f"  wrote {table_dir / 'metrics_aggregated.csv'}")
    print(f"  wrote {table_dir / 'metrics_per_seed.csv'}")

    if not args.no_figures:
        labels = [m["label"] for m in models]
        plots.calibration_figure(
            predictions, labels, HORIZON_LABELS,
            "Calibration - CKD-stage progression (test set)",
            fig_dir / "calibration_ckd.png", n_bins=args.calibration_bins)
        plots.pr_figure(
            predictions, labels, HORIZON_LABELS,
            "Precision-recall - CKD-stage progression (test set)",
            fig_dir / "pr_curves_ckd.png")
        plots.dca_figure(
            predictions, labels, HORIZON_LABELS,
            "Decision-curve analysis - CKD-stage progression (test set)",
            fig_dir / "dca_ckd.png")

    display = ["model", "horizon", "n", "pos", "prev",
               "auroc", "auroc_lo", "auroc_hi", "auprc", "brier"]
    print()
    print(pooled[display].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
