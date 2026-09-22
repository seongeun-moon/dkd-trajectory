"""Retrain the transformer on a temporal split (train <= 2010, test >= 2011).

The internal proxy for prospective validation reported in the Discussion:
discrimination dropped by about 20 percentage points, empirically confirming
concept drift across the study window. The reported split is random at patient
level, so it cannot speak to temporal transportability. Here the same
architecture is trained only on slices whose index date falls in 2003-2010 and
evaluated on every slice from 2011-2014, giving a genuine forward-in-time test
for the CKD-progression endpoint.

.. warning::

   ``--test-source`` controls how the late cohort is assembled, and the two
   options are not equivalent.

   ``pooled`` (the default, and what produced the ~20-percentage-point drop
   quoted in the method) takes every slice with an index date after the cutoff, including
   slices belonging to patients whose *earlier* slices trained the model. A
   patient with visits on both sides of the cutoff therefore appears in both
   training and test. The split is clean in time but not in patients, so the
   result is a temporal-transportability probe rather than a fully held-out
   evaluation, and if anything it flatters the model.

   ``held-out`` restricts the late cohort to the original test partition, which
   is patient-disjoint from training by construction. It is the stricter
   comparison, on a much smaller test set.

This is a single-seed run on a reduced epoch budget (see
``configs/transformer_temporal_split.yaml``); compare it against the archived
seed-averaged numbers with that in mind.

Usage
-----
    python transformer_temporal_split.py --data-root "$CKD_DATA_ROOT"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import evaluation
import outcomes
import paths
import transformer

CONFIG = Path(__file__).resolve().parent / "configs" / "transformer_temporal_split.yaml"

#: Column 1 of ``targets`` is the index date of the observation window.
COL_INDEX_DATE = 1


def index_years(data: dict) -> np.ndarray:
    """Calendar year of each slice's index date."""
    return pd.to_datetime(data["targets"][:, COL_INDEX_DATE]).year.values


def subset(data: dict, mask: np.ndarray) -> dict:
    """Row-subset every per-slice array in a split, leaving metadata intact."""
    n = len(mask)

    def take(value):
        if isinstance(value, dict):
            return {k: take(v) for k, v in value.items()}
        if hasattr(value, "__len__") and len(value) == n:
            return value[mask]
        return value

    out = {key: take(value) for key, value in data.items()
           if key in ("category_data", "prs_data")}
    out["targets"] = data["targets"][mask]
    out["conditions"] = data["conditions"][mask]
    out["length_info"] = data.get("length_info")
    return out


def concat(a: dict, b: dict) -> dict:
    """Concatenate two splits slice-wise."""
    def join(x, y):
        if isinstance(x, dict):
            return {k: join(x[k], y[k]) for k in x}
        if hasattr(x, "shape"):
            return np.concatenate([x, y])
        return x

    return {k: (join(a[k], b[k]) if k in b and b[k] is not None else a[k])
            for k in a}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    paths.add_data_arguments(parser, data=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cutoff-year", type=int, default=2010,
                        help="Last index year used for training (inclusive).")
    parser.add_argument("--test-source", choices=("pooled", "held-out"),
                        default="pooled",
                        help="pooled: every post-cutoff slice, including those "
                             "from patients seen in training (reproduces the "
                             "reported number). held-out: post-cutoff slices "
                             "from the original test partition only, which is "
                             "patient-disjoint from training.")
    parser.add_argument("--device", default="auto",
                        help="auto (default), cpu, mps or cuda.")
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    seed = args.seed or config["seeds"][0]
    cutoff = args.cutoff_year
    data_root = paths.data_root(args.data_root)
    out_dir = paths.work_dir(args.out_dir) / "temporal_split_retrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = transformer.select_device(args.device)
    print(f"device: {device}   seed: {seed}   cutoff: <= {cutoff}")

    original = {name: transformer.load_split(data_root, name,
                                             config["data"]["split_type"])
                for name in ("train", "valid", "test")}
    years = {name: index_years(split) for name, split in original.items()}
    for name, ys in years.items():
        print(f"{name:>5} index years: "
              f"{pd.Series(ys).value_counts().sort_index().to_dict()}")

    # Training and validation keep their original partition membership; only
    # slices at or before the cutoff are used.
    early_train = subset(original["train"], years["train"] <= cutoff)
    early_valid = subset(original["valid"], years["valid"] <= cutoff)

    late_test = subset(original["test"], years["test"] > cutoff)
    if args.test_source == "pooled":
        for name in ("train", "valid"):
            late_test = concat(late_test,
                               subset(original[name], years[name] > cutoff))
        print("\nNOTE: --test-source pooled includes post-cutoff slices from "
              "patients whose earlier slices are in training. The split is "
              "clean in time but not in patients; use --test-source held-out "
              "for the patient-disjoint comparison.\n")

    for split in (early_train, early_valid, late_test):
        split["targets"][:, -1] = outcomes.ckd_progression(split["targets"])

    print(f"early train: {len(early_train['targets']):>6} slices "
          f"(prevalence {early_train['targets'][:, -1].mean():.4f})")
    print(f"early valid: {len(early_valid['targets']):>6} slices "
          f"(prevalence {early_valid['targets'][:, -1].mean():.4f})")
    print(f"late test  : {len(late_test['targets']):>6} slices "
          f"(prevalence {late_test['targets'][:, -1].mean():.4f})")

    dim_info = transformer.derive_dim_info(config, data_root,
                                           original["train"])
    model, model_args = transformer.build_model(config, seed, device,
                                                dim_info=dim_info)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: {n_params:.1f} M parameters")

    # het_trans owns the training loop; save_log=False keeps it headless (no
    # TensorBoard event files, no log.txt) but writes the same checkpoint.
    transformer._ensure_importable()
    from main_graph_snubh import train as het_train

    run_dir = out_dir / "temporal_split"
    run_dir.mkdir(parents=True, exist_ok=True)
    het_train(model_args, model, early_train, early_valid, late_test, device,
              result_path=str(run_dir), save_log=False)

    # train() leaves the model on the last epoch's weights; the best ones are
    # only on disk, so read them back before scoring.
    transformer.load_checkpoint(model, run_dir.parent, run_dir.name)

    rows, probs, targets, conditions = evaluation.test_report(
        model, late_test, device, model_args, n_boot=args.bootstrap)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "transformer_temporal_split_metrics.csv", index=False)
    np.savez(out_dir / "transformer_temporal_split_predictions.npz",
             probs=probs, targets=targets, conditions=conditions)

    print(f"\n=== Transformer, temporal split (train <= {cutoff}, "
          f"test > {cutoff}), seed {seed} ===")
    print(table[["horizon", "n", "pos", "prev", "auroc", "auprc", "brier"]]
          .round(3).to_string(index=False))
    print(f"\nwrote {out_dir}/")
    print("Predictions are individual-level derived data: keep them inside the "
          "work directory, which is git-ignored, and do not publish them.")


if __name__ == "__main__":
    main()
