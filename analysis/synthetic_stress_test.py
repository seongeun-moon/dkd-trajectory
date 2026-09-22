"""Fully-generative synthetic validation, and the documented reason it fails.

Reviewers asked whether external validity could be demonstrated on synthetic
data in lieu of a second cohort. This script implements that: it fits an SDV
Gaussian-Copula synthesiser to the training feature matrix, samples a synthetic
cohort the size of the test set, applies distribution shifts to it, and scores
the real trained baselines on the result.

The outcome is a negative result, and it is reported as such: baseline AUROC on
the synthetic cohort collapses to ~0.55 against ~0.82 on the real test set. A
copula that reproduces the marginals and the linear correlation structure does
not reproduce the feature-outcome relationship for an outcome this rare, so a
synthetic cohort cannot substantiate external validity here. The targeted
perturbation test (``perturbation_stress_test.py``) is used instead.

Usage
-----
    python build_feature_matrix.py --data-root "$CKD_DATA_ROOT"
    python synthetic_stress_test.py
"""

from __future__ import annotations

import argparse
import time
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
from models import train_tree_baselines

warnings.filterwarnings("ignore")

CONFIG = Path(__file__).resolve().parent / "configs" / "baselines.yaml"
ALWAYS_OBSERVED = ("age", "sex", "BMI")

#: Panel order in the output figure.
PANEL_ORDER = ["real_test", "synth_baseline", "age+10y", "HbA1c+20%",
               "sex_flipped", "sparse50%", "eGFR-15%"]


def shift_age(df, names, rng, years=10):
    """Age the synthetic cohort (features are min-max scaled: +10 y ~ +0.10)."""
    out = df.copy()
    out["age"] = np.clip(out["age"] + years / 100, 0, 1)
    return out


def shift_hba1c(df, names, rng, factor=1.2):
    """Worsen glycaemic control, moving HbA1c and fasting glucose together."""
    out = df.copy()
    for column in ("HbA1c", "FBS"):
        if column in out:
            out[column] = np.clip(out[column] * factor, 0, 1)
    return out


def flip_sex(df, names, rng):
    """Invert the sex indicator, i.e. a cohort of the opposite sex mix."""
    out = df.copy()
    out["sex"] = 1 - out["sex"]
    return out


def sparse_measurements(df, names, rng, keep=0.5):
    """Blank a fraction of laboratory values, keeping demographics intact."""
    out = df.copy()
    for column in names:
        if column in ALWAYS_OBSERVED:
            continue
        out.loc[rng.random(len(out)) > keep, column] = 0
    return out


def shift_egfr(df, names, rng, delta=-0.15):
    """Lower baseline kidney function across the synthetic cohort."""
    out = df.copy()
    if "eGFR" in out:
        out["eGFR"] = np.clip(out["eGFR"] + delta, 0, 1)
    return out


SHIFTS = [
    ("age+10y", shift_age),
    ("HbA1c+20%", shift_hba1c),
    ("sex_flipped", flip_sex),
    ("sparse50%", sparse_measurements),
    ("eGFR-15%", shift_egfr),
]


def fit_synthesiser(Xtr, ytr, ttr, names):
    """Fit an SDV Gaussian-Copula synthesiser to the training matrix."""
    try:
        from sdv.metadata import Metadata
        from sdv.single_table import GaussianCopulaSynthesizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "This script needs SDV (`pip install 'sdv>=1.3'`).") from exc

    frame = pd.DataFrame(Xtr, columns=names)
    frame["horizon_code"] = ttr[:, 1].astype(int)
    frame["target"] = ytr

    metadata = Metadata.detect_from_dataframe(data=frame)
    for column in names:
        metadata.update_column(column_name=column, sdtype="numerical")
    for column in ("horizon_code", "target"):
        metadata.update_column(column_name=column, sdtype="categorical")

    synthesiser = GaussianCopulaSynthesizer(metadata)
    started = time.time()
    synthesiser.fit(frame)
    print(f"  synthesiser fitted in {time.time() - started:.1f}s")
    return synthesiser


def score_frame(frame, models, horizons, names, scenario, min_positives):
    """Score the trained baselines on a synthetic (or shifted) cohort."""
    rows = []
    for code, horizon in horizons.items():
        sub = frame[frame["horizon_code"] == code]
        if len(sub) < 20 or sub["target"].sum() < min_positives:
            continue
        y = sub["target"].astype(int).to_numpy()
        X = sub[names].to_numpy()
        for name in ("lgbm", "xgb"):
            model = models.get((name, horizon))
            if model is None:
                continue
            p = model.predict_proba(X)[:, 1]
            rows.append(dict(
                scenario=scenario, model=name, horizon=horizon, n=len(sub),
                pos=int(y.sum()), prev=float(y.mean()),
                auroc=float(roc_auc_score(y, p)),
                auprc=float(average_precision_score(y, p))))
    return rows


def score_real(Xte, yte, tte, models, horizons, min_positives):
    """Reference row set: the same models on the untouched real test data."""
    rows = []
    for code, horizon in horizons.items():
        idx = tte[:, 1] == code
        y = yte[idx]
        if y.sum() < min_positives:
            continue
        for name in ("lgbm", "xgb"):
            model = models.get((name, horizon))
            if model is None:
                continue
            p = model.predict_proba(Xte[idx])[:, 1]
            rows.append(dict(
                scenario="real_test", model=name, horizon=horizon,
                n=int(idx.sum()), pos=int(y.sum()), prev=float(y.mean()),
                auroc=float(roc_auc_score(y, p)),
                auprc=float(average_precision_score(y, p))))
    return rows


def plot(df, horizons, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, horizon in zip(axes.flat, horizons):
        sub = df[df.horizon == horizon]
        pivot = sub.pivot_table(index="scenario", columns="model", values="auroc")
        pivot = pivot.reindex([s for s in PANEL_ORDER if s in pivot.index])
        pivot.plot(kind="bar", ax=ax, rot=30, edgecolor="black")
        ax.set_title(f"{horizon} horizon")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.5, 1.0)
        ax.grid(alpha=0.3)
    fig.suptitle("Synthetic distribution-shift stress test (tree baselines, "
                 "CKD progression)", fontsize=13, weight="bold")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-positives", type=int, default=3)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

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
    print(f"features: {len(names)}, train {Xtr.shape}, test {Xte.shape}")

    models = train_tree_baselines(Xtr, ytr, ttr, horizons, config,
                                  args.min_positives)
    print(f"fitted {len(models)} models across {len(horizons)} horizons")

    synthesiser = fit_synthesiser(Xtr, ytr, ttr, names)
    synthetic = synthesiser.sample(num_rows=len(Xte))
    print(f"  sampled {len(synthetic)} synthetic rows")

    rng = np.random.default_rng(args.seed)
    rows = score_real(Xte, yte, tte, models, horizons, args.min_positives)
    rows += score_frame(synthetic, models, horizons, names, "synth_baseline",
                        args.min_positives)
    for scenario, shift in SHIFTS:
        rows += score_frame(shift(synthetic, names, rng), models, horizons,
                            names, scenario, args.min_positives)
        print(f"  evaluated {scenario}")

    df = pd.DataFrame(rows)
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_dir / "synthetic_stress_test_with_real.csv", index=False)
    df[df.scenario != "real_test"].to_csv(
        table_dir / "synthetic_stress_test.csv", index=False)

    pivot = df.pivot_table(index=["model", "horizon"], columns="scenario",
                           values="auroc").round(3)
    pivot.to_csv(table_dir / "synthetic_stress_pivot.csv")
    print(f"  wrote {table_dir / 'synthetic_stress_pivot.csv'}")

    if not args.no_figures:
        plot(df, list(horizons.values()),
             out_dir / "figures" / "synthetic_stress_test.png")

    print()
    print(pivot.to_string())

    real = df[df.scenario == "real_test"]["auroc"].mean()
    synth = df[df.scenario == "synth_baseline"]["auroc"].mean()
    print(f"\nMean AUROC on real test data:      {real:.3f}")
    print(f"Mean AUROC on synthetic cohort:    {synth:.3f}")
    print("The gap is the negative result reported in the response letter: a "
          "copula-generated cohort does not preserve the feature-outcome "
          "relationship for this rare outcome, so it cannot stand in for "
          "external validation.")


if __name__ == "__main__":
    main()
