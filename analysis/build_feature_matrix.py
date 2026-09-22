"""Flatten the longitudinal extract into the tabular matrix used by the baselines.

The transformer consumes irregular per-visit sequences directly. The tree and
linear baselines, the SHAP analysis and both stress tests instead operate on one
row per observation slice, obtained by averaging each variable over the visits
inside the slice's observation window -- the ``--agg average`` aggregation
described in the method

Layout of the produced ``features_tree.npz`` (written into the work directory,
which is git-ignored -- it is derived patient data and must not be published):

    X{tr,te}  (n_slices, 45)  mean-aggregated features, min-max scaled upstream
    y{tr,te}  (n_slices,)     CKD stage at the horizon (0-3); binarise as y > 0
    t{tr,te}  (n_slices, 2)   [observation-window months, horizon condition code]
    feature_names (45,)       42 frequently recorded + 3 low-frequency variables

Usage
-----
    python build_feature_matrix.py --data-root "$CKD_DATA_ROOT"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

import paths


def aggregate(sequences) -> np.ndarray:
    """Mean over the time axis of a ragged ``(n_visits, 1 + n_vars)`` sequence.

    Column 0 of each sequence is the visit time offset and is dropped; slices
    with no visits contribute a row of zeros.
    """
    rows = []
    for seq in sequences:
        arr = np.asarray(seq, dtype=float)
        if arr.size == 0:
            width = 0 if arr.ndim < 2 else arr.shape[1] - 1
            rows.append(np.zeros(width))
        else:
            rows.append(arr[:, 1:].mean(axis=0))
    return np.vstack(rows)


def build_split(data_root: Path, split: str, split_type: str = "frequency"):
    """Return ``(X, y, t)`` for one split."""
    path = data_root / f"reduced24_{split_type}_{split}.pkl"
    if not path.exists():
        raise SystemExit(f"Missing preprocessed split: {path}")
    data = joblib.load(path)
    usual = aggregate(data["inputs"]["usual"])
    unusual = aggregate(data["inputs"]["unusual"])
    X = np.hstack([usual, unusual])
    y = data["targets"][:, -1].astype(int)
    t = np.asarray(data["times"]).astype(int)
    return X, y, t


def feature_names(data_root: Path) -> np.ndarray:
    key_info = joblib.load(data_root / "key_info.pkl")
    names = list(key_info["usual_categories"]) + list(key_info["unusual_categories"])
    return np.asarray(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    paths.add_data_arguments(parser, data=True)
    parser.add_argument("--split-type", default="frequency")
    parser.add_argument(
        "--verify-against", type=Path, default=None,
        help="Optional existing features_tree.npz to check this build against.")
    args = parser.parse_args()

    data_root = paths.data_root(args.data_root)
    out_dir = paths.work_dir(args.out_dir)
    out_path = out_dir / "features_tree.npz"

    names = feature_names(data_root)
    Xtr, ytr, ttr = build_split(data_root, "train", args.split_type)
    Xte, yte, tte = build_split(data_root, "test", args.split_type)

    if Xtr.shape[1] != len(names):
        raise SystemExit(
            f"Feature-count mismatch: matrix has {Xtr.shape[1]} columns but "
            f"key_info.pkl names {len(names)} variables.")

    np.savez(out_path, Xtr=Xtr, ytr=ytr, ttr=ttr, Xte=Xte, yte=yte, tte=tte,
             feature_names=names)
    print(f"train {Xtr.shape}  test {Xte.shape}  ->  {out_path}")

    if args.verify_against:
        ref = np.load(args.verify_against, allow_pickle=True)
        for key, built in (("Xtr", Xtr), ("Xte", Xte), ("ytr", ytr),
                           ("yte", yte), ("ttr", ttr), ("tte", tte)):
            ok = np.allclose(ref[key], built, equal_nan=True)
            print(f"  {key}: {'match' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
