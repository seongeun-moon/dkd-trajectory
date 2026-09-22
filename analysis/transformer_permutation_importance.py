"""Permutation importance for the archived transformer.

For each of the three seeds the archived
checkpoint is scored on the test set, then re-scored once per input variable
with that variable shuffled across the test set; the AUROC drop is the
variable's importance. The 3 seeds x 45 variables are the 42 frequently
recorded and 3 low-frequency laboratory variables; prescription tokens are left
alone, since shuffling a medication code across patients produces sequences that
could not occur.

.. warning::

   ``--endpoint`` must match the endpoint the checkpoint was trained for. The
   archived ``performer_l1h4e64lr0.0001p20_*`` runs were trained for CKD-stage
   progression, which is both this script's default and the endpoint the study
   reports these importances for.

Usage
-----
    python transformer_permutation_importance.py \
        --data-root "$CKD_DATA_ROOT" --ckpt-root "$CKD_CKPT_ROOT"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

import outcomes
import paths
import transformer
from metrics import HORIZON_SUFFIX, auroc_by_horizon

CONFIG = Path(__file__).resolve().parent / "configs" / "transformer_performer.yaml"


def permuted_copy(data: dict, category: str, column: int, seed: int) -> dict:
    """Copy ``data`` with one variable shuffled across patients.

    ``category_data['inputs'][category]`` is an ``(n, t, 1 + n_vars)`` array
    whose column 0 is the visit time offset, so variable *i* lives at column
    ``i + 1``. Shuffling whole patient trajectories preserves each variable's
    marginal distribution and its within-patient temporal structure while
    destroying its association with the outcome.
    """
    inputs = data["category_data"]["inputs"]
    shuffled = {key: (value.copy() if hasattr(value, "copy") else value)
                for key, value in inputs.items()}
    array = shuffled[category]
    order = np.random.default_rng(seed).permutation(len(array))
    array[:, :, column] = array[order, :, column]

    category_data = dict(data["category_data"])
    category_data["inputs"] = shuffled
    return {**data, "category_data": category_data}


def variable_columns(key_info: dict) -> list[tuple[str, str, int]]:
    """``(category, variable name, column index)`` for every permutable input."""
    entries = []
    for category, key in (("usual", "usual_categories"),
                          ("unusual", "unusual_categories")):
        for offset, name in enumerate(list(key_info[key])):
            entries.append((category, name, offset + 1))
    return entries


def check_against_archive(probs, conditions, baseline, raw_results: Path,
                          seed: int, config: dict) -> dict:
    """Cross-check freshly computed scores against the archived predictions.

    Confirms that the checkpoint reloaded here reproduces the run that produced
    the study's numbers. A high correlation with a differing AUROC means
    the predictions match but the labels do not -- i.e. ``--endpoint`` disagrees
    with what the checkpoint was trained for.
    """
    prefix = config["run_name_template"].format(seed=seed).rsplit("_", 1)[0]
    report = {}
    for code, horizon in zip((2, 3, 4, 6), HORIZON_SUFFIX):
        suffix = HORIZON_SUFFIX[horizon]
        path = raw_results / "performer" / f"{prefix}_{seed}_{suffix}.csv"
        if not path.exists():
            continue
        archived = pd.read_csv(path)
        idx = conditions == code
        local = probs[idx, 1]
        report[suffix] = {
            "correlation": (float(np.corrcoef(local, archived.prediction)[0, 1])
                            if len(local) == len(archived) else float("nan")),
            "archived_auroc": float(roc_auc_score(archived.target,
                                                  archived.prediction)),
            "local_auroc": baseline.get(horizon, float("nan")),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    paths.add_data_arguments(parser, data=True, ckpt=True, raw_results=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--endpoint", default="ckd_progression",
                        choices=sorted(outcomes.ENDPOINTS),
                        help="Must match the checkpoint's training endpoint.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override the seeds listed in the config.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Independent shuffles per variable, averaged. Cost "
                             "is linear: 45 variables x 3 seeds x 1 repeat is "
                             "135 inference passes.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto",
                        help="auto (default), cpu, mps or cuda.")
    parser.add_argument("--check-archive", action="store_true",
                        help="Cross-check against the archived prediction CSVs "
                             "(needs --raw-results / $CKD_RAW_RESULTS).")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    seeds = args.seeds or config["seeds"]
    data_root = paths.data_root(args.data_root)
    ckpt_root = paths.ckpt_root(args.ckpt_root)
    out_dir = paths.work_dir(args.out_dir)
    out_path = out_dir / "permutation_importance_transformer.json"

    device = transformer.select_device(args.device)
    print(f"device: {device}   endpoint: {args.endpoint}")

    key_info = joblib.load(data_root / "key_info.pkl")
    variables = variable_columns(key_info)
    passes = len(variables) * args.repeats + 1
    print(f"permuting {len(variables)} variables x {len(seeds)} seeds "
          f"x {args.repeats} repeats ({passes} inference passes per seed)")

    test_data = transformer.load_split(data_root, "test",
                                       config["data"]["split_type"])
    dim_info = transformer.derive_dim_info(config, data_root, test_data)
    test_data["targets"][:, -1] = outcomes.label(test_data["targets"],
                                                 args.endpoint)

    results = {}
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        run_name = config["run_name_template"].format(seed=seed)
        model, _ = transformer.build_model(config, seed, device,
                                           dim_info=dim_info)
        transformer.load_checkpoint(model, ckpt_root, run_name)

        started = time.time()
        probs, targets, conditions = transformer.infer(
            model, test_data, device, args.batch_size)
        baseline = auroc_by_horizon(probs, targets, conditions)
        print(f"baseline inference in {time.time() - started:.1f}s; "
              f"AUROC {baseline}")

        entry = {"endpoint": args.endpoint, "baseline": baseline}
        if args.check_archive:
            entry["archive_check"] = check_against_archive(
                probs, conditions, baseline,
                paths.raw_results_root(args.raw_results), seed, config)
            print(f"archive check: {entry['archive_check']}")

        drops = {}
        for category, name, column in variables:
            # Average over independent shuffles; a single shuffle is a noisy
            # estimate of the AUROC drop, especially at the 10-year horizon
            # where only 38 events are available.
            repeats = []
            for r in range(args.repeats):
                perm_probs, perm_targets, perm_conditions = transformer.infer(
                    model,
                    permuted_copy(test_data, category, column, seed * 1000 + r),
                    device, args.batch_size)
                repeats.append(auroc_by_horizon(perm_probs, perm_targets,
                                                perm_conditions))
            scores = {h: float(np.mean([r[h] for r in repeats]))
                      for h in repeats[0]}
            if args.repeats > 1:
                scores.update({f"{h}_sd": float(np.std([r[h] for r in repeats]))
                               for h in repeats[0]})
            key = name if category == "usual" else f"unusual_{name}"
            drops[key] = scores
            print(f"  {key:>28}: overall {scores['overall']:.4f} "
                  f"(drop {baseline['overall'] - scores['overall']:+.4f})")

        entry["perm_aurocs"] = drops
        entry["repeats"] = args.repeats
        results[str(seed)] = entry
        # Written after every seed so a long run stays resumable/inspectable.
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"  wrote {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
