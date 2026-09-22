"""Construction, checkpoint loading and inference for the Performer model.

The model definition itself lives in ``het_trans/`` (the training code used
for the study, vendored unchanged apart from the removal of site-specific
data paths). This module is the thin, shared layer the reproduction scripts in
the analysis scripts use so that architecture, hyperparameters and the inference loop are
declared exactly once, in ``configs/transformer_performer.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from paths import REPO_ROOT

HET_TRANS = REPO_ROOT / "het_trans"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "transformer_performer.yaml"


def _ensure_importable():
    """Put the vendored training code on ``sys.path`` (idempotent)."""
    if str(HET_TRANS) not in sys.path:
        sys.path.insert(0, str(HET_TRANS))


def select_device(prefer: str = "auto") -> torch.device:
    """Resolve the compute device.

    ``auto`` prefers CUDA, then MPS, then CPU. Note that for this Performer at
    the study's batch size, CPU was measured faster than MPS on Apple
    silicon -- pass ``--device cpu`` if MPS throughput disappoints.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_args(config: dict, seed: int, **overrides) -> SimpleNamespace:
    """Assemble the ``args`` namespace the model constructor expects."""
    model_cfg = dict(config["model"])
    train_cfg = dict(config.get("training", {}))
    args = SimpleNamespace(**model_cfg, **train_cfg, seed=seed)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def derive_dim_info(config: dict, data_root=None, sample_data=None) -> dict:
    """Resolve the input/embedding dimensions the architecture is built from.

    These describe the *shape* of the preprocessed extract, so they are read
    from the data whenever the data is available, and fall back to the values
    recorded in the config -- which are those of the archived runs -- when it is
    not. Precedence:

    1. ``model_config.pkl`` beside the data, if present (written by the original
       training runs, and authoritative for their checkpoints);
    2. derived from ``key_info.pkl`` / ``prs_embed_dict.pkl`` and a loaded split,
       mirroring ``prepare_data()`` in ``het_trans/main_graph_snubh.py``;
    3. the ``dim_info`` block of the config.
    """
    if data_root is not None:
        data_root = Path(data_root)

        recorded = data_root / "model_config.pkl"
        if recorded.exists():
            import joblib
            return joblib.load(recorded)["dim_info"]

        prs_path = data_root / "prs_embed_dict.pkl"
        if sample_data is not None and prs_path.exists():
            import joblib
            key_info = joblib.load(data_root / "key_info.pkl")
            prs_info = joblib.load(prs_path)
            model_cfg = config["model"]
            feature_set = config["data"]["feature_set"]

            dim_info = {"importance_embed": 3, "obsn_embed": 2,
                        "time_embed": 2 if model_cfg["pos_encode"] else None}
            for category in sample_data["category_data"]["inputs"]:
                dim_info[f"{category}_input_dim"] = len(
                    key_info[f"{category}_categories"])
                dim_info[f"{category}_len"] = sample_data["length_info"][category]
            dim_info["prs_input_dim"] = sample_data["length_info"]["prs_num"]
            dim_info["prs_len"] = sample_data["length_info"]["prs"]
            dim_info["code_embed"] = len(prs_info["code_embed"])
            dim_info["unit_embed"] = len(prs_info["unit_embed"])
            dim_info["num_condition"] = max(
                3, len(np.unique(sample_data["conditions"][:, 1])) + 1)
            dim_info["input_dim"] = model_cfg["embed_dim"] * len(feature_set)
            if "prs" in feature_set:
                dim_info["input_dim"] += model_cfg["embed_dim"]
            dim_info["output_dim"] = len(
                np.unique(sample_data["targets"][:, -1]))
            if dim_info["time_embed"] is None:
                dim_info["time_embed"] = len(
                    joblib.load(data_root / "time_embed_dict.pkl"))
            return dim_info

    return config["dim_info"]


def build_model(config: dict, seed: int, device: torch.device,
                dim_info: dict | None = None, **overrides):
    """Instantiate the Performer with the study's architecture."""
    _ensure_importable()
    from model.het_perf_obsn_mask_snubh import trans_model  # noqa: E402

    torch.manual_seed(seed)
    np.random.seed(seed)
    args = build_args(config, seed, **overrides)
    dims = dim_info or config["dim_info"]
    return trans_model(args, config["categories"], dims).to(device), args


def load_checkpoint(model, ckpt_root: Path, run_name: str,
                    filename: str = "best_valid_loss_model.pt"):
    """Load an archived run's weights into ``model``.

    Checkpoints are trained on restricted data and are not distributed with this
    repository; ``ckpt_root`` is supplied at run time (see ``paths``).
    """
    path = Path(ckpt_root) / run_name / filename
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise SystemExit(
            f"{path} does not fit the architecture that was built.\n\n{exc}\n\n"
            f"This means the checkpoint was trained on a differently shaped "
            f"dataset (a different --data_suffix, prescription vocabulary or "
            f"set of prediction horizons). Point --data-root at the extract "
            f"that run used, so the dimensions are read from its "
            f"model_config.pkl / prs_embed_dict.pkl, or correct the dim_info "
            f"block of the config.") from exc
    model.eval()
    return model


def load_split(data_root: Path, split: str, split_type: str = "frequency"):
    """Load one preprocessed split through the training code's own loader."""
    _ensure_importable()
    from utils import load_data  # noqa: E402

    return load_data(data_path=str(Path(data_root)) + "/",
                     split_type=split_type, data_type=split)


def infer(model, data, device: torch.device, batch_size: int = 256):
    """Run the model over a split and return ``(probs, targets, conditions)``."""
    _ensure_importable()
    from utils import iterate_minibatches  # noqa: E402

    probs, targets, conditions = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in iterate_minibatches(
                category=data["category_data"], prs=data["prs_data"],
                conditions=data["conditions"], targets=data["targets"],
                batch_size=batch_size, shuffle=False, device=device):
            inputs, tg, times, obsn_mask, tokens = batch
            pred, _, _, _ = model(inputs, tokens, times, tg["condition"],
                                  obsn_mask, mode="eval")
            probs.append(torch.softmax(pred, dim=-1).cpu().numpy())
            targets.append(tg["CKD_targets"].cpu().numpy())
            conditions.append(tg["condition"].cpu().numpy())
    return (np.concatenate(probs), np.concatenate(targets),
            np.concatenate(conditions))
