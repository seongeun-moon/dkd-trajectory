"""Scoring a trained transformer on a held-out split.

The training loop itself lives in ``het_trans/main_graph_snubh.py``; this module
only reads a trained model and turns its predictions into the per-horizon table
the analysis scripts report.
"""

import numpy as np
import torch
import torch.nn as nn

import transformer
from metrics import per_horizon


def evaluate(model, data, device, loss_fn, batch_size: int):
    """Return ``(probs, targets, conditions, mean_loss)`` for one split."""
    transformer._ensure_importable()
    from utils import iterate_minibatches  # noqa: E402

    model.eval()
    probs, targets, conditions, losses = [], [], [], []
    with torch.no_grad():
        for batch in iterate_minibatches(
                category=data["category_data"], prs=data["prs_data"],
                conditions=data["conditions"], targets=data["targets"],
                batch_size=batch_size, shuffle=False, device=device):
            inputs, tg, times, obsn_mask, tokens = batch
            pred, _, _, _ = model(inputs, tokens, times, tg["condition"],
                                  obsn_mask, mode="eval")
            losses.append(loss_fn(pred, tg["CKD_targets"]).item())
            probs.append(torch.softmax(pred, dim=-1).cpu().numpy())
            targets.append(tg["CKD_targets"].cpu().numpy())
            conditions.append(tg["condition"].cpu().numpy())
    return (np.concatenate(probs), np.concatenate(targets),
            np.concatenate(conditions), float(np.mean(losses)))


def test_report(model, test_data, device, args, n_boot: int = 0):
    """Per-horizon metrics on the held-out split, plus the raw predictions."""
    loss_fn = nn.CrossEntropyLoss()
    probs, targets, conditions, _ = evaluate(
        model, test_data, device, loss_fn, args.batch_size)
    rows = per_horizon(probs, targets, conditions, n_boot=n_boot)
    return rows, probs, targets, conditions
