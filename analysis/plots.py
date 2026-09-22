"""Calibration, precision-recall and decision curves.

One implementation, shared by every script that reports these panels, so the
CKD-progression and eGFR-decline figures are drawn identically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (average_precision_score,  # noqa: E402
                             precision_recall_curve)

#: Stable colours so a model keeps its colour across every figure.
COLORS = {
    "Transformer": "#d62728",
    "LightGBM": "#2ca02c",
    "XGBoost": "#ff7f0e",
    "LR": "#1f77b4",
    "SVM": "#9467bd",
}


def _color(label: str) -> str | None:
    return COLORS.get(label)


def net_benefit(y, p, thresholds):
    """Decision-curve net benefit for the model and for a treat-all strategy."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    prevalence = y.mean()
    odds = thresholds / (1 - thresholds)
    model = np.array([
        ((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * o
        for t, o in zip(thresholds, odds)])
    treat_all = prevalence - (1 - prevalence) * odds
    return model, treat_all


def _panels(horizons, figsize_per_panel=(4.6, 4.4)):
    width = figsize_per_panel[0] * len(horizons)
    fig, axes = plt.subplots(1, len(horizons),
                             figsize=(width, figsize_per_panel[1]))
    return fig, np.atleast_1d(axes)


def calibration_figure(predictions, labels, horizons, title, out_path: Path,
                       n_bins: int = 10):
    """Reliability diagram per horizon (quantile bins, rare-outcome friendly)."""
    fig, axes = _panels(horizons)
    for ax, horizon in zip(axes, horizons):
        for label in labels:
            item = predictions.get((label, horizon))
            if item is None:
                continue
            y, p = item
            try:
                frac_pos, mean_pred = calibration_curve(
                    y, p, n_bins=n_bins, strategy="quantile")
            except ValueError:
                continue
            ax.plot(mean_pred, frac_pos, marker="o", ms=4, lw=1.5,
                    label=label, color=_color(label))
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_title(f"{horizon} horizon")
        ax.set_xlabel("Mean predicted probability")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Observed event fraction")
    axes[-1].legend(fontsize=8)
    _finish(fig, title, out_path)


def pr_figure(predictions, labels, horizons, title, out_path: Path):
    """Precision-recall curves per horizon, with the prevalence baseline."""
    fig, axes = _panels(horizons)
    for ax, horizon in zip(axes, horizons):
        prevalence = None
        for label in labels:
            item = predictions.get((label, horizon))
            if item is None:
                continue
            y, p = item
            prevalence = float(np.mean(y))
            precision, recall, _ = precision_recall_curve(y, p)
            ax.plot(recall, precision, lw=1.5, color=_color(label),
                    label=f"{label} (AP={average_precision_score(y, p):.2f})")
        if prevalence is not None:
            ax.axhline(prevalence, color="gray", ls=":", lw=1)
            ax.set_title(f"{horizon} horizon (prev={prevalence:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Precision")
    _finish(fig, title, out_path)


def dca_figure(predictions, labels, horizons, title, out_path: Path,
               thresholds=None):
    """Decision-curve analysis per horizon against treat-all / treat-none."""
    thresholds = np.linspace(0.01, 0.5, 50) if thresholds is None else thresholds
    fig, axes = _panels(horizons)
    for ax, horizon in zip(axes, horizons):
        treat_all = None
        for label in labels:
            item = predictions.get((label, horizon))
            if item is None:
                continue
            y, p = item
            model_nb, treat_all = net_benefit(y, p, thresholds)
            ax.plot(thresholds, model_nb, lw=1.5, color=_color(label), label=label)
        if treat_all is not None:
            ax.plot(thresholds, treat_all, "k--", lw=1, label="Treat all")
            ax.set_ylim(-0.02, max(0.05, float(np.max(treat_all)) * 1.1))
        ax.axhline(0, color="gray", ls=":", lw=1, label="Treat none")
        ax.set_title(f"{horizon} horizon")
        ax.set_xlabel("Threshold probability")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Net benefit")
    _finish(fig, title, out_path)


def _finish(fig, title, out_path: Path):
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
