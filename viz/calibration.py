"""Metric plots: reliability diagram (ECE), per-class AP bars, occlusion-stratified AP."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .palette import class_color


def reliability_diagram(calib: List[Tuple[float, bool]], path: str, n_bins: int = 15):
    scores = np.array([c[0] for c in calib]) if calib else np.array([0.0])
    correct = np.array([1.0 if c[1] else 0.0 for c in calib]) if calib else np.array([0.0])
    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs, weights = [], [], []
    for i in range(n_bins):
        m = (scores > bins[i]) & (scores <= bins[i + 1])
        if m.sum() == 0:
            accs.append(0); confs.append((bins[i] + bins[i + 1]) / 2); weights.append(0); continue
        accs.append(correct[m].mean()); confs.append(scores[m].mean()); weights.append(m.sum())
    ece = sum(w / max(len(scores), 1) * abs(a - c) for a, c, w in zip(accs, confs, weights))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.bar(bins[:-1], accs, width=1 / n_bins, align="edge", alpha=0.8,
           color="#0072B2", edgecolor="white", label="model")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy (TP rate)")
    ax.set_title(f"Reliability  ECE={ece:.3f}")
    ax.legend(loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return ece


def per_class_ap_bar(classes: List[str], per_class_ap: List[float], path: str):
    fig, ax = plt.subplots(figsize=(max(5, len(classes) * 0.9), 4))
    vals = [0 if (v != v) else v for v in per_class_ap]  # nan->0
    colors = [tuple(v / 255 for v in class_color(i)) for i in range(len(classes))]
    ax.bar(range(len(classes)), vals, color=colors)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("AP@0.5"); ax.set_ylim(0, 1); ax.set_title("Per-class AP@0.5")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def occlusion_bar(levels: List[str], maps: Dict[str, float], path: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    xs = list(maps.keys()); ys = [maps[k] for k in xs]
    ax.plot(xs, ys, "-o", color="#D55E00", linewidth=2)
    ax.set_ylabel("mAP@0.5"); ax.set_ylim(0, 1)
    ax.set_title("Occlusion-stratified mAP (expect decline)")
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
