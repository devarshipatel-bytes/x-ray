"""Operator attention-reallocation map.

Innovation (d) from the plan: instead of box-and-alert, produce a single confidence-weighted
"look here, in this order" saliency map + a ranked priority list of regions. Designed to
redirect a fatigued operator's attention to rare targets without adding boxes to hunt through.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .attention import decode_detections, cross_attention_map, encoder_saliency


def operator_attention_map(caches: Dict, size: int, score_thresh: float = 0.25,
                           enc_weight: float = 0.3) -> Tuple[np.ndarray, List[Dict]]:
    dets = decode_detections(caches, score_thresh=score_thresh)
    acc = np.zeros((size, size), np.float32)
    for d in dets:
        acc += d["score"] * cross_attention_map(caches, d["query"], size)
    if acc.max() > 0:
        acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)
    enc = encoder_saliency(caches, size)
    combined = acc + enc_weight * enc
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

    # rank detections by score -> priority order for the operator
    dets = sorted(dets, key=lambda d: -d["score"])
    for rank, d in enumerate(dets, start=1):
        cx, cy, w, h = d["box"]
        d["priority"] = rank
        d["center_px"] = (float(cx * size), float(cy * size))
    return combined, dets
