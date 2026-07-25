"""Dataset registry. Swap datasets by config (dataset.name)."""
from __future__ import annotations

from typing import Dict, Optional

from .transforms import build_train_transform, build_eval_transform, collate_fn, in_channels

_REGISTRY = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


@register("opixray")
def _build_opixray(cfg, split, transform, occlusion_level=None):
    from .opixray import build_dataset
    return build_dataset(cfg, split, transform, occlusion_level)


# --- M6: PIDray / HiXray via the shared COCO-json loader (swap dataset.root + classes) ---
@register("pidray")
@register("hixray")
def _build_coco_style(cfg, split, transform, occlusion_level=None):
    from .coco_style import build_dataset
    return build_dataset(cfg, split, transform, occlusion_level)

# SIXray uses its own CSV-style annotations + heavy negative sampling; add data/sixray.py
# and register it here when needed (same (image_tensor, target) contract).


def build_dataset(cfg: Dict, split: str, train: bool, occlusion_level: Optional[str] = None):
    name = cfg["dataset"]["name"]
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {list(_REGISTRY)}")
    transform = build_train_transform(cfg) if train else build_eval_transform(cfg)
    return _REGISTRY[name](cfg, split, transform, occlusion_level)


__all__ = ["build_dataset", "collate_fn", "in_channels"]
