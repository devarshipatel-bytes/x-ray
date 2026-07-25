"""YAML config loading with dotted CLI overrides and reproducibility helpers."""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import yaml


def load_config(path: str, overrides: List[str] | None = None) -> Dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set_dotted(cfg, key.strip(), _coerce(val.strip()))
    return cfg


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def _set_dotted(cfg: Dict, dotted: str, val):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
