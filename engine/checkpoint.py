"""Checkpointing with Colab-Free resume.

Writes `last.pth` every epoch and `best.pth` on metric improvement. On Colab set
`training.output_dir` to a Google Drive path so a disconnect/restart resumes seamlessly.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch


def save_checkpoint(path: str, model, optimizer, scaler, scheduler, epoch: int,
                    best_metric: float, cfg: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "cfg": cfg,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scaler=None, scheduler=None,
                    map_location="cpu") -> Dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def maybe_resume(cfg: Dict, model, optimizer, scaler, scheduler) -> Optional[Dict]:
    """Resume from explicit cfg.training.resume, else auto-resume from output_dir/last.pth."""
    out_dir = cfg["training"]["output_dir"]
    resume = cfg["training"].get("resume", "")
    auto = os.path.join(out_dir, "last.pth")
    path = resume or (auto if os.path.isfile(auto) else "")
    if not path or not os.path.isfile(path):
        return None
    print(f"[checkpoint] resuming from {path}")
    return load_checkpoint(path, model, optimizer, scaler, scheduler)
