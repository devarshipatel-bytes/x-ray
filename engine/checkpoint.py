"""Checkpoint save / load / auto-resume.

A checkpoint carries model + optimizer + scaler + scheduler + epoch + the config it was
trained with, so resuming restores the exact optimizer and LR-schedule state rather than
restarting them. `engine.train` writes:

  last.pth        every epoch (overwritten) — what auto-resume picks up
  epoch_XXX.pth   every training.ckpt_interval epochs — archival, never overwritten
  final.pth       on completion
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


def maybe_resume(cfg: Dict, model, optimizer, scaler, scheduler,
                 auto: bool = True) -> Optional[Dict]:
    """Resume from an explicit cfg.training.resume path, else from output_dir/last.pth.

    An explicit path that does not exist is an error, not a silent fresh start — that
    would quietly discard a run you meant to continue. Pass auto=False to ignore
    last.pth and start over.
    """
    out_dir = cfg["training"]["output_dir"]
    resume = cfg["training"].get("resume", "")
    if resume:
        if not os.path.isfile(resume):
            raise FileNotFoundError(f"--resume path does not exist: {resume}")
        path = resume
    else:
        candidate = os.path.join(out_dir, "last.pth")
        if not auto or not os.path.isfile(candidate):
            return None
        path = candidate
    print(f"[checkpoint] resuming from {path}")
    return load_checkpoint(path, model, optimizer, scaler, scheduler)
