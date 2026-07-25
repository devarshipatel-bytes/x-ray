"""X-DETR training loop (AMP + grad-accum + Colab-Free resume).

Usage:
  python -m engine.train --config configs/xdetr_opixray.yaml
  python -m engine.train --config configs/xdetr_opixray.yaml --set training.epochs=30 model.dec_layers=3
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from data import build_dataset, collate_fn, in_channels
from models import build_model, build_matcher, build_criterion
from .checkpoint import maybe_resume, save_checkpoint
from .config import get_device, load_config, set_seed


def build_optimizer(model, cfg):
    tr = cfg["training"]
    backbone_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if n.startswith("backbone.") else other_params).append(p)
    groups = [
        {"params": other_params, "lr": tr["lr"]},
        {"params": backbone_params, "lr": tr["lr_backbone"]},
    ]
    return torch.optim.AdamW(groups, lr=tr["lr"], weight_decay=tr["weight_decay"])


def move_targets(targets, device):
    out = []
    for t in targets:
        d = {}
        for k, v in t.items():
            d[k] = v.to(device) if torch.is_tensor(v) else v
        out.append(d)
    return out


def train_one_epoch(model, criterion, loader, optimizer, scaler, device, cfg, epoch):
    model.train()
    tr = cfg["training"]
    accum = max(1, tr.get("grad_accum", 1))
    clip = tr.get("clip_grad_norm", 0.0)
    use_amp = tr.get("amp", True) and device.type == "cuda"

    running = {}
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    for it, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = move_targets(targets, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(imgs)
            losses = criterion(out, targets)
            loss = losses["loss"] / accum

        scaler.scale(loss).backward()
        if (it + 1) % accum == 0:
            if clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for k, v in losses.items():
            if "_aux" in k:
                continue
            running[k] = running.get(k, 0.0) + float(v.detach())
        if it % 50 == 0:
            msg = "  ".join(f"{k}={float(v.detach()):.3f}" for k, v in losses.items() if "_aux" not in k)
            print(f"  epoch {epoch} it {it}/{len(loader)}  {msg}", flush=True)

    n = len(loader)
    avg = {k: v / n for k, v in running.items()}
    avg["epoch_time_s"] = round(time.time() - t0, 1)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides e.g. training.epochs=30")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg["training"]["seed"])
    device = get_device()
    out_dir = cfg["training"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] device={device}  out_dir={out_dir}")

    ch = in_channels(cfg)
    train_ds = build_dataset(cfg, split="train", train=True)
    print(f"[train] {len(train_ds)} training images, {ch}-channel input")
    loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                        num_workers=cfg["training"]["num_workers"], collate_fn=collate_fn,
                        pin_memory=(device.type == "cuda"), drop_last=True)

    model = build_model(cfg, ch).to(device)
    criterion = build_criterion(cfg, build_matcher(cfg)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] X-DETR params: {n_params:.1f}M")

    optimizer = build_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("amp", True) and device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["training"]["lr_drop_epoch"], gamma=0.1)

    start_epoch, best = 0, -1.0
    ckpt = maybe_resume(cfg, model, optimizer, scaler, scheduler)
    if ckpt:
        start_epoch = ckpt["epoch"] + 1
        best = ckpt.get("best_metric", -1.0)

    log_path = os.path.join(out_dir, "train_log.jsonl")
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        stats = train_one_epoch(model, criterion, loader, optimizer, scaler, device, cfg, epoch)
        scheduler.step()
        stats["epoch"] = epoch
        print(f"[train] epoch {epoch} done: {stats}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        # checkpoint EVERY epoch (Colab-Free safety)
        save_checkpoint(os.path.join(out_dir, "last.pth"), model, optimizer, scaler,
                        scheduler, epoch, best, cfg)
    print("[train] finished.")


if __name__ == "__main__":
    main()
