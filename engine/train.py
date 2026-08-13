"""X-DETR training loop — single local GPU (AMP + grad accumulation + resume).

Every hyperparameter can be set three ways, in increasing priority:
  1. configs/xdetr_opixray.yaml          (the defaults)
  2. a named CLI flag                    (--epochs, --batch-size, --img-size, ...)
  3. --set dotted.key=value              (escape hatch for keys without a flag)

Checkpoints, all written to --output-dir:
  last.pth        overwritten every epoch  -> the resume point, auto-detected on restart
  epoch_XXX.pth   archival snapshot every --ckpt-interval epochs (default 10, 0 = off)
  final.pth       written once training completes

Examples
--------
# baseline (12 GB GPU)
python -m engine.train --config configs/xdetr_opixray.yaml

# explicit hyperparameters
python -m engine.train --config configs/xdetr_opixray.yaml \
    --epochs 80 --batch-size 8 --img-size 512 --lr 1e-4 --ckpt-interval 10

# fast smoke test (~1 min) before committing to a long run
python -m engine.train --config configs/xdetr_opixray.yaml \
    --epochs 1 --limit 20 --batch-size 2 --img-size 320 --log-interval 1

# resume: automatic if last.pth exists in --output-dir, or point at one explicitly
python -m engine.train --config configs/xdetr_opixray.yaml \
    --resume runs/opixray_xdetr/last.pth
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data import build_dataset, collate_fn, in_channels
from models import build_model, build_matcher, build_criterion
from .checkpoint import maybe_resume, save_checkpoint
from .config import get_device, load_config, set_dotted, set_seed

# Named CLI flags map onto dotted config keys. A flag left as None is never applied,
# so the YAML value survives untouched.
_FLAG_TO_KEY = {
    "data_root":      "dataset.root",
    "img_size":       "input.size",
    "material_proxy": "input.use_material_proxy",
    "backbone":       "model.backbone",
    "hidden_dim":     "model.hidden_dim",
    "enc_layers":     "model.enc_layers",
    "dec_layers":     "model.dec_layers",
    "num_queries":    "model.num_queries",
    "epochs":         "training.epochs",
    "batch_size":     "training.batch_size",
    "grad_accum":     "training.grad_accum",
    "lr":             "training.lr",
    "lr_backbone":    "training.lr_backbone",
    "weight_decay":   "training.weight_decay",
    "clip_grad":      "training.clip_grad_norm",
    "lr_drop":        "training.lr_drop_epoch",
    "workers":        "training.num_workers",
    "seed":           "training.seed",
    "amp":            "training.amp",
    "resume":         "training.resume",
    "ckpt_interval":  "training.ckpt_interval",
    "output_dir":     "training.output_dir",
}


# ----------------------------------------------------------------------- CLI ---
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m engine.train",
        description="Train X-DETR on one local GPU. Flags override the YAML config.")
    ap.add_argument("--config", required=True, help="path to the YAML config")

    g = ap.add_argument_group("data")
    g.add_argument("--data-root", help="dataset root containing train/ and test/")
    g.add_argument("--img-size", type=int, help="square input size after letterboxing (e.g. 512)")
    g.add_argument("--material-proxy", action=argparse.BooleanOptionalAction, default=None,
                   help="append the 2 material-proxy channels (5ch input); --no-material-proxy ablates it")
    g.add_argument("--workers", type=int, help="DataLoader worker processes")
    g.add_argument("--limit", type=int, help="train on only the first N images (smoke tests)")

    g = ap.add_argument_group("model")
    g.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50"])
    g.add_argument("--hidden-dim", type=int, help="transformer width (default 256)")
    g.add_argument("--enc-layers", type=int, help="AIFI encoder layers")
    g.add_argument("--dec-layers", type=int, help="decoder layers (fewer = faster/less VRAM)")
    g.add_argument("--num-queries", type=int, help="object queries (fewer = faster/less VRAM)")

    g = ap.add_argument_group("optimization")
    g.add_argument("--epochs", type=int)
    g.add_argument("--batch-size", type=int, help="images per forward pass (VRAM-bound)")
    g.add_argument("--grad-accum", type=int,
                   help="effective batch = batch-size x grad-accum (costs no extra VRAM)")
    g.add_argument("--lr", type=float, help="base LR for the transformer/heads")
    g.add_argument("--lr-backbone", type=float, help="lower LR for the pretrained backbone")
    g.add_argument("--weight-decay", type=float)
    g.add_argument("--clip-grad", type=float, help="max grad norm (0 disables clipping)")
    g.add_argument("--lr-drop", type=int, help="epoch at which LR is multiplied by 0.1")
    g.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="fp16 mixed precision (CUDA only); --no-amp forces fp32")
    g.add_argument("--seed", type=int)

    g = ap.add_argument_group("checkpointing and logging")
    g.add_argument("--output-dir", help="where checkpoints and logs are written")
    g.add_argument("--ckpt-interval", type=int,
                   help="write epoch_XXX.pth every N epochs (0 = only last.pth/final.pth)")
    g.add_argument("--resume", help="checkpoint path to resume from")
    g.add_argument("--no-auto-resume", action="store_true",
                   help="start fresh even if output-dir/last.pth exists")
    g.add_argument("--log-interval", type=int, default=25,
                   help="iterations between progress lines (default 25)")
    g.add_argument("--set", nargs="*", default=[], metavar="k.k=v",
                   help="extra dotted config overrides")
    return ap


def apply_cli_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Fold the named flags into cfg. Applied after --set so flags win."""
    for flag, key in _FLAG_TO_KEY.items():
        val = getattr(args, flag, None)
        if val is not None:
            set_dotted(cfg, key, val)
    cfg["training"].setdefault("ckpt_interval", 10)
    cfg["training"].setdefault("grad_accum", 1)
    cfg["training"].setdefault("resume", "")
    return cfg


# ------------------------------------------------------------------ helpers ---
def build_optimizer(model, cfg: Dict):
    """Two param groups: the pretrained backbone gets a 10x smaller LR (DETR convention)."""
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


def build_scaler(enabled: bool):
    """GradScaler across torch versions (torch.cuda.amp.GradScaler is deprecated in 2.4+)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def move_targets(targets, device):
    return [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
             for k, v in t.items()} for t in targets]


def fmt_hms(seconds: float) -> str:
    s = int(max(seconds, 0))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"


def gpu_mem_str(device) -> str:
    if device.type != "cuda":
        return "n/a"
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    return f"{peak:.1f}/{total:.0f}G"


def print_header(cfg: Dict, args, model, device, n_train: int, n_iters: int, use_amp: bool):
    tr, mo = cfg["training"], cfg["model"]
    ch = in_channels(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        dev = f"{props.name}  ({props.total_memory / 2 ** 30:.0f} GB VRAM)"
    else:
        dev = "CPU  (no CUDA device found — training will be very slow)"
    interval = int(tr["ckpt_interval"])
    rows = [
        ("device", f"{dev}  |  torch {torch.__version__}"),
        ("dataset", f"{cfg['dataset']['root']}  |  {n_train} train images  |  "
                    f"{cfg['dataset']['num_classes']} classes"),
        ("input", f"{cfg['input']['size']}x{cfg['input']['size']}  {ch}ch "
                  f"({'RGB + material proxy' if ch == 5 else 'RGB only'})"),
        ("model", f"{mo['backbone']}  dim={mo['hidden_dim']}  enc={mo['enc_layers']}  "
                  f"dec={mo['dec_layers']}  queries={mo['num_queries']}  "
                  f"aux_loss={mo.get('aux_loss', True)}"),
        ("params", f"{total / 1e6:.1f}M total  |  {trainable / 1e6:.1f}M trainable"),
        ("batch", f"{tr['batch_size']} x {tr['grad_accum']} accum = "
                  f"{tr['batch_size'] * tr['grad_accum']} effective  |  {n_iters} it/epoch"),
        ("precision", "fp16 AMP" if use_amp else "fp32"),
        ("optimizer", f"AdamW  lr={tr['lr']:.2e}  backbone_lr={tr['lr_backbone']:.2e}  "
                      f"wd={tr['weight_decay']:.2e}  clip={tr['clip_grad_norm']}"),
        ("schedule", f"{tr['epochs']} epochs  |  LR x0.1 at epoch {tr['lr_drop_epoch']}"),
        ("output", tr["output_dir"]),
        ("checkpoints", "last.pth every epoch" +
                        (f"  |  epoch_XXX.pth every {interval} epochs" if interval > 0 else "")),
    ]
    width = max(len(k) for k, _ in rows)
    bar = "=" * 96
    print(bar)
    print("X-DETR — occlusion-robust prohibited-item detection")
    print(bar)
    for k, v in rows:
        print(f"  {k:>{width}} : {v}")
    print(bar, flush=True)


# ------------------------------------------------------------------- epoch ----
def train_one_epoch(model, criterion, loader, optimizer, scaler, device, cfg,
                    epoch: int, log_interval: int, use_amp: bool,
                    elapsed_before: float) -> Dict:
    model.train()
    tr = cfg["training"]
    accum = max(1, int(tr["grad_accum"]))
    clip = float(tr.get("clip_grad_norm", 0.0))
    total_epochs = int(tr["epochs"])
    n_iters = len(loader)

    running: Dict[str, float] = {}
    n_logged = 0
    skipped = 0
    t_epoch = time.time()
    t_window, imgs_window = t_epoch, 0
    optimizer.zero_grad(set_to_none=True)

    for it, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = move_targets(targets, device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(imgs)
            losses = criterion(out, targets)

        loss_val = float(losses["loss"].detach())
        if not math.isfinite(loss_val):
            # Do not backward a non-finite loss — it would poison every weight.
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(losses["loss"] / accum).backward()
        # Step on accumulation boundaries, and always on the last iteration so a
        # partial tail batch still contributes instead of being silently dropped.
        if (it + 1) % accum == 0 or (it + 1) == n_iters:
            if clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Detach before any float() — the raw tensors still carry grad_fn here.
        scalars = {k: float(v.detach()) for k, v in losses.items() if "_aux" not in k}
        for k, v in scalars.items():
            running[k] = running.get(k, 0.0) + v
        n_logged += 1
        imgs_window += imgs.shape[0]

        if it % log_interval == 0 or (it + 1) == n_iters:
            now = time.time()
            ips = imgs_window / max(now - t_window, 1e-6)
            t_window, imgs_window = now, 0
            done = it + 1
            eta_epoch = (now - t_epoch) / done * (n_iters - done)
            print(f"  ep {epoch + 1:>3}/{total_epochs}  it {done:>4}/{n_iters} "
                  f"{100 * done / n_iters:>3.0f}%  "
                  f"loss {loss_val:7.3f}  "
                  f"ce {scalars['loss_ce']:6.3f} "
                  f"bbox {scalars['loss_bbox']:6.3f} "
                  f"giou {scalars['loss_giou']:6.3f}  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                  f"{ips:5.1f} img/s  mem {gpu_mem_str(device)}  "
                  f"eta {fmt_hms(eta_epoch)}", flush=True)

    epoch_time = time.time() - t_epoch
    stats = {k: v / max(n_logged, 1) for k, v in running.items()}
    stats["epoch"] = epoch
    stats["lr"] = optimizer.param_groups[0]["lr"]
    stats["epoch_time_s"] = round(epoch_time, 1)
    stats["images_per_s"] = round(n_logged * int(tr["batch_size"]) / max(epoch_time, 1e-6), 2)
    stats["skipped_nonfinite"] = skipped
    stats["peak_mem_gb"] = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                            if device.type == "cuda" else None)

    remaining = (total_epochs - epoch - 1) * epoch_time
    msg = (f"[epoch {epoch + 1}/{total_epochs}] "
           f"loss {stats.get('loss', 0):.3f}  "
           f"ce {stats.get('loss_ce', 0):.3f}  "
           f"bbox {stats.get('loss_bbox', 0):.3f}  "
           f"giou {stats.get('loss_giou', 0):.3f}  |  "
           f"{fmt_hms(epoch_time)}/epoch  "
           f"elapsed {fmt_hms(elapsed_before + epoch_time)}  "
           f"eta {fmt_hms(remaining)}")
    if skipped:
        msg += f"  |  WARNING: skipped {skipped} non-finite iters"
    print(msg, flush=True)
    return stats


# -------------------------------------------------------------------- main ----
def main() -> int:
    args = build_argparser().parse_args()
    cfg = apply_cli_overrides(load_config(args.config, args.set), args)
    tr = cfg["training"]

    set_seed(int(tr["seed"]))
    device = get_device()
    out_dir = tr["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    use_amp = bool(tr.get("amp", True)) and device.type == "cuda"
    if bool(tr.get("amp", True)) and device.type != "cuda":
        print("[train] AMP requested but no CUDA device — falling back to fp32.")

    # data
    ch = in_channels(cfg)
    train_ds = build_dataset(cfg, split="train", train=True)
    if args.limit:
        n = min(args.limit, len(train_ds))
        train_ds = Subset(train_ds, range(n))
        print(f"[train] --limit {args.limit}: using {n} images (smoke-test mode)")
    workers = int(tr["num_workers"])
    loader = DataLoader(
        train_ds, batch_size=int(tr["batch_size"]), shuffle=True, num_workers=workers,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"), drop_last=True,
        persistent_workers=(workers > 0),
    )
    if len(loader) == 0:
        raise RuntimeError(
            f"0 iterations per epoch: {len(train_ds)} images < batch_size {tr['batch_size']} "
            f"with drop_last=True. Lower --batch-size or raise --limit.")

    # model / loss / optim
    model = build_model(cfg, ch).to(device)
    criterion = build_criterion(cfg, build_matcher(cfg)).to(device)
    optimizer = build_optimizer(model, cfg)
    scaler = build_scaler(use_amp)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(tr["lr_drop_epoch"]), gamma=0.1)

    print_header(cfg, args, model, device, len(train_ds), len(loader), use_amp)

    # resume
    start_epoch, best = 0, -1.0
    ckpt = maybe_resume(cfg, model, optimizer, scaler, scheduler,
                        auto=not args.no_auto_resume)
    if ckpt:
        start_epoch = int(ckpt["epoch"]) + 1
        best = float(ckpt.get("best_metric", -1.0))
        print(f"[train] resumed at epoch {start_epoch}/{tr['epochs']}", flush=True)
    if start_epoch >= int(tr["epochs"]):
        print(f"[train] nothing to do: checkpoint is already at epoch {start_epoch} "
              f"of {tr['epochs']}. Raise --epochs to continue training.")
        return 0

    # snapshot the fully-resolved config next to the checkpoints
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    log_path = os.path.join(out_dir, "train_log.jsonl")
    interval = int(tr["ckpt_interval"])
    t_start = time.time()
    for epoch in range(start_epoch, int(tr["epochs"])):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        stats = train_one_epoch(model, criterion, loader, optimizer, scaler, device, cfg,
                                epoch, args.log_interval, use_amp, time.time() - t_start)
        scheduler.step()

        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")

        # last.pth every epoch: cheap (it is overwritten) and makes any crash resumable.
        save_checkpoint(os.path.join(out_dir, "last.pth"), model, optimizer, scaler,
                        scheduler, epoch, best, cfg)
        if interval > 0 and (epoch + 1) % interval == 0:
            tag = os.path.join(out_dir, f"epoch_{epoch + 1:03d}.pth")
            save_checkpoint(tag, model, optimizer, scaler, scheduler, epoch, best, cfg)
            print(f"[ckpt] {tag}", flush=True)

    save_checkpoint(os.path.join(out_dir, "final.pth"), model, optimizer, scaler,
                    scheduler, int(tr["epochs"]) - 1, best, cfg)
    print(f"\n[train] done in {fmt_hms(time.time() - t_start)}. "
          f"Checkpoints in {out_dir} (final.pth, last.pth). Log: {log_path}")
    print(f"[train] next: python -m engine.evaluate --config {args.config} "
          f"--weights {os.path.join(out_dir, 'final.pth')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
