"""M1 correctness gate: overfit a tiny subset to near-zero loss.

Overfits the first --n real OPIXray images; boxes should visually snap and the loss
should collapse. Validates model/matcher/loss wiring end-to-end on real data.

  python -m scripts.overfit --config configs/xdetr_opixray.yaml --n 10 --iters 300

  # 4 GB GPU: shrink the model AND use --amp (fp16) — ResNet-50 backbone activations at
  # 512px in fp32 fill a 4 GB card even at batch 2/dec_layers=3.
  python -m scripts.overfit --config configs/xdetr_opixray.yaml --n 10 --iters 300 --amp \
      --set model.dec_layers=3 model.num_queries=100 training.batch_size=2 input.size=384
"""
from __future__ import annotations

import argparse

import torch

from data import build_dataset, collate_fn, in_channels
from engine.config import get_device, load_config, set_seed
from models import build_model, build_matcher, build_criterion


def real_batch(cfg, n, device):
    ds = build_dataset(cfg, split="train", train=False)  # eval transform = no aug (stable overfit)
    n = min(n, len(ds))
    batch = [ds[i] for i in range(n)]
    imgs, targets = collate_fn(batch)
    imgs = imgs.to(device)
    targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
    return imgs, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true", help="fp16 mixed precision (for small-VRAM GPUs)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(0)
    device = get_device()
    ch = in_channels(cfg)

    model = build_model(cfg, ch).to(device)
    criterion = build_criterion(cfg, build_matcher(cfg)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[overfit] X-DETR params: {n_params:.1f}M  device={device}")

    imgs, targets = real_batch(cfg, args.n, device)
    print(f"[overfit] batch: {imgs.shape[0]} images, {sum(len(t['labels']) for t in targets)} boxes")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[overfit] AMP (fp16) enabled")
    model.train()
    first, last = None, None
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(imgs)
            losses = criterion(out, targets)
            loss = losses["loss"]
        scaler.scale(loss).backward()
        if use_amp:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(opt)
        scaler.update()
        if it == 0:
            first = float(loss.detach())
        last = float(loss.detach())
        if it % max(1, args.iters // 20) == 0 or it == args.iters - 1:
            main_only = {k: round(float(v), 3) for k, v in losses.items() if "_aux" not in k}
            print(f"  it {it:4d}  {main_only}")

    print(f"\n[overfit] loss {first:.3f} -> {last:.3f}")
    ok = last < first * 0.25
    print("[overfit] PASS ✔  (loss collapsed — wiring correct)" if ok else
          "[overfit] WARN ✘  (loss did not collapse enough — increase --iters or inspect)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
