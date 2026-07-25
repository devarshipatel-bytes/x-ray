"""Export X-DETR to ONNX for lightweight local (4 GB) inference.

  python -m scripts.export_onnx --config configs/xdetr_opixray.yaml \
      --weights runs/opixray_xdetr/last.pth --out runs/opixray_xdetr/xdetr.onnx

Note: exports the raw model (logits + boxes). Post-processing (sigmoid, threshold,
coordinate mapping) stays in Python via models.postprocess.
"""
from __future__ import annotations

import argparse

import torch

from data import in_channels
from engine.checkpoint import load_checkpoint
from engine.config import load_config
from models import build_model


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["pred_logits"], out["pred_boxes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ch = in_channels(cfg)
    size = cfg["input"]["size"]
    model = build_model(cfg, ch)
    load_checkpoint(args.weights, model, map_location="cpu")
    model.eval()

    wrapper = ExportWrapper(model)
    dummy = torch.randn(1, ch, size, size)
    torch.onnx.export(
        wrapper, dummy, args.out, opset_version=args.opset,
        input_names=["image"], output_names=["pred_logits", "pred_boxes"],
        dynamic_axes={"image": {0: "batch"}, "pred_logits": {0: "batch"}, "pred_boxes": {0: "batch"}},
    )
    print(f"[export] wrote {args.out}  (input {ch}x{size}x{size})")


if __name__ == "__main__":
    main()
