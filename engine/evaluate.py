"""Evaluation: per-class AP@0.5, occlusion-stratified (OL1/2/3), and calibration (ECE).

Self-contained (no pycocotools). AP uses VOC all-point interpolation. ECE is computed at
the detection level: confidence vs. probability-a-detection-is-a-true-positive.

Usage:
  python -m engine.evaluate --config configs/xdetr_opixray.yaml --weights runs/opixray_xdetr/last.pth
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import build_dataset, collate_fn, in_channels
from models import build_model, postprocess
from models.box_ops import box_cxcywh_to_xyxy, box_iou
from .checkpoint import load_checkpoint
from .config import get_device, load_config


def _gt_to_original(boxes_norm_cxcywh: torch.Tensor, t: Dict) -> torch.Tensor:
    """Ground-truth normalized cxcywh -> original-image xyxy pixels."""
    if boxes_norm_cxcywh.numel() == 0:
        return boxes_norm_cxcywh.new_zeros((0, 4))
    bx = box_cxcywh_to_xyxy(boxes_norm_cxcywh).clone()
    size = int(t["size"][0]); ratio = float(t["ratio"])
    pad_x, pad_y = int(t["pad"][0]), int(t["pad"][1])
    oh, ow = int(t["orig_size"][0]), int(t["orig_size"][1])
    bx = bx * size
    bx[:, [0, 2]] = (bx[:, [0, 2]] - pad_x) / ratio
    bx[:, [1, 3]] = (bx[:, [1, 3]] - pad_y) / ratio
    bx[:, [0, 2]] = bx[:, [0, 2]].clamp(0, ow)
    bx[:, [1, 3]] = bx[:, [1, 3]].clamp(0, oh)
    return bx


@torch.no_grad()
def run_inference(model, cfg, device, split="test", occlusion_level=None, score_thresh=0.05):
    ds = build_dataset(cfg, split=split, train=False, occlusion_level=occlusion_level)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        num_workers=cfg["training"]["num_workers"], collate_fn=collate_fn)
    preds, gts = {}, {}
    model.eval()
    for imgs, targets in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        res = postprocess(out, targets, score_thresh=score_thresh, max_dets=cfg["eval"]["max_dets"])
        for r, t in zip(res, targets):
            iid = t["image_id"]
            preds[iid] = {"boxes": r["boxes"].numpy(), "scores": r["scores"].numpy(),
                          "labels": r["labels"].numpy()}
            gt_boxes = _gt_to_original(t["boxes"], t).numpy()
            gts[iid] = {"boxes": gt_boxes, "labels": t["labels"].numpy()}
    return preds, gts


def _voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap(preds, gts, num_classes, iou_thresh=0.5):
    """Returns (per_class_ap list, mAP, calibration_records list of (score, is_tp))."""
    per_class = []
    calib = []
    for c in range(num_classes):
        dets = []  # (score, image_id, box)
        npos = 0
        gt_matched = {}
        for iid, g in gts.items():
            mask = g["labels"] == c
            gt_matched[iid] = np.zeros(int(mask.sum()), dtype=bool)
            npos += int(mask.sum())
        for iid, p in preds.items():
            mask = p["labels"] == c
            for s, b in zip(p["scores"][mask], p["boxes"][mask]):
                dets.append((float(s), iid, b))
        dets.sort(key=lambda x: -x[0])
        tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
        for i, (s, iid, box) in enumerate(dets):
            g = gts[iid]
            gmask = g["labels"] == c
            gboxes = g["boxes"][gmask]
            is_tp = False
            if len(gboxes):
                ious, _ = box_iou(torch.tensor(box).float().view(1, 4), torch.tensor(gboxes).float())
                ious = ious.numpy()[0]
                j = int(ious.argmax())
                if ious[j] >= iou_thresh and not gt_matched[iid][j]:
                    tp[i] = 1; gt_matched[iid][j] = True; is_tp = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
            calib.append((s, is_tp))
        if npos == 0:
            per_class.append(float("nan")); continue
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        rec = tp_cum / npos
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        per_class.append(_voc_ap(rec, prec))
    valid = [a for a in per_class if not np.isnan(a)]
    mAP = float(np.mean(valid)) if valid else 0.0
    return per_class, mAP, calib


def expected_calibration_error(calib, n_bins=15):
    if not calib:
        return 0.0
    scores = np.array([c[0] for c in calib])
    correct = np.array([1.0 if c[1] else 0.0 for c in calib])
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (scores > bins[i]) & (scores <= bins[i + 1])
        if m.sum() == 0:
            continue
        conf = scores[m].mean(); acc = correct[m].mean()
        ece += (m.sum() / len(scores)) * abs(acc - conf)
    return float(ece)


def evaluate_split(model, cfg, device, split="test", occlusion_level=None):
    preds, gts = run_inference(model, cfg, device, split, occlusion_level,
                               score_thresh=cfg["eval"]["score_thresh"])
    num_classes = cfg["dataset"]["num_classes"]
    per_class, mAP, calib = compute_ap(preds, gts, num_classes, cfg["eval"]["iou_thresh"])
    ece = expected_calibration_error(calib)
    return {"per_class_ap": per_class, "mAP": mAP, "ECE": ece, "n_images": len(preds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    device = get_device()
    model = build_model(cfg, in_channels(cfg)).to(device)
    load_checkpoint(args.weights, model, map_location=device)
    classes = cfg["dataset"]["classes"]

    print("\n=== Overall (test) ===")
    overall = evaluate_split(model, cfg, device, "test", None)
    for c, apv in zip(classes, overall["per_class_ap"]):
        print(f"  AP@0.5  {c:20s} {apv:.3f}")
    print(f"  mAP@0.5 = {overall['mAP']:.3f}   ECE = {overall['ECE']:.3f}   images={overall['n_images']}")

    results = {"overall": overall, "occlusion": {}}
    for lvl in cfg["dataset"].get("occlusion_levels", []):
        try:
            r = evaluate_split(model, cfg, device, "test", lvl)
            results["occlusion"][lvl] = r
            print(f"=== {lvl} ===  mAP@0.5 = {r['mAP']:.3f}  (images={r['n_images']})")
        except FileNotFoundError as e:
            print(f"[eval] skip {lvl}: {e}")

    out_dir = cfg["training"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] wrote {os.path.join(out_dir, 'eval_results.json')}")


if __name__ == "__main__":
    main()
