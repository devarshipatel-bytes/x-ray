"""Evaluation: precision/recall/F1, PR curves, per-class AP@0.5, occlusion strata, ECE.

Self-contained (no pycocotools). AP uses VOC all-point interpolation. Matching is greedy in
descending score order and each ground-truth box can be claimed once, so a second detection
on the same object counts as a false positive (there is no NMS — X-DETR is a set predictor).

Two score thresholds do different jobs:
  eval.score_thresh     — low (0.05). Keeps the tail of the PR curve so AP is not truncated.
  eval.operating_thresh — realistic (0.30). The point at which P/R/F1 are reported, i.e. what
                          an operator would actually see in the UI.

Usage:
  python -m engine.evaluate --config configs/xdetr_opixray.yaml --weights runs/opixray_xdetr/final.pth
  python -m engine.evaluate --config configs/xdetr_opixray.yaml --weights ... --operating-thresh 0.5
  python -m engine.evaluate --config configs/xdetr_opixray.yaml --weights ... --no-plots
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

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


def _voc_ap(rec, prec) -> float:
    """VOC all-point interpolated AP (area under the monotonised PR curve)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# --------------------------------------------------------------- PR machinery ---
def compute_pr(preds, gts, num_classes, iou_thresh=0.5):
    """Per-class PR curve + AP, plus pooled (score, is_tp) records for calibration.

    Returns (per_class, calib) where each per_class entry holds the full curve:
      ap, n_gt, n_det, scores (descending), tp_cum, fp_cum, recall, precision
    """
    per_class: List[Dict] = []
    calib: List[tuple] = []
    for c in range(num_classes):
        dets = []           # (score, image_id, box)
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
        scores = np.array([d[0] for d in dets], dtype=np.float64)
        for i, (s, iid, box) in enumerate(dets):
            g = gts[iid]
            gboxes = g["boxes"][g["labels"] == c]
            is_tp = False
            if len(gboxes):
                ious, _ = box_iou(torch.tensor(box).float().view(1, 4),
                                  torch.tensor(gboxes).float())
                ious = ious.numpy()[0]
                j = int(ious.argmax())
                if ious[j] >= iou_thresh and not gt_matched[iid][j]:
                    tp[i] = 1; gt_matched[iid][j] = True; is_tp = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
            calib.append((s, is_tp))

        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        if npos > 0:
            recall = tp_cum / npos
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap = _voc_ap(recall, precision)
        else:
            # No ground truth for this class: AP undefined, curve meaningless.
            recall = np.zeros(len(dets)); precision = np.zeros(len(dets))
            ap = float("nan")
        per_class.append({"ap": ap, "n_gt": npos, "n_det": len(dets), "scores": scores,
                          "tp_cum": tp_cum, "fp_cum": fp_cum,
                          "recall": recall, "precision": precision})
    return per_class, calib


def operating_point(cs: Dict, thresh: float) -> Dict:
    """Precision/recall/F1 for one class keeping only detections with score >= thresh."""
    if cs["n_gt"] == 0:
        nan = float("nan")
        return {"precision": nan, "recall": nan, "f1": nan, "tp": 0, "fp": 0, "n_kept": 0}
    scores = cs["scores"]
    # scores is descending, so -scores ascends: this counts entries with score >= thresh.
    n = int(np.searchsorted(-scores, -float(thresh), side="right"))
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "n_kept": 0}
    tp = float(cs["tp_cum"][n - 1]); fp = float(cs["fp_cum"][n - 1])
    p = tp / max(tp + fp, 1e-9)
    r = tp / cs["n_gt"]
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"precision": p, "recall": r, "f1": f1, "tp": int(tp), "fp": int(fp), "n_kept": n}


def best_f1(cs: Dict) -> Dict:
    """The best achievable F1 on the curve, and the score threshold that reaches it."""
    nan = float("nan")
    if cs["n_gt"] == 0 or len(cs["scores"]) == 0:
        return {"f1": nan, "score": nan, "precision": nan, "recall": nan}
    p, r = cs["precision"], cs["recall"]
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    i = int(np.argmax(f1))
    return {"f1": float(f1[i]), "score": float(cs["scores"][i]),
            "precision": float(p[i]), "recall": float(r[i])}


def micro_average(per_class: List[Dict], thresh: float) -> Dict:
    """Detection-pooled P/R/F1: every class's TPs and FPs summed before dividing."""
    tp = fp = n_gt = 0
    for cs in per_class:
        if cs["n_gt"] == 0:
            continue
        op = operating_point(cs, thresh)
        tp += op["tp"]; fp += op["fp"]; n_gt += cs["n_gt"]
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(n_gt, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
            "tp": tp, "fp": fp, "n_gt": n_gt}


def expected_calibration_error(calib, n_bins=15) -> float:
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


def _nanmean(vals) -> float:
    vals = [v for v in vals if v == v]      # drop NaN
    return float(np.mean(vals)) if vals else 0.0


def evaluate_split(model, cfg, device, split="test", occlusion_level=None,
                   operating_thresh: Optional[float] = None):
    """Full metric bundle for one split. Curves are returned under key 'curves'."""
    ev = cfg["eval"]
    thresh = float(ev.get("operating_thresh", 0.3) if operating_thresh is None
                   else operating_thresh)
    preds, gts = run_inference(model, cfg, device, split, occlusion_level,
                               score_thresh=ev["score_thresh"])
    num_classes = cfg["dataset"]["num_classes"]
    per_class, calib = compute_pr(preds, gts, num_classes, ev["iou_thresh"])

    rows = []
    for c, cs in enumerate(per_class):
        op = operating_point(cs, thresh)
        bf = best_f1(cs)
        rows.append({
            "class_id": c, "n_gt": cs["n_gt"], "n_det": cs["n_det"], "ap": cs["ap"],
            "precision": op["precision"], "recall": op["recall"], "f1": op["f1"],
            "tp": op["tp"], "fp": op["fp"], "n_kept": op["n_kept"],
            "best_f1": bf["f1"], "best_f1_score": bf["score"],
            "best_f1_precision": bf["precision"], "best_f1_recall": bf["recall"],
        })

    return {
        "operating_thresh": thresh,
        "iou_thresh": ev["iou_thresh"],
        "per_class": rows,
        "per_class_ap": [cs["ap"] for cs in per_class],       # kept for back-compat
        "mAP": _nanmean([cs["ap"] for cs in per_class]),
        "macro_precision": _nanmean([r["precision"] for r in rows]),
        "macro_recall": _nanmean([r["recall"] for r in rows]),
        "macro_f1": _nanmean([r["f1"] for r in rows]),
        "micro": micro_average(per_class, thresh),
        "ECE": expected_calibration_error(calib),
        "n_images": len(preds),
        "n_detections": int(sum(cs["n_det"] for cs in per_class)),
        "curves": per_class,        # numpy arrays — stripped before JSON
        "calib": calib,             # stripped before JSON
    }


# ------------------------------------------------------------------ reporting ---
def print_report(res: Dict, classes: List[str], title: str):
    t = res["operating_thresh"]
    print(f"\n=== {title} ===")
    print(f"  {'class':<20} {'n_gt':>6} {'n_det':>6} {'AP@.5':>7} "
          f"{'P@' + f'{t:.2f}':>8} {'R@' + f'{t:.2f}':>8} {'F1@' + f'{t:.2f}':>9} "
          f"{'bestF1':>7} {'@score':>7}")
    print("  " + "-" * 90)
    for name, r in zip(classes, res["per_class"]):
        ap = "  n/a  " if r["ap"] != r["ap"] else f"{r['ap']:7.3f}"
        if r["n_gt"] == 0:
            print(f"  {name:<20} {r['n_gt']:>6} {r['n_det']:>6} {ap:>7} "
                  f"{'—':>8} {'—':>8} {'—':>9} {'—':>7} {'—':>7}")
            continue
        print(f"  {name:<20} {r['n_gt']:>6} {r['n_det']:>6} {ap:>7} "
              f"{r['precision']:8.3f} {r['recall']:8.3f} {r['f1']:9.3f} "
              f"{r['best_f1']:7.3f} {r['best_f1_score']:7.2f}")
    print("  " + "-" * 90)
    print(f"  {'macro (mean of classes)':<34} {res['mAP']:7.3f} "
          f"{res['macro_precision']:8.3f} {res['macro_recall']:8.3f} {res['macro_f1']:9.3f}")
    m = res["micro"]
    print(f"  {'micro (pooled detections)':<34} {'':>7} "
          f"{m['precision']:8.3f} {m['recall']:8.3f} {m['f1']:9.3f}"
          f"   (TP {m['tp']}, FP {m['fp']}, GT {m['n_gt']})")
    print(f"\n  mAP@{res['iou_thresh']} = {res['mAP']:.3f}   ECE = {res['ECE']:.3f}   "
          f"images = {res['n_images']}   detections = {res['n_detections']}")


def _json_safe(res: Dict) -> Dict:
    """Drop the numpy curve arrays so the result dict is JSON-serializable."""
    return {k: v for k, v in res.items() if k not in ("curves", "calib")}


def save_curves_npz(path: str, classes: List[str], res: Dict):
    """Raw PR points for replotting (e.g. paper figures) without re-running inference."""
    arrays = {}
    for name, cs in zip(classes, res["curves"]):
        key = name.replace("/", "_")
        arrays[f"{key}__recall"] = cs["recall"]
        arrays[f"{key}__precision"] = cs["precision"]
        arrays[f"{key}__scores"] = cs["scores"]
    np.savez_compressed(path, **arrays)


def make_plots(out_dir: str, classes: List[str], overall: Dict, occlusion: Dict):
    """Reliability diagram, per-class AP bars, PR curves, occlusion trend."""
    from viz.calibration import (occlusion_bar, per_class_ap_bar, pr_curves,
                                 reliability_diagram)
    os.makedirs(out_dir, exist_ok=True)
    written = []

    p = os.path.join(out_dir, "pr_curves.png")
    pr_curves(classes, overall["curves"], p, operating_thresh=overall["operating_thresh"])
    written.append(p)

    p = os.path.join(out_dir, "reliability.png")
    reliability_diagram(overall["calib"], p)
    written.append(p)

    p = os.path.join(out_dir, "per_class_ap.png")
    per_class_ap_bar(classes, overall["per_class_ap"], p)
    written.append(p)

    if occlusion:
        p = os.path.join(out_dir, "occlusion_map.png")
        occlusion_bar(list(occlusion), {k: v["mAP"] for k, v in occlusion.items()}, p)
        written.append(p)
        for lvl, res in occlusion.items():
            p = os.path.join(out_dir, f"pr_curves_{lvl}.png")
            pr_curves(classes, res["curves"], p,
                      operating_thresh=res["operating_thresh"],
                      title=f"Precision-Recall ({lvl}, IoU {res['iou_thresh']})")
            written.append(p)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m engine.evaluate")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--operating-thresh", type=float,
                    help="score threshold at which P/R/F1 are reported (default eval.operating_thresh)")
    ap.add_argument("--plots-dir", help="where figures go (default <output_dir>/plots)")
    ap.add_argument("--no-plots", action="store_true", help="metrics only, skip matplotlib")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    device = get_device()
    model = build_model(cfg, in_channels(cfg)).to(device)
    load_checkpoint(args.weights, model, map_location=device)
    classes = cfg["dataset"]["classes"]
    out_dir = cfg["training"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    overall = evaluate_split(model, cfg, device, "test", None, args.operating_thresh)
    print_report(overall, classes, "Overall (test)")

    occlusion: Dict[str, Dict] = {}
    for lvl in cfg["dataset"].get("occlusion_levels", []):
        try:
            occlusion[lvl] = evaluate_split(model, cfg, device, "test", lvl,
                                            args.operating_thresh)
        except FileNotFoundError as e:
            print(f"\n[eval] skip {lvl}: {e}")
    for lvl, res in occlusion.items():
        print_report(res, classes, f"Occlusion {lvl}")
    if occlusion:
        print("\n=== Occlusion summary (expect a decline) ===")
        for lvl, res in occlusion.items():
            print(f"  {lvl}:  mAP@0.5 = {res['mAP']:.3f}   "
                  f"macro F1 = {res['macro_f1']:.3f}   images = {res['n_images']}")

    results = {"overall": _json_safe(overall),
               "occlusion": {k: _json_safe(v) for k, v in occlusion.items()}}
    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] wrote {json_path}")

    npz_path = os.path.join(out_dir, "pr_curves.npz")
    save_curves_npz(npz_path, classes, overall)
    print(f"[eval] wrote {npz_path}")

    if not args.no_plots:
        plots_dir = args.plots_dir or os.path.join(out_dir, "plots")
        try:
            for p in make_plots(plots_dir, classes, overall, occlusion):
                print(f"[eval] wrote {p}")
        except ImportError as e:
            print(f"[eval] plots skipped ({e}); metrics above are unaffected. "
                  f"pip install matplotlib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
