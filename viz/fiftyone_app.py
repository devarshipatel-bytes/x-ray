"""Interactive FP/FN failure browsing with FiftyOne (optional dependency).

  pip install fiftyone
  python -m viz.fiftyone_app --config configs/xdetr_opixray.yaml --weights runs/opixray_xdetr/last.pth

Loads the OPIXray test split, runs X-DETR, attaches GT + predictions as FiftyOne fields,
and launches the app so you can filter by false positives / false negatives and eyeball them.
"""
from __future__ import annotations

import argparse

import torch

from data import build_dataset, in_channels
from engine.checkpoint import load_checkpoint
from engine.config import get_device, load_config
from engine.evaluate import _gt_to_original
from models import build_model, postprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--score", type=float, default=0.3)
    args = ap.parse_args()

    import fiftyone as fo  # lazy import

    cfg = load_config(args.config)
    device = get_device()
    classes = cfg["dataset"]["classes"]
    model = build_model(cfg, in_channels(cfg)).to(device)
    load_checkpoint(args.weights, model, map_location=device)
    model.eval()

    ds = build_dataset(cfg, split="test", train=False)
    samples = []
    for i in range(min(args.limit, len(ds))):
        img, target = ds[i]
        path = ds.samples[i]["image"]
        oh, ow = int(target["orig_size"][0]), int(target["orig_size"][1])
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device))
        res = postprocess(out, [target], score_thresh=args.score)[0]

        s = fo.Sample(filepath=path)
        gt_boxes = _gt_to_original(target["boxes"], target).numpy()
        s["ground_truth"] = fo.Detections(detections=[
            _det(classes, int(l), b, ow, oh) for l, b in zip(target["labels"].numpy(), gt_boxes)])
        s["predictions"] = fo.Detections(detections=[
            _det(classes, int(l), b, ow, oh, float(sc))
            for l, b, sc in zip(res["labels"].numpy(), res["boxes"].numpy(), res["scores"].numpy())])
        samples.append(s)

    dataset = fo.Dataset("xdetr_opixray")
    dataset.add_samples(samples)
    dataset.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")
    print("[fiftyone] launching app; filter eval_fp / eval_fn to inspect failures.")
    session = fo.launch_app(dataset)
    session.wait()


def _det(classes, label, box, ow, oh, conf=None):
    import fiftyone as fo
    x1, y1, x2, y2 = box
    rel = [x1 / ow, y1 / oh, (x2 - x1) / ow, (y2 - y1) / oh]
    kw = {"label": classes[label], "bounding_box": rel}
    if conf is not None:
        kw["confidence"] = conf
    return fo.Detection(**kw)


if __name__ == "__main__":
    main()
