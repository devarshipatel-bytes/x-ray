"""Verify the dataset is laid out and parseable BEFORE starting a long training run.

Reports, per split: which folders were resolved, image/annotation counts, how many boxes
parsed, the per-class distribution, and any annotation class token the config does not
know about (the most common silent failure — a name mismatch makes boxes vanish without
an error). Also checks the OL1/OL2/OL3 occlusion lists used by engine.evaluate.

  python -m scripts.check_dataset --config configs/xdetr_opixray.yaml
  python -m scripts.check_dataset --config configs/xdetr_opixray.yaml --data-root /mnt/d/OPIXray

Exit code 0 = ready to train, 1 = something needs fixing.
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import Counter

from data.transforms import build_eval_transform
from engine.config import load_config, set_dotted


def _describe_split(cfg, split: str, classes) -> tuple[bool, Counter, Counter]:
    """Returns (ok, per_class_box_counts, unknown_class_tokens)."""
    from data.opixray import OPIXrayDataset

    print(f"\n--- split: {split} ---")
    root = cfg["dataset"]["root"]
    if not os.path.isdir(os.path.join(root, split)):
        print(f"  MISSING: {os.path.join(root, split)}")
        return False, Counter(), Counter()

    # Resolve the same directories the loader will use.
    probe = OPIXrayDataset.__new__(OPIXrayDataset)
    probe.root = root
    probe.ann_format = cfg["dataset"].get("ann_format", "opixray_txt")
    img_dir, ann_dir = probe._split_dirs(split)
    ext = "*.xml" if probe.ann_format == "voc_xml" else "*.txt"
    n_imgs = sum(len(glob.glob(os.path.join(img_dir, f"*{e}")))
                 for e in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"))
    ann_files = sorted(glob.glob(os.path.join(ann_dir, ext)))
    print(f"  images      : {n_imgs:>6}  in {img_dir}")
    print(f"  annotations : {len(ann_files):>6}  in {ann_dir}")
    if n_imgs == 0 or not ann_files:
        print("  FAIL: need both images and annotations here.")
        return False, Counter(), Counter()

    # Surface class tokens the config does not recognize.
    known = {OPIXrayDataset._norm(c) for c in classes}
    unknown: Counter = Counter()
    if probe.ann_format != "voc_xml":
        for path in ann_files[:2000]:
            with open(path, errors="ignore") as f:
                for line in f:
                    toks = line.strip().split()
                    if len(toks) < 5:
                        continue
                    tok = toks[-5]
                    if OPIXrayDataset._norm(tok) not in known:
                        unknown[tok] += 1

    ds = OPIXrayDataset(cfg, split, build_eval_transform(cfg))
    per_class: Counter = Counter()
    empty = 0
    for s in ds.samples:
        if len(s["labels"]) == 0:
            empty += 1
        for lab in s["labels"]:
            per_class[classes[int(lab)]] += 1
    matched = len(ds.samples)
    print(f"  matched pairs: {matched:>6}"
          f"   ({len(ann_files) - matched} annotations had no image file)")
    print(f"  boxes        : {sum(per_class.values()):>6}"
          f"   ({empty} images with zero usable boxes)")
    for name in classes:
        print(f"      {name:<20} {per_class[name]:>6}")

    ok = matched > 0 and sum(per_class.values()) > 0
    if not ok:
        print("  FAIL: no usable boxes parsed — check the annotation format and class names.")
    return ok, per_class, unknown


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.check_dataset")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", help="override dataset.root")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.data_root:
        set_dotted(cfg, "dataset.root", args.data_root)
    root = cfg["dataset"]["root"]
    classes = list(cfg["dataset"]["classes"])

    print("=" * 78)
    print("dataset check")
    print("=" * 78)
    print(f"  root    : {root}  ({'exists' if os.path.isdir(root) else 'MISSING'})")
    print(f"  format  : {cfg['dataset'].get('ann_format', 'opixray_txt')}")
    print(f"  classes : {', '.join(classes)}")
    if not os.path.isdir(root):
        print("\nFAIL: dataset.root does not exist. Point --data-root at your unzipped copy, "
              "or move it to this path.")
        return 1

    ok = True
    all_unknown: Counter = Counter()
    for split in ("train", "test"):
        try:
            split_ok, _, unknown = _describe_split(cfg, split, classes)
        except Exception as e:                       # loader raises on an unusable split
            print(f"  FAIL: {type(e).__name__}: {e}")
            split_ok, unknown = False, Counter()
        ok = ok and split_ok
        all_unknown.update(unknown)

    print("\n--- occlusion lists (engine.evaluate) ---")
    from data.opixray import OPIXrayDataset
    probe = OPIXrayDataset.__new__(OPIXrayDataset)
    probe.root = root
    for lvl in cfg["dataset"].get("occlusion_levels", []):
        try:
            stems = probe._occlusion_stems(lvl)
            print(f"  {lvl}: {len(stems)} image stems")
        except FileNotFoundError:
            print(f"  {lvl}: not found — overall eval still works, "
                  f"occlusion-stratified numbers will be skipped")

    if all_unknown:
        ok = False
        print("\n--- unrecognized class tokens in annotations ---")
        for tok, n in all_unknown.most_common(15):
            print(f"  {tok!r}: {n} lines  (silently dropped)")
        print("  FIX: make dataset.classes in the config match these names exactly "
              "(matching ignores case, spaces, - and _).")

    print("\n" + "=" * 78)
    if ok:
        print("PASS — dataset looks good. Next: python -m scripts.sanity_data "
              f"--config {args.config} --n 6 --out assets/sanity.png")
        return 0
    print("FAIL — fix the issues above before training. "
          "See scripts/download_opixray.md for the expected layout.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
