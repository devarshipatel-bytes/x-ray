# X-DETR — Occlusion-Robust Prohibited-Item Detection in X-ray Security Imagery

A **custom, RF-DETR/RT-DETR-inspired object detector** for prohibited-item detection in
dual-energy X-ray screening images. Trains on a **single consumer GPU (12 GB)** and runs
inference + rich visualizations on far less. Screening *decision support* — it highlights
and **ranks** regions for a human operator; the operator decision stays the authority.

> Scope: authorized screening-decision-support research only. Not guidance on concealment
> or defeating screening.

## Why this design (the honest constraints)

- **One consumer GPU, no cluster** → AMP fp16, gradient accumulation for a larger effective
  batch, no custom CUDA ops, and per-epoch checkpointing so any crash is resumable.
- **No dual-energy physics in public data** → public sets ship *pseudo-color RGB*; true
  effective-Z needs raw scanner data. We approximate with **material-proxy channels** derived
  from the color LUT (`data/material_proxy.py`) — a proxy, clearly labeled as such.

## Model: X-DETR (~30–36 M params)

`image (3 or 5 ch) → ResNet-50 → P3/P4/P5 → input_proj → HybridEncoder(AIFI + CCFM) →
top-k query selection → TransformerDecoder (self+cross attn, iterative box refinement) →
class (focal) + box (L1+GIoU) heads`, matched with a Hungarian matcher. Pure PyTorch — **no
custom CUDA op** — so it runs anywhere. See `models/xdetr.py`.

The X-ray-specific contribution is the **5-channel material-proxy input** (RGB + organic/metal
proxies); ablate by setting `input.use_material_proxy: false` (or `--no-material-proxy`).

## Layout

```
configs/   xdetr_opixray.yaml        # one config drives everything (CLI-overridable)
data/      opixray.py  transforms.py  material_proxy.py  coco_style.py (PIDray/HiXray)
models/    xdetr.py backbone.py encoder.py decoder.py heads.py matcher.py losses.py box_ops.py
engine/    train.py  evaluate.py  checkpoint.py  config.py
viz/       gallery.py detect_overlay.py attention.py gradcam.py operator_map.py
           calibration.py heatmap.py common.py fiftyone_app.py
app/       gradio_demo.py            # local operator UI
scripts/   check_dataset.py sanity_data.py overfit.py gallery_batch.py export_onnx.py
           download_opixray.md
notebooks/ infer_viz.ipynb
```

## Quickstart

```bash
pip install -r requirements.txt

# 1) get OPIXray (see scripts/download_opixray.md) into data/OPIXray, then verify the layout
python -m scripts.check_dataset --config configs/xdetr_opixray.yaml

# 2) eyeball the loader: boxes, letterboxing, material-proxy channels
python -m scripts.sanity_data --config configs/xdetr_opixray.yaml --n 6 --out assets/sanity.png

# 3) wiring gate — overfit 10 real images (boxes should snap), must print PASS
python -m scripts.overfit --config configs/xdetr_opixray.yaml --n 10 --iters 300

# 4) train
python -m engine.train --config configs/xdetr_opixray.yaml --epochs 50 --batch-size 8

# 5) evaluate: per-class AP@0.5, occlusion OL1/2/3, ECE
python -m engine.evaluate --config configs/xdetr_opixray.yaml --weights runs/opixray_xdetr/final.pth

# 6) visualization galleries (6-panel: detections + heatmaps + operator map)
python -m scripts.gallery_batch --config configs/xdetr_opixray.yaml \
    --weights runs/opixray_xdetr/final.pth --n 12 --out assets/galleries

# 7) local operator demo
python app/gradio_demo.py --config configs/xdetr_opixray.yaml --weights runs/opixray_xdetr/final.pth
```

## Training

Every hyperparameter has a flag — `python -m engine.train --help` lists them all. Flags
override the YAML; `--set dotted.key=value` reaches anything without a flag.

```bash
# defaults from the config (50 epochs, batch 8, 512px, fp16)
python -m engine.train --config configs/xdetr_opixray.yaml

# explicit hyperparameters
python -m engine.train --config configs/xdetr_opixray.yaml \
    --epochs 80 --batch-size 8 --img-size 512 --lr 1e-4 --ckpt-interval 10

# 1-minute smoke test before a long run
python -m engine.train --config configs/xdetr_opixray.yaml \
    --epochs 1 --limit 20 --batch-size 2 --img-size 320 --log-interval 1

# resume (automatic if runs/<dir>/last.pth exists)
python -m engine.train --config configs/xdetr_opixray.yaml
```

Checkpoints land in `--output-dir` (default `runs/opixray_xdetr`):

| file | when |
|---|---|
| `last.pth` | overwritten every epoch — the auto-resume point |
| `epoch_XXX.pth` | every `--ckpt-interval` epochs (default 10), never overwritten |
| `final.pth` | on completion |
| `train_log.jsonl`, `config_used.yaml` | per-epoch metrics and the exact resolved config |

### Fitting your VRAM

VRAM scales with `batch_size × img_size²`, then with `dec_layers` and `num_queries`.
If you hit `CUDA out of memory`, apply in this order — the first two cost you nothing in
final accuracy:

```bash
--batch-size 4 --grad-accum 2      # same effective batch of 8, half the activations
--img-size 384                     # cheapest real win; costs some small-object recall
--dec-layers 3 --num-queries 100   # ~40% faster, measurably lower AP
--backbone resnet18                # last resort
```

Rough guide at 512px with `--amp`: **12 GB** → `--batch-size 8`; **8 GB** → `--batch-size 4
--grad-accum 2`; **6 GB** → add `--img-size 384`; **4 GB** → inference only.

Keep `--amp` on unless you are debugging numerics. `--workers` should be near your physical
core count — JPEG decode of the 1225×954 originals, not the GPU, is usually the bottleneck.

## Visualizations

- **Detections** — clean per-class boxes (`viz/detect_overlay.py`).
- **Decoder cross-attention** — per-object "where the model looked" (`viz/attention.py`).
- **Eigen-CAM** — gradient-free backbone saliency (`viz/gradcam.py`).
- **Operator attention map** — confidence-weighted, **ranked** "look here first" (`viz/operator_map.py`).
- **Calibration** — reliability diagram + ECE, per-class AP bars, occlusion trend (`viz/calibration.py`).
- **FiftyOne** — interactive FP/FN failure gallery (`viz/fiftyone_app.py`, optional).

`notebooks/infer_viz.ipynb` walks through all of it against a trained checkpoint.

## Switching datasets

`dataset.name: pidray` (or `hixray`) uses the shared COCO-json loader (`data/coco_style.py`) —
set `root`, ann files, and `classes`. OPIXray occlusion-stratified eval is OPIXray-specific.
