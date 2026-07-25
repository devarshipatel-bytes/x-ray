# X-DETR — Occlusion-Robust Prohibited-Item Detection in X-ray Security Imagery

A **custom, RF-DETR/RT-DETR-inspired object detector** for prohibited-item detection in
dual-energy X-ray screening images, built to be **trainable on Colab Free (T4 16 GB)** and to
**run inference + rich visualizations locally on a 4 GB GPU**. Screening *decision support* —
it highlights and **ranks** regions for a human operator; the operator decision stays the authority.

> Scope: authorized screening-decision-support research only. Not guidance on concealment
> or defeating screening.

## Why this design (the honest constraints)

- **4 GB VRAM can't train a DETR** → train on Colab (T4), infer/visualize locally (fp16 fits ~2–3 GB).
- **No dual-energy physics in public data** → public sets ship *pseudo-color RGB*; true
  effective-Z needs raw scanner data. We approximate with **material-proxy channels** derived
  from the color LUT (`data/material_proxy.py`) — a proxy, clearly labeled as such.
- **Colab Free disconnects** → training checkpoints to Drive **every epoch** and auto-resumes.

## Model: X-DETR (~30–36 M params)

`image (3 or 5 ch) → ResNet-50 → P3/P4/P5 → input_proj → HybridEncoder(AIFI + CCFM) →
top-k query selection → TransformerDecoder (self+cross attn, iterative box refinement) →
class (focal) + box (L1+GIoU) heads`, matched with a Hungarian matcher. Pure PyTorch — **no
custom CUDA op** — so it runs anywhere. See `models/xdetr.py`.

The X-ray-specific contribution is the **5-channel material-proxy input** (RGB + organic/metal
proxies); ablate by setting `input.use_material_proxy: false`.

## Layout

```
configs/   xdetr_opixray.yaml        # one config drives everything (CLI-overridable)
data/      opixray.py  transforms.py  material_proxy.py  coco_style.py (PIDray/HiXray)
models/    xdetr.py backbone.py encoder.py decoder.py heads.py matcher.py losses.py box_ops.py
engine/    train.py  evaluate.py  checkpoint.py  config.py
viz/       gallery.py detect_overlay.py attention.py gradcam.py operator_map.py
           calibration.py heatmap.py common.py fiftyone_app.py
app/       gradio_demo.py            # local 4 GB operator UI
scripts/   overfit.py sanity_data.py gallery_batch.py export_onnx.py download_opixray.md
notebooks/ colab_train.ipynb  colab_infer_viz.ipynb
```

## Quickstart

```bash
pip install -r requirements.txt          # torch/torchvision preinstalled on Colab

# 0) wiring gate — NO dataset needed, must print PASS
python -m scripts.overfit --config configs/xdetr_opixray.yaml --synthetic --iters 60

# 1) get OPIXray (see scripts/download_opixray.md), then sanity-check the data
python -m scripts.sanity_data --config configs/xdetr_opixray.yaml --n 6 --out assets/sanity.png

# 2) overfit 10 real images (boxes should snap) — correctness on real data
python -m scripts.overfit --config configs/xdetr_opixray.yaml --n 10 --iters 300

# 3) train (Colab T4). Locally you can smoke-test with tiny settings:
python -m engine.train --config configs/xdetr_opixray.yaml \
    --set training.epochs=50 training.output_dir=runs/opixray_xdetr

# 4) evaluate: per-class AP@0.5, occlusion OL1/2/3, ECE
python -m engine.evaluate --config configs/xdetr_opixray.yaml --weights runs/opixray_xdetr/last.pth

# 5) visualization galleries (6-panel: detections + heatmaps + operator map)
python -m scripts.gallery_batch --config configs/xdetr_opixray.yaml \
    --weights runs/opixray_xdetr/last.pth --n 12 --out assets/galleries

# 6) local operator demo (4 GB GPU)
python app/gradio_demo.py --config configs/xdetr_opixray.yaml --weights runs/opixray_xdetr/last.pth
```

Train on Colab with the notebooks in `notebooks/` (Drive checkpointing + resume built in).

## Visualizations

- **Detections** — clean per-class boxes (`viz/detect_overlay.py`).
- **Decoder cross-attention** — per-object "where the model looked" (`viz/attention.py`).
- **Eigen-CAM** — gradient-free backbone saliency (`viz/gradcam.py`).
- **Operator attention map** — confidence-weighted, **ranked** "look here first" (`viz/operator_map.py`).
- **Calibration** — reliability diagram + ECE, per-class AP bars, occlusion trend (`viz/calibration.py`).
- **FiftyOne** — interactive FP/FN failure gallery (`viz/fiftyone_app.py`, optional).

## Switching datasets

`dataset.name: pidray` (or `hixray`) uses the shared COCO-json loader (`data/coco_style.py`) —
set `root`, ann files, and `classes`. OPIXray occlusion-stratified eval is OPIXray-specific.

## Tuning for Colab-Free speed

Drop `model.dec_layers` 6→3 and `model.num_queries` 300→100, keep `input.size: 512`,
`training.amp: true`, `training.batch_size: 4`, `training.grad_accum: 2`. ~7–15 min/epoch on T4.
