"""Local operator demo (runs on the 4 GB RTX 2050).

  python app/gradio_demo.py --config configs/xdetr_opixray.yaml --weights runs/opixray_xdetr/last.pth

Upload/select an X-ray -> detections + attention heatmaps + ranked operator attention map.
Uses fp16 on CUDA when available so it fits comfortably in 4 GB.
"""
from __future__ import annotations

import argparse
import os
import sys

import gradio as gr
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import get_device, load_config
from viz.common import load_model, infer_single, tensor_to_rgb
from viz.detect_overlay import draw_on_original
from viz.attention import decode_detections, cross_attention_map, encoder_saliency
from viz.gradcam import EigenCAM
from viz.operator_map import operator_attention_map
from viz.heatmap import overlay


def build_ui(model, cfg, device, classes):
    size = cfg["input"]["size"]
    cam = EigenCAM(model)

    def run(pil: Image.Image, score_thresh: float):
        if pil is None:
            return None, None, None, None, "Upload an X-ray image."
        result, caches, tensor, target = infer_single(model, pil, cfg, device, score_thresh)
        square = tensor_to_rgb(tensor, cfg)

        det_img = draw_on_original(pil, result, classes)
        cam_img = overlay(square, cam(tensor, size), alpha=0.55, cmap="inferno")

        dets = decode_detections(caches, score_thresh)
        if dets:
            cxa = cross_attention_map(caches, dets[0]["query"], size)
            attn_img = overlay(square, cxa, alpha=0.6, cmap="magma")
        else:
            attn_img = overlay(square, encoder_saliency(caches, size), alpha=0.55, cmap="viridis")

        op_map, op_dets = operator_attention_map(caches, size, max(0.15, score_thresh - 0.1))
        op_img = overlay(square, op_map, alpha=0.55, cmap="inferno")

        lines = [f"#{d['priority']}  {classes[d['label']]}  conf={d['score']:.2f}" for d in op_dets[:10]]
        summary = "**Ranked attention (look here first):**\n\n" + ("\n".join(lines) or "no items above threshold")
        return det_img, cam_img, attn_img, op_img, summary

    with gr.Blocks(title="X-DETR — X-ray Screening Assist") as demo:
        gr.Markdown("# X-DETR — Prohibited-Item Screening Assist\n"
                    "Decision support: highlights and *ranks* regions for the operator. "
                    "The operator decision remains the authority.")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="X-ray image")
                thr = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="score threshold")
                btn = gr.Button("Analyze", variant="primary")
                summary = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Row():
                    o1 = gr.Image(label="detections")
                    o2 = gr.Image(label="Eigen-CAM")
                with gr.Row():
                    o3 = gr.Image(label="cross-attention")
                    o4 = gr.Image(label="operator attention (ranked)")
        btn.click(run, [inp, thr], [o1, o2, o3, o4, summary])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    device = get_device()
    model = load_model(cfg, args.weights or None, device)
    classes = cfg["dataset"]["classes"]
    print(f"[demo] device={device}. Launching Gradio...")
    build_ui(model, cfg, device, classes).launch(share=args.share)


if __name__ == "__main__":
    main()
