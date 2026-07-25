"""Eigen-CAM class-activation heatmap (gradient-free, robust for DETR-style models).

Hooks the backbone's last conv stage (P5), takes the first principal component of the
activation map as the saliency. No backprop needed -> cheap and stable on 4 GB.
"""
from __future__ import annotations

import numpy as np
import torch


class EigenCAM:
    def __init__(self, model, target_module=None):
        self.model = model
        self.act = None
        mod = target_module if target_module is not None else model.backbone.layer4
        mod.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.act = out.detach()

    @torch.no_grad()
    def __call__(self, tensor: torch.Tensor, size: int) -> np.ndarray:
        device = next(self.model.parameters()).device
        self.model.eval()
        self.model(tensor.unsqueeze(0).to(device))
        A = self.act[0]                       # C,H,W
        C, H, W = A.shape
        X = A.reshape(C, H * W).T.float()     # HW,C
        X = X - X.mean(0, keepdim=True)
        # first right singular vector -> principal projection
        _, _, Vt = torch.linalg.svd(X, full_matrices=False)
        cam = (X @ Vt[0]).reshape(H, W)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        from PIL import Image
        im = Image.fromarray((cam.cpu().numpy() * 255).astype(np.uint8)).resize((size, size))
        return np.asarray(im, dtype=np.float32) / 255.0
