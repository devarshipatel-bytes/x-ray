"""ResNet backbone producing multi-scale features P3/P4/P5 (strides 8/16/32).

- Frozen BatchNorm (standard for DETR-family small-batch training).
- Stem + stage1 frozen (transfer-learning stability on small data).
- Adapts conv1 to N input channels (5 = RGB + 2 material proxies), initializing the
  extra channels from the mean of the pretrained RGB filters so we keep ImageNet priors.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torchvision


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm with fixed affine + running stats (no updates). DETR-standard."""

    def __init__(self, n: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        # present in standard BatchNorm2d state_dicts (incl. torchvision pretrained
        # weights); unused in the frozen forward pass but needed so load_state_dict matches.
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        return x * scale + (b - rm * scale)


_ARCH = {
    "resnet18": (torchvision.models.resnet18, [128, 256, 512]),
    "resnet34": (torchvision.models.resnet34, [128, 256, 512]),
    "resnet50": (torchvision.models.resnet50, [512, 1024, 2048]),
}


class Backbone(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = True, in_channels: int = 5):
        super().__init__()
        if name not in _ARCH:
            raise KeyError(f"backbone {name} not supported; choose {list(_ARCH)}")
        ctor, ch = _ARCH[name]
        weights = "DEFAULT" if pretrained else None
        net = ctor(weights=weights, norm_layer=FrozenBatchNorm2d)

        # adapt first conv to in_channels
        if in_channels != 3:
            old = net.conv1
            new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                new.weight[:, :3] = old.weight
                if in_channels > 3:
                    mean_w = old.weight.mean(dim=1, keepdim=True)
                    new.weight[:, 3:] = mean_w.repeat(1, in_channels - 3, 1, 1)
            net.conv1 = new

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2   # P3, stride 8
        self.layer3 = net.layer3   # P4, stride 16
        self.layer4 = net.layer4   # P5, stride 32
        self.num_channels: List[int] = ch  # [P3, P4, P5]

        # freeze stem + stage1
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for p in self.layer1.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"p3": c3, "p4": c4, "p5": c5}


def build_backbone(cfg: Dict, in_channels: int) -> Backbone:
    m = cfg["model"]
    return Backbone(m["backbone"], m.get("pretrained_backbone", True), in_channels)
