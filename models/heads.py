"""Prediction heads: classification (focal) and box regression (MLP)."""
from __future__ import annotations

import math

import torch
from torch import nn

from .decoder import MLP


def build_class_head(dim: int, num_classes: int, prior_prob: float = 0.01) -> nn.Linear:
    head = nn.Linear(dim, num_classes)
    # focal-loss bias init so early training isn't dominated by the many negatives
    bias = -math.log((1 - prior_prob) / prior_prob)
    nn.init.constant_(head.bias, bias)
    return head


def build_bbox_head(dim: int) -> MLP:
    head = MLP(dim, dim, 4, num_layers=3)
    # start near identity refinement
    nn.init.constant_(head.layers[-1].weight, 0.0)
    nn.init.constant_(head.layers[-1].bias, 0.0)
    return head
