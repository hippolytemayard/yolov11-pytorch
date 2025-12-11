"""
Detection Head for YOLO v11.

This module implements the detection head that predicts bounding boxes
and class probabilities for object detection.
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from yolo.models.common import Conv
from yolo.utils.boxes import make_anchors


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module.
    
    Converts discrete distribution predictions to continuous box coordinates
    using a learnable linear combination.
    
    Reference: https://ieeexplore.ieee.org/document/9792391
    
    Args:
        ch: Number of distribution bins (default: 16)
    """
    
    def __init__(self, ch: int = 16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(ch, 1, kernel_size=1, bias=False).requires_grad_(False)
        # Initialize with linear weights [0, 1, 2, ..., ch-1]
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = nn.Parameter(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert distribution to box coordinates.
        
        Args:
            x: Input tensor of shape (B, 4*ch, A) where A is number of anchors
            
        Returns:
            Box coordinates of shape (B, 4, A)
        """
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class DetectHead(nn.Module):
    """
    Detection head for YOLO v11.
    
    Predicts bounding boxes and class probabilities from multi-scale features.
    Uses decoupled heads for box and class predictions.
    
    Args:
        num_classes: Number of object classes (default: 80 for COCO)
        filters: Tuple of channel sizes for each scale level
    """
    
    # Shared tensors across instances
    anchors: torch.Tensor
    strides: torch.Tensor

    def __init__(self, num_classes: int = 80, filters: Tuple[int, ...] = ()):
        super().__init__()
        self.ch = 16  # DFL channels (distribution bins)
        self.nc = num_classes
        self.nl = len(filters)  # Number of detection layers
        self.no = num_classes + self.ch * 4  # Outputs per anchor
        self.stride = torch.zeros(self.nl)  # Computed during build
        
        # Register empty tensors
        self.register_buffer('anchors', torch.empty(0))
        self.register_buffer('strides', torch.empty(0))

        # Compute head dimensions
        box_ch = max(64, filters[0] // 4)
        cls_ch = max(80, filters[0], num_classes)

        # DFL module for box regression
        self.dfl = DFL(self.ch)
        
        # Box regression heads (one per scale)
        self.box = nn.ModuleList([
            nn.Sequential(
                Conv(f, box_ch, nn.SiLU(), k=3, p=1),
                Conv(box_ch, box_ch, nn.SiLU(), k=3, p=1),
                nn.Conv2d(box_ch, 4 * self.ch, kernel_size=1)
            ) for f in filters
        ])
        
        # Classification heads (one per scale)
        self.cls = nn.ModuleList([
            nn.Sequential(
                Conv(f, f, nn.SiLU(), k=3, p=1, g=f),  # Depthwise
                Conv(f, cls_ch, nn.SiLU()),
                Conv(cls_ch, cls_ch, nn.SiLU(), k=3, p=1, g=cls_ch),  # Depthwise
                Conv(cls_ch, cls_ch, nn.SiLU()),
                nn.Conv2d(cls_ch, num_classes, kernel_size=1)
            ) for f in filters
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass for detection head.
        
        Args:
            features: List of feature maps from FPN at different scales
            
        Returns:
            During training: List of raw predictions per scale
            During inference: Concatenated predictions with decoded boxes
        """
        outputs = []
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            outputs.append(torch.cat([box(features[i]), cls(features[i])], dim=1))

        if self.training:
            return outputs

        # Inference mode: decode predictions
        self.anchors, self.strides = (
            t.transpose(0, 1) for t in make_anchors(outputs, self.stride)
        )
        
        # Reshape and concatenate
        x = torch.cat([o.view(outputs[0].shape[0], self.no, -1) for o in outputs], dim=2)
        box, cls = x.split((4 * self.ch, self.nc), dim=1)

        # Decode boxes using DFL
        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), dim=1)

        return torch.cat((box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self) -> None:
        """
        Initialize biases for better convergence.
        
        Box biases are set to 1.0, class biases are set based on
        class prior probability.
        """
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # Box bias
            box[-1].bias.data[:] = 1.0
            # Class bias (prior: ~0.01 objects per anchor)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

