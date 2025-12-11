"""
Instance Segmentation Head for YOLO v11.

This module implements instance segmentation by extending the detection head
with mask prediction capabilities using prototype masks.
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo.models.common import Conv
from yolo.models.heads.detect import DetectHead


class Proto(nn.Module):
    """
    Prototype mask generation module for segmentation.
    
    Generates a set of prototype masks that are linearly combined
    with predicted coefficients to produce instance masks.
    
    Architecture matches Ultralytics YOLOv11:
    - Conv1: in_ch -> c_ (3x3)
    - Upsample: c_ -> c_ (ConvTranspose2d 2x2)
    - Conv2: c_ -> c_ (3x3)
    - Conv3: c_ -> nm (1x1)
    
    Args:
        in_ch: Input channels from backbone
        c_: Intermediate channels (= in_ch for Ultralytics)
        nm: Number of prototype masks (default: 32)
    """
    
    def __init__(self, in_ch: int, c_: int = 64, nm: int = 32):
        super().__init__()
        self.conv1 = Conv(in_ch, c_, nn.SiLU(), k=3, p=1)
        self.upsample = nn.ConvTranspose2d(c_, c_, kernel_size=2, stride=2, bias=True)
        self.conv2 = Conv(c_, c_, nn.SiLU(), k=3, p=1)
        self.conv3 = Conv(c_, nm, nn.SiLU(), k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate prototype masks from features."""
        return self.conv3(self.conv2(self.upsample(self.conv1(x))))


class SegmentHead(DetectHead):
    """
    Instance segmentation head for YOLO v11.
    
    Extends the detection head with:
    - Mask prototype generation (Proto module)
    - Mask coefficient prediction for each detection
    
    The final instance mask is computed as:
        mask = sigmoid(mask_coefficients @ prototypes)
    
    Architecture matches Ultralytics YOLOv11 exactly:
    - Proto uses c_ = filters[0] internal channels
    - Mask heads use c4 = max(filters[0] // 4, nm) as internal channels
      (SINGLE value for ALL scales, not per-scale)
    
    Args:
        num_classes: Number of object classes
        filters: Tuple of channel sizes for each scale level
        num_masks: Number of mask prototypes (default: 32)
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        filters: Tuple[int, ...] = (),
        num_masks: int = 32
    ):
        super().__init__(num_classes, filters)
        
        self.nm = num_masks  # Number of mask prototypes
        self.no = num_classes + self.ch * 4 + num_masks  # Outputs per anchor
        
        # Proto intermediate channels = filters[0] (matches Ultralytics)
        proto_ch = filters[0]
        
        # Prototype mask generator (operates on highest resolution feature)
        self.proto = Proto(filters[0], proto_ch, num_masks)
        
        # Mask coefficient heads (one per scale)
        # Ultralytics uses c4 = max(filters[0] // 4, nm) as a SINGLE value for ALL scales
        mask_ch = max(filters[0] // 4, num_masks)
        self.mask = nn.ModuleList([
            nn.Sequential(
                Conv(f, mask_ch, nn.SiLU(), k=3, p=1),
                Conv(mask_ch, mask_ch, nn.SiLU(), k=3, p=1),
                nn.Conv2d(mask_ch, num_masks, kernel_size=1)
            ) for f in filters
        ])

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass for segmentation head.
        
        Args:
            features: List of feature maps from FPN
            
        Returns:
            Tuple of:
                - Detection outputs (list or concatenated tensor)
                - Prototype masks
        """
        # Generate prototypes from highest resolution feature
        proto = self.proto(features[0])
        
        outputs = []
        for i, (box, cls, mask) in enumerate(zip(self.box, self.cls, self.mask)):
            box_out = box(features[i])
            cls_out = cls(features[i])
            mask_out = mask(features[i])
            outputs.append(torch.cat([box_out, cls_out, mask_out], dim=1))

        if self.training:
            return outputs, proto

        # Inference mode
        self.anchors, self.strides = (
            t.transpose(0, 1) 
            for t in self._make_anchors(outputs, self.stride)
        )
        
        # Reshape and concatenate
        x = torch.cat([o.view(outputs[0].shape[0], self.no, -1) for o in outputs], dim=2)
        box, cls, masks = x.split((4 * self.ch, self.nc, self.nm), dim=1)

        # Decode boxes
        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), dim=1)

        return torch.cat((box * self.strides, cls.sigmoid(), masks), dim=1), proto

    def _make_anchors(
        self,
        features: List[torch.Tensor],
        strides: torch.Tensor,
        offset: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor points for all feature levels."""
        from yolo.utils.boxes import make_anchors
        return make_anchors(features, strides, offset)

    @staticmethod
    def process_masks(
        protos: torch.Tensor,
        masks_in: torch.Tensor,
        bboxes: torch.Tensor,
        shape: Tuple[int, int],
        upsample: bool = False
    ) -> torch.Tensor:
        """
        Process mask predictions to generate instance masks.
        
        Args:
            protos: Prototype masks (B, nm, H, W)
            masks_in: Mask coefficients (N, nm) for N detections
            bboxes: Bounding boxes (N, 4) in xyxy format
            shape: Target output shape (H, W)
            upsample: Whether to upsample masks to target shape
            
        Returns:
            Instance masks of shape (N, H, W)
        """
        c, mh, mw = protos.shape[1:]
        
        # Compute instance masks
        masks = (masks_in @ protos.view(c, -1)).sigmoid().view(-1, mh, mw)
        
        # Crop masks to bounding boxes
        masks = SegmentHead._crop_mask(masks, bboxes, shape)
        
        if upsample:
            masks = F.interpolate(masks.unsqueeze(0), shape, mode='bilinear', align_corners=False)[0]
        
        return masks

    @staticmethod
    def _crop_mask(masks: torch.Tensor, boxes: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """
        Crop masks to bounding box regions.
        
        Args:
            masks: Predicted masks (N, H, W)
            boxes: Bounding boxes (N, 4) in xyxy format
            shape: Original image shape (H, W)
            
        Returns:
            Cropped masks
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
        
        # Create coordinate grids
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
        
        # Apply box constraints
        scale_w, scale_h = w / shape[1], h / shape[0]
        return masks * ((r >= x1 * scale_w) * (r < x2 * scale_w) * 
                        (c >= y1 * scale_h) * (c < y2 * scale_h))
