"""
Pose Estimation Head for YOLO v11.

This module implements human pose estimation by extending the detection head
with keypoint prediction capabilities.
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from yolo.models.common import Conv
from yolo.models.heads.detect import DetectHead, DFL


class PoseHead(DetectHead):
    """
    Pose estimation head for YOLO v11.
    
    Extends the detection head with keypoint prediction for human pose estimation.
    Each keypoint has (x, y, visibility) predictions.
    
    Architecture matches Ultralytics YOLOv11:
    - cls head: uses max(num_classes, filters[0]) as internal channels 
    - kpt head: uses max(nk*3, filters[0]//4) as internal channels
      (51 for n/s, 64 for m/l, 96 for x)
    
    Args:
        num_classes: Number of object classes (typically 1 for person)
        filters: Tuple of channel sizes for each scale level
        num_keypoints: Number of keypoints to predict (default: 17 for COCO)
    """
    
    # COCO keypoint names for reference
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # COCO skeleton connections (pairs of keypoint indices)
    SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # legs
        [6, 12], [7, 13],  # body
        [6, 7],  # shoulders
        [6, 8], [7, 9], [8, 10], [9, 11],  # arms
        [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]  # face
    ]

    def __init__(
        self,
        num_classes: int = 1,
        filters: Tuple[int, ...] = (),
        num_keypoints: int = 17
    ):
        # Initialize base but we'll override cls head
        nn.Module.__init__(self)
        
        self.ch = 16  # DFL channels (distribution bins)
        self.nc = num_classes
        self.nl = len(filters)  # Number of detection layers
        self.nk = num_keypoints  # Number of keypoints
        self.kpt_shape = (num_keypoints, 3)  # (x, y, visibility) per keypoint
        self.no = num_classes + self.ch * 4 + num_keypoints * 3  # Outputs per anchor
        self.stride = torch.zeros(self.nl)
        
        # Register empty tensors
        self.register_buffer('anchors', torch.empty(0))
        self.register_buffer('strides', torch.empty(0))

        # Compute head dimensions - matches Ultralytics
        box_ch = max(64, filters[0] // 4)
        # For pose (nc=1), cls_ch = max(nc, filters[0]) - SINGLE value for all scales
        cls_ch = max(num_classes, filters[0])
        # Keypoint channels = max(nk * 3, filters[0] // 4)
        # This gives: 51 for n/s (51 > 16, 51 > 32), 64 for m/l (51 < 64), 96 for x (51 < 96)
        kpt_ch = max(num_keypoints * 3, filters[0] // 4)

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
        
        # Classification heads (one per scale) - uses cls_ch for ALL scales
        self.cls = nn.ModuleList([
            nn.Sequential(
                Conv(f, f, nn.SiLU(), k=3, p=1, g=f),  # Depthwise
                Conv(f, cls_ch, nn.SiLU()),
                Conv(cls_ch, cls_ch, nn.SiLU(), k=3, p=1, g=cls_ch),  # Depthwise
                Conv(cls_ch, cls_ch, nn.SiLU()),
                nn.Conv2d(cls_ch, num_classes, kernel_size=1)
            ) for f in filters
        ])
        
        # Keypoint prediction heads (one per scale)
        # Ultralytics uses max(nk*3, filters[0]//4) as internal channels
        self.kpt = nn.ModuleList([
            nn.Sequential(
                Conv(f, kpt_ch, nn.SiLU(), k=3, p=1),
                Conv(kpt_ch, kpt_ch, nn.SiLU(), k=3, p=1),
                nn.Conv2d(kpt_ch, num_keypoints * 3, kernel_size=1)
            ) for f in filters
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass for pose estimation head.
        
        Args:
            features: List of feature maps from FPN
            
        Returns:
            During training: List of raw predictions per scale
            During inference: Concatenated predictions with decoded boxes and keypoints
        """
        outputs = []
        for i, (box, cls, kpt) in enumerate(zip(self.box, self.cls, self.kpt)):
            box_out = box(features[i])
            cls_out = cls(features[i])
            kpt_out = kpt(features[i])
            outputs.append(torch.cat([box_out, cls_out, kpt_out], dim=1))

        if self.training:
            return outputs

        # Inference mode
        self.anchors, self.strides = (
            t.transpose(0, 1) 
            for t in self._make_anchors(outputs, self.stride)
        )
        
        # Reshape and concatenate
        x = torch.cat([o.view(outputs[0].shape[0], self.no, -1) for o in outputs], dim=2)
        box, cls, kpts = x.split((4 * self.ch, self.nc, self.nk * 3), dim=1)

        # Decode boxes
        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), dim=1)
        
        # Decode keypoints
        kpts = self._decode_keypoints(kpts)

        return torch.cat((box * self.strides, cls.sigmoid(), kpts), dim=1)

    def _decode_keypoints(self, kpts: torch.Tensor) -> torch.Tensor:
        """
        Decode keypoint predictions relative to anchor points.
        
        Args:
            kpts: Raw keypoint predictions (B, nk*3, A)
            
        Returns:
            Decoded keypoints (B, nk*3, A)
        """
        b, _, a = kpts.shape
        kpts = kpts.view(b, self.nk, 3, a)
        
        # Decode x, y relative to anchors
        kpts[:, :, :2] = kpts[:, :, :2] * 2.0 + self.anchors.unsqueeze(0).unsqueeze(0)
        kpts[:, :, :2] = kpts[:, :, :2] * self.strides
        
        # Apply sigmoid to visibility
        kpts[:, :, 2:3] = kpts[:, :, 2:3].sigmoid()
        
        return kpts.view(b, self.nk * 3, a)

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
    def kpts_decode(
        kpts: torch.Tensor,
        img_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Scale keypoints to image coordinates.
        
        Args:
            kpts: Keypoints (N, nk, 3) with (x, y, visibility)
            img_shape: Target image shape (H, W)
            
        Returns:
            Scaled keypoints
        """
        kpts = kpts.clone()
        kpts[..., 0] *= img_shape[1]  # x
        kpts[..., 1] *= img_shape[0]  # y
        return kpts

    @staticmethod
    def kpts_to_coco_format(
        kpts: torch.Tensor,
        scores: torch.Tensor
    ) -> List[dict]:
        """
        Convert keypoints to COCO annotation format.
        
        Args:
            kpts: Keypoints (N, nk, 3)
            scores: Detection scores (N,)
            
        Returns:
            List of COCO-format annotations
        """
        annotations = []
        for i in range(kpts.shape[0]):
            keypoints = kpts[i].view(-1).tolist()
            annotations.append({
                'keypoints': keypoints,
                'score': float(scores[i]),
                'category_id': 1  # person
            })
        return annotations
