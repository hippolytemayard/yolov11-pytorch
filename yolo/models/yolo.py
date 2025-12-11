"""
YOLO v11 Unified Model Implementation.

This module provides the complete YOLO v11 model with support for:
- Object Detection (YOLODetect)
- Instance Segmentation (YOLOSegment)
- Pose Estimation (YOLOPose)

All models share the same backbone and neck architecture.
"""

from typing import List, Tuple, Optional, Union, Dict

import torch
import torch.nn as nn

from yolo.models.backbone import DarkNet
from yolo.models.neck import DarkFPN
from yolo.models.heads import DetectHead, SegmentHead, PoseHead
from yolo.models.common import Conv, fuse_conv_bn


# Model configurations
MODEL_CONFIGS = {
    'n': {  # Nano
        'width': [3, 16, 32, 64, 128, 256],
        'depth': [1, 1, 1, 1, 1, 1],
        'csp': [False, True]
    },
    't': {  # Tiny
        'width': [3, 24, 48, 96, 192, 384],
        'depth': [1, 1, 1, 1, 1, 1],
        'csp': [False, True]
    },
    's': {  # Small
        'width': [3, 32, 64, 128, 256, 512],
        'depth': [1, 1, 1, 1, 1, 1],
        'csp': [False, True]
    },
    'm': {  # Medium
        'width': [3, 64, 128, 256, 512, 512],
        'depth': [1, 1, 1, 1, 1, 1],
        'csp': [True, True]
    },
    'l': {  # Large
        'width': [3, 64, 128, 256, 512, 512],
        'depth': [2, 2, 2, 2, 2, 2],
        'csp': [True, True]
    },
    'x': {  # Extra Large
        'width': [3, 96, 192, 384, 768, 768],
        'depth': [2, 2, 2, 2, 2, 2],
        'csp': [True, True]
    }
}


class YOLO(nn.Module):
    """
    Base YOLO v11 model for object detection.
    
    This model combines the DarkNet backbone, FPN neck, and detection head
    to perform object detection at multiple scales.
    
    Args:
        width: List of channel widths for each stage
        depth: List of depths (number of blocks) for each stage
        csp: List of booleans indicating CSP module type
        num_classes: Number of object classes
    """
    
    def __init__(
        self,
        width: List[int],
        depth: List[int],
        csp: List[bool],
        num_classes: int
    ):
        super().__init__()
        
        # Backbone and neck
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)
        
        # Detection head
        filters = (width[3], width[4], width[5])
        self.head = DetectHead(num_classes, filters)
        
        # Initialize strides
        with torch.no_grad():
            dummy = torch.zeros(1, width[0], 256, 256)
            outputs = self.forward(dummy)
            self.head.stride = torch.tensor([256 / x.shape[-2] for x in outputs])
            self.stride = self.head.stride
            self.head.initialize_biases()

    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            During training: List of predictions per scale
            During inference: Concatenated predictions
        """
        features = self.net(x)
        features = self.fpn(features)
        return self.head(list(features))

    def fuse(self) -> 'YOLO':
        """
        Fuse Conv2d and BatchNorm2d layers for faster inference.
        
        This optimization merges consecutive conv-bn pairs into single
        conv layers, reducing memory access and computation.
        
        Returns:
            Self with fused layers
        """
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'norm'):
                m.conv = fuse_conv_bn(m.conv, m.norm)
                m.forward = m.forward_fused
                delattr(m, 'norm')
        return self

    @property
    def num_classes(self) -> int:
        """Number of object classes."""
        return self.head.nc

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class YOLODetect(YOLO):
    """
    YOLO v11 for Object Detection.
    
    Alias for the base YOLO class for clarity.
    """
    pass


class YOLOSegment(nn.Module):
    """
    YOLO v11 for Instance Segmentation.
    
    Extends YOLO with mask prediction capabilities for instance-level
    segmentation of objects.
    
    Args:
        width: List of channel widths for each stage
        depth: List of depths (number of blocks) for each stage
        csp: List of booleans indicating CSP module type
        num_classes: Number of object classes
        num_masks: Number of mask prototypes
    """
    
    def __init__(
        self,
        width: List[int],
        depth: List[int],
        csp: List[bool],
        num_classes: int,
        num_masks: int = 32
    ):
        super().__init__()
        
        # Backbone and neck
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)
        
        # Segmentation head
        filters = (width[3], width[4], width[5])
        self.head = SegmentHead(num_classes, filters, num_masks)
        
        # Initialize strides
        with torch.no_grad():
            dummy = torch.zeros(1, width[0], 256, 256)
            features = self.fpn(self.net(dummy))
            outputs = [torch.zeros(1, self.head.no, f.shape[-2], f.shape[-1]) for f in features]
            self.head.stride = torch.tensor([256 / x.shape[-2] for x in outputs])
            self.stride = self.head.stride
            self.head.initialize_biases()

    def forward(
        self,
        x: torch.Tensor
    ) -> Union[Tuple[List[torch.Tensor], torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of (predictions, prototypes)
        """
        features = self.net(x)
        features = self.fpn(features)
        return self.head(list(features))

    def fuse(self) -> 'YOLOSegment':
        """Fuse Conv2d and BatchNorm2d layers."""
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'norm'):
                m.conv = fuse_conv_bn(m.conv, m.norm)
                m.forward = m.forward_fused
                delattr(m, 'norm')
        return self


class YOLOPose(nn.Module):
    """
    YOLO v11 for Pose Estimation.
    
    Extends YOLO with keypoint prediction for human pose estimation.
    
    Args:
        width: List of channel widths for each stage
        depth: List of depths (number of blocks) for each stage
        csp: List of booleans indicating CSP module type
        num_classes: Number of object classes (typically 1 for person)
        num_keypoints: Number of keypoints to predict
    """
    
    def __init__(
        self,
        width: List[int],
        depth: List[int],
        csp: List[bool],
        num_classes: int = 1,
        num_keypoints: int = 17
    ):
        super().__init__()
        
        # Backbone and neck
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)
        
        # Pose head
        filters = (width[3], width[4], width[5])
        self.head = PoseHead(num_classes, filters, num_keypoints)
        
        # Initialize strides
        with torch.no_grad():
            dummy = torch.zeros(1, width[0], 256, 256)
            features = self.fpn(self.net(dummy))
            outputs = [torch.zeros(1, self.head.no, f.shape[-2], f.shape[-1]) for f in features]
            self.head.stride = torch.tensor([256 / x.shape[-2] for x in outputs])
            self.stride = self.head.stride
            self.head.initialize_biases()

    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            During training: List of predictions per scale
            During inference: Concatenated predictions with keypoints
        """
        features = self.net(x)
        features = self.fpn(features)
        return self.head(list(features))

    def fuse(self) -> 'YOLOPose':
        """Fuse Conv2d and BatchNorm2d layers."""
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'norm'):
                m.conv = fuse_conv_bn(m.conv, m.norm)
                m.forward = m.forward_fused
                delattr(m, 'norm')
        return self


# Factory functions for different model sizes
def yolo_v11_n(num_classes: int = 80, task: str = 'detect') -> Union[YOLO, YOLOSegment, YOLOPose]:
    """Create YOLOv11-Nano model."""
    config = MODEL_CONFIGS['n']
    return _create_model(config, num_classes, task)


def yolo_v11_t(num_classes: int = 80, task: str = 'detect') -> Union[YOLO, YOLOSegment, YOLOPose]:
    """Create YOLOv11-Tiny model."""
    config = MODEL_CONFIGS['t']
    return _create_model(config, num_classes, task)


def yolo_v11_s(num_classes: int = 80, task: str = 'detect') -> Union[YOLO, YOLOSegment, YOLOPose]:
    """Create YOLOv11-Small model."""
    config = MODEL_CONFIGS['s']
    return _create_model(config, num_classes, task)


def yolo_v11_m(num_classes: int = 80, task: str = 'detect') -> Union[YOLO, YOLOSegment, YOLOPose]:
    """Create YOLOv11-Medium model."""
    config = MODEL_CONFIGS['m']
    return _create_model(config, num_classes, task)


def yolo_v11_l(num_classes: int = 80, task: str = 'detect') -> Union[YOLO, YOLOSegment, YOLOPose]:
    """Create YOLOv11-Large model."""
    config = MODEL_CONFIGS['l']
    return _create_model(config, num_classes, task)


def yolo_v11_x(num_classes: int = 80, task: str = 'detect') -> Union[YOLO, YOLOSegment, YOLOPose]:
    """Create YOLOv11-XLarge model."""
    config = MODEL_CONFIGS['x']
    return _create_model(config, num_classes, task)


def _create_model(
    config: Dict,
    num_classes: int,
    task: str
) -> Union[YOLO, YOLOSegment, YOLOPose]:
    """Create a YOLO model based on configuration and task."""
    width = config['width']
    depth = config['depth']
    csp = config['csp']
    
    if task == 'detect':
        return YOLO(width, depth, csp, num_classes)
    elif task == 'segment':
        return YOLOSegment(width, depth, csp, num_classes)
    elif task == 'pose':
        return YOLOPose(width, depth, csp, num_classes=1)
    else:
        raise ValueError(f"Unknown task: {task}. Choose from 'detect', 'segment', 'pose'")

