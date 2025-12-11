"""
YOLO v11 Model Architecture Components.

This module contains all the neural network building blocks:
- Backbone: DarkNet with CSP connections
- Neck: Feature Pyramid Network (FPN)
- Heads: Detection, Segmentation, Pose
"""

from yolo.models.backbone import DarkNet
from yolo.models.neck import DarkFPN
from yolo.models.yolo import (
    YOLO,
    YOLODetect,
    YOLOSegment,
    YOLOPose,
    yolo_v11_n,
    yolo_v11_s,
    yolo_v11_m,
    yolo_v11_l,
    yolo_v11_x,
)

__all__ = [
    "DarkNet",
    "DarkFPN",
    "YOLO",
    "YOLODetect",
    "YOLOSegment",
    "YOLOPose",
    "yolo_v11_n",
    "yolo_v11_s",
    "yolo_v11_m",
    "yolo_v11_l",
    "yolo_v11_x",
]

