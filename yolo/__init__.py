"""YOLO v11 Open Source - Pure PyTorch implementation of YOLOv11."""

__version__ = "0.1.0"
__author__ = "Hippolyte Mayard"

from yolo.models import (
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

