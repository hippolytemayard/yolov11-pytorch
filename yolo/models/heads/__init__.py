"""
YOLO v11 Head Modules.

This package contains task-specific heads:
- DetectHead: Object detection
- SegmentHead: Instance segmentation
- PoseHead: Pose estimation
"""

from yolo.models.heads.detect import DetectHead, DFL
from yolo.models.heads.segment import SegmentHead
from yolo.models.heads.pose import PoseHead

__all__ = [
    "DetectHead",
    "DFL",
    "SegmentHead",
    "PoseHead",
]

