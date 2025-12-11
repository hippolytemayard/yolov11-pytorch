"""
YOLO v11 Data Handling.

This package provides:
- Dataset classes for detection, segmentation, and pose
- Data augmentation utilities
- Data loading helpers
"""

from yolo.data.dataset import (
    DetectionDataset,
    SegmentationDataset,
    PoseDataset,
)
from yolo.data.augmentations import (
    augment_hsv,
    random_perspective,
    mosaic,
    mixup,
    letterbox,
)

__all__ = [
    "DetectionDataset",
    "SegmentationDataset",
    "PoseDataset",
    "augment_hsv",
    "random_perspective",
    "mosaic",
    "mixup",
    "letterbox",
]

