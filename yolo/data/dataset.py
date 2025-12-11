"""
Dataset classes for YOLO v11.

This module provides dataset implementations for:
- Object detection
- Instance segmentation
- Pose estimation

All datasets follow a common interface for easy integration.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from yolo.data.augmentations import (
    augment_hsv,
    letterbox,
    random_perspective,
    mosaic,
    mixup,
)

# Supported image formats
IMAGE_FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')


class BaseDataset(Dataset):
    """
    Base dataset class for YOLO training.
    
    Handles image loading, caching, and basic augmentation.
    
    Args:
        image_paths: List of image file paths
        input_size: Input image size for the model
        augment: Whether to apply augmentations
        cache: Whether to cache images in memory
    """
    
    def __init__(
        self,
        image_paths: List[str],
        input_size: int = 640,
        augment: bool = True,
        cache: bool = False
    ):
        self.image_paths = image_paths
        self.input_size = input_size
        self.augment = augment
        self.cache = cache
        
        self.mosaic = augment
        self.n = len(image_paths)
        self.indices = list(range(self.n))
        
        # Image cache
        self._cache = {} if cache else None

    def __len__(self) -> int:
        return self.n

    def load_image(self, index: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Load an image by index.
        
        Args:
            index: Image index
            
        Returns:
            Tuple of (image, original shape)
        """
        path = self.image_paths[index]
        
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        
        image = cv2.imread(path)
        assert image is not None, f"Failed to load image: {path}"
        
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        
        if r != 1:
            interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA
            image = cv2.resize(image, (int(w * r), int(h * r)), interpolation=interp)
        
        result = (image, (h, w))
        
        if self._cache is not None:
            self._cache[index] = result
        
        return result


class DetectionDataset(BaseDataset):
    """
    Dataset for object detection training.
    
    Expects YOLO-format labels (class x_center y_center width height).
    
    Args:
        image_paths: List of image file paths
        input_size: Input image size
        params: Augmentation parameters
        augment: Whether to apply augmentations
    """
    
    def __init__(
        self,
        image_paths: List[str],
        input_size: int = 640,
        params: Optional[Dict] = None,
        augment: bool = True
    ):
        super().__init__(image_paths, input_size, augment)
        
        self.params = params or self._default_params()
        self.labels = self._load_labels()

    def _default_params(self) -> Dict:
        """Default augmentation parameters."""
        return {
            'mosaic': 1.0,
            'mixup': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'flip_lr': 0.5,
            'flip_ud': 0.0,
        }

    def _load_labels(self) -> Dict[str, np.ndarray]:
        """Load labels for all images."""
        labels = {}
        
        for path in self.image_paths:
            label_path = self._get_label_path(path)
            
            if os.path.exists(label_path):
                with open(label_path) as f:
                    data = [x.split() for x in f.read().strip().splitlines()]
                    if data:
                        label = np.array(data, dtype=np.float32)
                    else:
                        label = np.zeros((0, 5), dtype=np.float32)
            else:
                label = np.zeros((0, 5), dtype=np.float32)
            
            labels[path] = label
        
        return labels

    def _get_label_path(self, image_path: str) -> str:
        """Get label path from image path."""
        return image_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Tuple of (image, classes, boxes, indices)
        """
        index = self.indices[index]
        
        if self.mosaic and random.random() < self.params['mosaic']:
            image, labels = self._load_mosaic(index)
            
            if random.random() < self.params['mixup']:
                index2 = random.choice(self.indices)
                image2, labels2 = self._load_mosaic(index2)
                image, labels = mixup(image, labels, image2, labels2)
        else:
            image, shape = self.load_image(index)
            h, w = image.shape[:2]
            
            image, ratio, pad = letterbox(image, self.input_size, scale_up=self.augment)
            
            labels = self.labels[self.image_paths[index]].copy()
            if labels.size:
                labels[:, 1:] = self._wh2xy(labels[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            
            if self.augment:
                image, labels = random_perspective(
                    image, labels,
                    degrees=self.params['degrees'],
                    translate=self.params['translate'],
                    scale=self.params['scale'],
                    shear=self.params['shear']
                )

        h, w = image.shape[:2]
        nl = len(labels)
        
        if nl:
            cls = labels[:, 0:1]
            box = labels[:, 1:5]
            box = self._xy2wh(box, w, h)
        else:
            cls = np.zeros((0, 1))
            box = np.zeros((0, 4))

        if self.augment:
            augment_hsv(image, self.params['hsv_h'], self.params['hsv_s'], self.params['hsv_v'])
            
            if random.random() < self.params['flip_ud']:
                image = np.flipud(image)
                if nl:
                    box[:, 1] = 1 - box[:, 1]
            
            if random.random() < self.params['flip_lr']:
                image = np.fliplr(image)
                if nl:
                    box[:, 0] = 1 - box[:, 0]

        # Convert to tensors
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)
        
        return (
            torch.from_numpy(image),
            torch.from_numpy(cls) if nl else torch.zeros((0, 1)),
            torch.from_numpy(box) if nl else torch.zeros((0, 4)),
            torch.zeros(nl)
        )

    def _load_mosaic(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load mosaic augmented sample."""
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)
        
        images = []
        labels_list = []
        
        for i in indices:
            img, _ = self.load_image(i)
            images.append(img)
            labels_list.append(self.labels[self.image_paths[i]].copy())
        
        image, labels = mosaic(images, labels_list, self.input_size)
        
        image, labels = random_perspective(
            image, labels,
            degrees=self.params['degrees'],
            translate=self.params['translate'],
            scale=self.params['scale'],
            border=(-self.input_size // 2, -self.input_size // 2)
        )
        
        return image, labels

    @staticmethod
    def _wh2xy(x: np.ndarray, w: float, h: float, pad_w: float, pad_h: float) -> np.ndarray:
        """Convert from normalized xywh to pixel xyxy."""
        y = np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h
        return y

    @staticmethod
    def _xy2wh(x: np.ndarray, w: int, h: int) -> np.ndarray:
        """Convert from pixel xyxy to normalized xywh."""
        x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1e-3)
        x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1e-3)
        
        y = np.copy(x)
        y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
        y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
        y[:, 2] = (x[:, 2] - x[:, 0]) / w
        y[:, 3] = (x[:, 3] - x[:, 1]) / h
        return y

    @staticmethod
    def collate_fn(batch: List) -> Tuple[torch.Tensor, Dict]:
        """Collate function for DataLoader."""
        samples, cls, box, indices = zip(*batch)
        
        cls = torch.cat(cls, dim=0)
        box = torch.cat(box, dim=0)
        
        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] = indices[i] + i
        indices = torch.cat(new_indices, dim=0)
        
        return torch.stack(samples, dim=0), {'cls': cls, 'box': box, 'idx': indices}


class SegmentationDataset(DetectionDataset):
    """
    Dataset for instance segmentation training.
    
    Extends DetectionDataset with mask loading.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        input_size: int = 640,
        params: Optional[Dict] = None,
        augment: bool = True
    ):
        super().__init__(image_paths, input_size, params, augment)
        self.masks = self._load_masks()

    def _load_masks(self) -> Dict[str, np.ndarray]:
        """Load segmentation masks for all images."""
        masks = {}
        # Mask loading implementation would go here
        # Format: polygon coordinates per instance
        for path in self.image_paths:
            masks[path] = None  # Placeholder
        return masks

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        """Get a segmentation training sample."""
        image, cls, box, idx = super().__getitem__(index)
        
        # Add mask data
        masks = self.masks.get(self.image_paths[self.indices[index]])
        
        return image, {
            'cls': cls,
            'box': box,
            'idx': idx,
            'masks': masks
        }


class PoseDataset(DetectionDataset):
    """
    Dataset for pose estimation training.
    
    Extends DetectionDataset with keypoint loading.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        input_size: int = 640,
        params: Optional[Dict] = None,
        augment: bool = True,
        num_keypoints: int = 17
    ):
        self.num_keypoints = num_keypoints
        super().__init__(image_paths, input_size, params, augment)

    def _load_labels(self) -> Dict[str, np.ndarray]:
        """Load labels with keypoints for all images."""
        labels = {}
        
        for path in self.image_paths:
            label_path = self._get_label_path(path)
            
            if os.path.exists(label_path):
                with open(label_path) as f:
                    data = [x.split() for x in f.read().strip().splitlines()]
                    if data:
                        # Format: class x y w h kpt1_x kpt1_y kpt1_v ... kptN_x kptN_y kptN_v
                        label = np.array(data, dtype=np.float32)
                    else:
                        label = np.zeros((0, 5 + self.num_keypoints * 3), dtype=np.float32)
            else:
                label = np.zeros((0, 5 + self.num_keypoints * 3), dtype=np.float32)
            
            labels[path] = label
        
        return labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        """Get a pose estimation training sample."""
        image, cls, box, idx = DetectionDataset.__getitem__(self, index)
        
        # Extract keypoints from labels
        labels = self.labels[self.image_paths[self.indices[index]]]
        
        if len(labels) and labels.shape[1] > 5:
            keypoints = labels[:, 5:].reshape(-1, self.num_keypoints, 3)
        else:
            keypoints = np.zeros((0, self.num_keypoints, 3))
        
        return image, {
            'cls': cls,
            'box': box,
            'idx': idx,
            'keypoints': torch.from_numpy(keypoints.astype(np.float32))
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for YOLO training.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    collate_fn = getattr(dataset, 'collate_fn', None)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

