"""
Data augmentation utilities for YOLO v11.

This module provides various augmentation techniques:
- HSV color augmentation
- Random perspective transforms
- Mosaic augmentation
- MixUp augmentation
- Letterbox resizing
"""

import math
import random
from typing import Tuple, Optional, Dict, List

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    target_size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resize image with padding to target size while maintaining aspect ratio.
    
    Args:
        image: Input image (H, W, C)
        target_size: Target size (width and height)
        color: Padding color (B, G, R)
        auto: Auto-size padding to match stride
        scale_fill: Stretch image to fill target size
        scale_up: Allow scaling up
        stride: Stride for auto padding
        
    Returns:
        Tuple of:
            - Resized and padded image
            - Scale ratios (width_ratio, height_ratio)
            - Padding (pad_width, pad_height)
    """
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(target_size / shape[0], target_size / shape[1])
    if not scale_up:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = target_size - new_unpad[0]
    dh = target_size - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (target_size, target_size)
        r = target_size / shape[1], target_size / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=color)

    return image, (r, r), (dw, dh)


def augment_hsv(
    image: np.ndarray,
    hue_gain: float = 0.015,
    sat_gain: float = 0.7,
    val_gain: float = 0.4
) -> None:
    """
    Apply HSV color-space augmentation (in-place).
    
    Args:
        image: Input image (BGR)
        hue_gain: Hue augmentation range (fraction of 180)
        sat_gain: Saturation augmentation range
        val_gain: Value augmentation range
    """
    if hue_gain or sat_gain or val_gain:
        r = np.random.uniform(-1, 1, 3) * [hue_gain, sat_gain, val_gain] + 1
        
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)


def random_perspective(
    image: np.ndarray,
    labels: np.ndarray,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    border: Tuple[int, int] = (0, 0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random perspective transformation to image and labels.
    
    Args:
        image: Input image
        labels: Labels (N, 5) with (cls, x1, y1, x2, y2)
        degrees: Rotation range (degrees)
        translate: Translation range (fraction)
        scale: Scale range
        shear: Shear range (degrees)
        perspective: Perspective distortion (not implemented)
        border: Border for mosaic
        
    Returns:
        Tuple of (transformed image, transformed labels)
    """
    height = image.shape[0] + border[0] * 2
    width = image.shape[1] + border[1] * 2

    # Center translation
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2
    C[1, 2] = -image.shape[0] / 2

    # Rotation and scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined matrix
    M = T @ S @ R @ C
    
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        image = cv2.warpAffine(image, M[:2], dsize=(width, height), 
                               borderValue=(114, 114, 114))

    # Transform labels
    n = len(labels)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n, 8)

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # Clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # Filter candidates
        indices = _box_candidates(labels[:, 1:5].T * s, new.T)
        labels = labels[indices]
        labels[:, 1:5] = new[indices]

    return image, labels


def _box_candidates(
    box1: np.ndarray,
    box2: np.ndarray,
    wh_thr: float = 2,
    ar_thr: float = 100,
    area_thr: float = 0.1
) -> np.ndarray:
    """
    Filter box candidates based on size and aspect ratio.
    
    Args:
        box1: Original boxes (4, N)
        box2: Transformed boxes (4, N)
        wh_thr: Minimum width/height threshold
        ar_thr: Maximum aspect ratio threshold
        area_thr: Minimum area ratio threshold
        
    Returns:
        Boolean mask of valid boxes
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    
    return ((w2 > wh_thr) & (h2 > wh_thr) & 
            (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & 
            (ar < ar_thr))


def mosaic(
    images: List[np.ndarray],
    labels_list: List[np.ndarray],
    target_size: int = 640
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply mosaic augmentation combining 4 images.
    
    Args:
        images: List of 4 images
        labels_list: List of 4 label arrays
        target_size: Output image size
        
    Returns:
        Tuple of (mosaic image, combined labels)
    """
    assert len(images) == 4, "Mosaic requires exactly 4 images"
    
    s = target_size
    mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
    
    # Random center point
    xc = int(random.uniform(s * 0.5, s * 1.5))
    yc = int(random.uniform(s * 0.5, s * 1.5))
    
    labels4 = []
    
    for i, (img, labels) in enumerate(zip(images, labels_list)):
        h, w = img.shape[:2]
        
        # Place image in mosaic
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        
        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        pad_w = x1a - x1b
        pad_h = y1a - y1b
        
        # Update labels
        if len(labels):
            labels = labels.copy()
            labels[:, 1] = labels[:, 1] * w + pad_w  # x1
            labels[:, 2] = labels[:, 2] * h + pad_h  # y1
            labels[:, 3] = labels[:, 3] * w + pad_w  # x2
            labels[:, 4] = labels[:, 4] * h + pad_h  # y2
            labels4.append(labels)
    
    labels4 = np.concatenate(labels4, 0) if labels4 else np.zeros((0, 5))
    
    # Clip labels
    labels4[:, 1:5] = np.clip(labels4[:, 1:5], 0, s * 2)
    
    return mosaic_img, labels4


def mixup(
    image1: np.ndarray,
    labels1: np.ndarray,
    image2: np.ndarray,
    labels2: np.ndarray,
    alpha: float = 32.0,
    beta: float = 32.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply MixUp augmentation combining two images.
    
    Reference: https://arxiv.org/abs/1710.09412
    
    Args:
        image1: First image
        labels1: First image labels
        image2: Second image
        labels2: Second image labels
        alpha: Beta distribution alpha parameter
        beta: Beta distribution beta parameter
        
    Returns:
        Tuple of (mixed image, combined labels)
    """
    r = np.random.beta(alpha, beta)
    image = (image1 * r + image2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels1, labels2), 0)
    
    return image, labels


def flip_horizontal(
    image: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply horizontal flip to image and labels.
    
    Args:
        image: Input image
        labels: Labels (N, 5) with (cls, x1, y1, x2, y2) normalized
        
    Returns:
        Tuple of (flipped image, updated labels)
    """
    image = np.fliplr(image)
    
    if len(labels):
        labels = labels.copy()
        labels[:, 1] = 1 - labels[:, 1]  # x center
    
    return image, labels


def flip_vertical(
    image: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply vertical flip to image and labels.
    
    Args:
        image: Input image
        labels: Labels (N, 5) with (cls, x, y, w, h) normalized
        
    Returns:
        Tuple of (flipped image, updated labels)
    """
    image = np.flipud(image)
    
    if len(labels):
        labels = labels.copy()
        labels[:, 2] = 1 - labels[:, 2]  # y center
    
    return image, labels

