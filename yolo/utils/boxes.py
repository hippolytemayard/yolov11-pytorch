"""
Bounding box utilities for YOLO v11.

This module provides functions for:
- Coordinate transformations (wh2xy, xy2wh)
- Anchor generation
- IoU computation
- Non-maximum suppression
"""

import math
from time import time
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchvision


def wh2xy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from center format to corner format.
    
    Args:
        x: Boxes in (cx, cy, w, h) format
        
    Returns:
        Boxes in (x1, y1, x2, y2) format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def xy2wh(x: torch.Tensor, w: int, h: int, clip: bool = True) -> torch.Tensor:
    """
    Convert bounding boxes from corner format to center format (normalized).
    
    Args:
        x: Boxes in (x1, y1, x2, y2) format
        w: Image width for normalization
        h: Image height for normalization
        clip: Whether to clip coordinates to image bounds
        
    Returns:
        Boxes in normalized (cx, cy, w, h) format
    """
    if clip:
        x[..., [0, 2]] = x[..., [0, 2]].clip(0, w - 1e-3)
        x[..., [1, 3]] = x[..., [1, 3]].clip(0, h - 1e-3)
    
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # center x
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # center y
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def make_anchors(
    features: List[torch.Tensor],
    strides: torch.Tensor,
    offset: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor points for all feature levels.
    
    Creates a grid of anchor points centered on each feature map cell.
    
    Args:
        features: List of feature maps from different scales
        strides: Stride for each feature level
        offset: Offset from cell corner (0.5 = center)
        
    Returns:
        Tuple of (anchor_points, stride_tensor)
    """
    assert features is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = features[0].dtype, features[0].device
    
    for i, stride in enumerate(strides):
        _, _, h, w = features[i].shape
        
        # Create grid
        sx = torch.arange(w, device=device, dtype=dtype) + offset
        sy = torch.arange(h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        
        anchor_tensor.append(torch.stack((sx, sy), dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)


def compute_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute Complete IoU (CIoU) between two sets of boxes.
    
    CIoU considers:
    - Overlap area (IoU)
    - Center distance
    - Aspect ratio consistency
    
    Reference: https://arxiv.org/abs/1911.08287
    
    Args:
        box1: First set of boxes (N, 4) in xyxy format
        box2: Second set of boxes (M, 4) in xyxy format
        eps: Small value for numerical stability
        
    Returns:
        CIoU values (N, M) or (N,) if shapes are aligned
    """
    # Extract coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)
    
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    
    # CIoU components
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + 
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
    
    # Aspect ratio term
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    
    return iou - (rho2 / c2 + v * alpha)


def non_max_suppression(
    outputs: torch.Tensor,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.65,
    max_det: int = 300,
    num_classes: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Perform Non-Maximum Suppression (NMS) on detection predictions.
    
    Args:
        outputs: Raw predictions (B, C, A) where C = 4 + num_classes
        conf_threshold: Confidence threshold for filtering
        iou_threshold: IoU threshold for NMS
        max_det: Maximum detections per image
        num_classes: Number of classes (inferred if None)
        
    Returns:
        List of detections per image, each (N, 6) with (x1, y1, x2, y2, conf, cls)
    """
    max_wh = 7680  # Maximum box dimension
    max_nms = 30000  # Maximum boxes for NMS

    bs = outputs.shape[0]
    nc = num_classes or (outputs.shape[1] - 4)
    xc = outputs[:, 4:4 + nc].amax(1) > conf_threshold

    start = time()
    limit = 0.5 + 0.05 * bs  # Time limit
    
    output = [torch.zeros((0, 6), device=outputs.device)] * bs
    
    for idx, x in enumerate(outputs):
        x = x.transpose(0, -1)[xc[idx]]

        if not x.shape[0]:
            continue

        # Separate box and class predictions
        box, cls = x.split((4, nc), dim=1)
        box = wh2xy(box)
        
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), dim=1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), dim=1)[conf.view(-1) > conf_threshold]

        n = x.shape[0]
        if not n:
            continue
            
        # Sort by confidence
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        indices = indices[:max_det]

        output[idx] = x[indices]
        
        if (time() - start) > limit:
            break

    return output


def scale_boxes(
    boxes: torch.Tensor,
    from_shape: Tuple[int, int],
    to_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple[float, Tuple[float, float]]] = None
) -> torch.Tensor:
    """
    Scale bounding boxes from one image size to another.
    
    Args:
        boxes: Boxes to scale (N, 4) in xyxy format
        from_shape: Source shape (H, W)
        to_shape: Target shape (H, W)
        ratio_pad: Optional (ratio, (pad_w, pad_h)) from letterbox
        
    Returns:
        Scaled boxes
    """
    if ratio_pad is None:
        gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
        pad = ((from_shape[1] - to_shape[1] * gain) / 2,
               (from_shape[0] - to_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    boxes = boxes.clone()
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    
    # Clip to image bounds
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, to_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, to_shape[0])
    
    return boxes

