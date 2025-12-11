"""
Evaluation metrics for YOLO v11.

This module provides:
- Average Precision (AP) computation
- mAP calculation at various IoU thresholds
- Precision/Recall curves
"""

from typing import Tuple, Optional

import numpy as np
import torch

from yolo.utils.boxes import compute_iou


def compute_metric(
    output: torch.Tensor,
    target: torch.Tensor,
    iou_thresholds: torch.Tensor
) -> torch.Tensor:
    """
    Compute matching between predictions and ground truth at various IoU thresholds.
    
    Args:
        output: Predictions (N, 6) with (x1, y1, x2, y2, conf, cls)
        target: Ground truth (M, 5) with (cls, x1, y1, x2, y2)
        iou_thresholds: IoU thresholds to evaluate (T,)
        
    Returns:
        Boolean tensor (N, T) indicating correct predictions
    """
    # Compute pairwise IoU
    a1, a2 = target[:, 1:].unsqueeze(1).chunk(2, dim=2)
    b1, b2 = output[:, :4].unsqueeze(0).chunk(2, dim=2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = np.zeros((output.shape[0], iou_thresholds.shape[0]), dtype=bool)
    
    for i, thresh in enumerate(iou_thresholds):
        # Find matches (IoU > threshold and class matches)
        matches = torch.where((iou >= thresh) & (target[:, 0:1] == output[:, 5]))
        
        if matches[0].shape[0]:
            # Stack matches: [gt_idx, pred_idx, iou]
            match_data = torch.cat([
                torch.stack(matches, 1),
                iou[matches[0], matches[1]][:, None]
            ], 1).cpu().numpy()
            
            if len(match_data) > 1:
                # Sort by IoU (descending)
                match_data = match_data[match_data[:, 2].argsort()[::-1]]
                # Keep unique predictions (best match per prediction)
                match_data = match_data[np.unique(match_data[:, 1], return_index=True)[1]]
                # Keep unique ground truths (best match per GT)
                match_data = match_data[np.unique(match_data[:, 0], return_index=True)[1]]
            
            correct[match_data[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def smooth(y: np.ndarray, f: float = 0.1) -> np.ndarray:
    """Apply box filter smoothing to array."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


def compute_ap(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Compute Average Precision for all classes.
    
    Based on: https://github.com/rafaelpadilla/Object-Detection-Metrics
    
    Args:
        tp: True positives (N, num_iou_thresholds)
        conf: Confidence scores (N,)
        pred_cls: Predicted classes (N,)
        target_cls: Ground truth classes (M,)
        eps: Small value for numerical stability
        
    Returns:
        Tuple of:
            - True positives (C,)
            - False positives (C,)
            - Mean precision
            - Mean recall
            - mAP@0.5
            - mAP@0.5:0.95
    """
    # Sort by confidence (descending)
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    # Initialize arrays
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))
    ap = np.zeros((nc, tp.shape[1]))
    px = np.linspace(0, 1, 1000)

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of predictions
        
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        # AP for each IoU threshold
        for j in range(tp.shape[1]):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute precision envelope
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))

            # Integrate (101-point interpolation for COCO)
            x = np.linspace(0, 1, 101)
            ap[ci, j] = np.trapz(np.interp(x, m_rec, m_pre), x)

    # Compute F1 score
    f1 = 2 * p * r / (p + r + eps)

    # Find max F1 point
    i = smooth(f1.mean(0)).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    
    tp_sum = (r * nt).round()
    fp_sum = (tp_sum / (p + eps) - tp_sum).round()
    
    ap50, mean_ap = ap[:, 0], ap.mean(1)
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap_total = ap50.mean(), mean_ap.mean()

    return tp_sum, fp_sum, m_pre, m_rec, map50, mean_ap_total


class MetricTracker:
    """
    Track and compute metrics during training/evaluation.
    
    Attributes:
        iou_thresholds: IoU thresholds for mAP computation
    """
    
    def __init__(self, num_classes: int = 80):
        self.num_classes = num_classes
        self.iou_thresholds = torch.linspace(0.5, 0.95, 10)
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.metrics = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> None:
        """
        Update metrics with batch predictions.
        
        Args:
            predictions: Predictions (N, 6) with (x1, y1, x2, y2, conf, cls)
            targets: Ground truth (M, 5) with (cls, x1, y1, x2, y2)
        """
        if predictions.shape[0] == 0:
            if targets.shape[0]:
                self.metrics.append((
                    torch.zeros((0, self.iou_thresholds.numel()), dtype=torch.bool),
                    torch.zeros(0),
                    torch.zeros(0),
                    targets[:, 0]
                ))
            return

        if targets.shape[0]:
            metric = compute_metric(
                predictions,
                targets,
                self.iou_thresholds.to(predictions.device)
            )
            self.metrics.append((
                metric,
                predictions[:, 4],
                predictions[:, 5],
                targets[:, 0]
            ))
        else:
            self.metrics.append((
                torch.zeros((predictions.shape[0], self.iou_thresholds.numel()), dtype=torch.bool),
                predictions[:, 4],
                predictions[:, 5],
                torch.zeros(0)
            ))

    def compute(self) -> Tuple[float, float, float, float]:
        """
        Compute final metrics.
        
        Returns:
            Tuple of (precision, recall, mAP@50, mAP@50:95)
        """
        if not self.metrics:
            return 0, 0, 0, 0

        metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*self.metrics)]
        
        if len(metrics) and metrics[0].any():
            _, _, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)
            return m_pre, m_rec, map50, mean_ap
        
        return 0, 0, 0, 0

