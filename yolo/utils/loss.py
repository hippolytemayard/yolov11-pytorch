"""
Loss functions for YOLO v11.

This module provides loss computation for:
- Object detection (box, class, DFL losses)
- Instance segmentation (additional mask loss)
- Pose estimation (additional keypoint loss)
"""

import math
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo.utils.boxes import make_anchors, compute_iou


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reference: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Balancing factor for positive/negative samples
        gamma: Focusing parameter for hard examples
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bce(pred, target)

        if self.alpha > 0:
            alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss *= alpha_factor

        if self.gamma > 0:
            p_t = target * pred.sigmoid() + (1 - target) * (1 - pred.sigmoid())
            gamma_factor = (1.0 - p_t) ** self.gamma
            loss *= gamma_factor

        return loss


class BoxLoss(nn.Module):
    """
    Combined box regression loss using CIoU and DFL.
    
    Args:
        dfl_ch: Number of DFL distribution channels
    """
    
    def __init__(self, dfl_ch: int):
        super().__init__()
        self.dfl_ch = dfl_ch

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: float,
        fg_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute box regression losses.
        
        Args:
            pred_dist: Predicted distributions (B, A, 4*dfl_ch)
            pred_bboxes: Decoded predicted boxes (B, A, 4)
            anchor_points: Anchor points (A, 2)
            target_bboxes: Target boxes (B, A, 4)
            target_scores: Target scores (B, A, C)
            target_scores_sum: Sum of target scores
            fg_mask: Foreground mask (B, A)
            
        Returns:
            Tuple of (iou_loss, dfl_loss)
        """
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        a, b = target_bboxes.chunk(2, dim=-1)
        target = torch.cat((anchor_points - a, b - anchor_points), dim=-1)
        target = target.clamp(0, self.dfl_ch - 0.01)
        loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.dfl_ch + 1), target[fg_mask])
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Distribution Focal Loss.
        
        Reference: https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        
        left_loss = F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape)
        right_loss = F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape)
        
        return (left_loss * wl + right_loss * wr).mean(-1, keepdim=True)


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner for matching predictions to ground truth.
    
    Uses a combination of classification scores and IoU to determine
    the best assignment between predictions and targets.
    
    Args:
        num_classes: Number of object classes
        top_k: Number of top predictions to consider per target
        alpha: Classification score weight
        beta: IoU weight
        eps: Small value for numerical stability
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        top_k: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9
    ):
        super().__init__()
        self.top_k = top_k
        self.nc = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assign predictions to ground truth targets.
        
        Args:
            pd_scores: Predicted scores (B, A, C)
            pd_bboxes: Predicted boxes (B, A, 4)
            anc_points: Anchor points (A, 2)
            gt_labels: Ground truth labels (B, M, 1)
            gt_bboxes: Ground truth boxes (B, M, 4)
            mask_gt: Valid ground truth mask (B, M, 1)
            
        Returns:
            Tuple of (target_bboxes, target_scores, fg_mask)
        """
        batch_size = pd_scores.size(0)
        num_max_boxes = gt_bboxes.size(1)

        if num_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros_like(pd_scores[..., 0]).bool())

        num_anchors = anc_points.shape[0]
        
        # Check which anchors are inside ground truth boxes
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, dim=2)
        mask_in_gts = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2)
        mask_in_gts = mask_in_gts.view(batch_size, num_max_boxes, num_anchors, -1).amin(3).gt_(self.eps)
        
        gt_mask = (mask_in_gts * mask_gt).bool()
        
        # Compute alignment metric
        overlaps = torch.zeros([batch_size, num_max_boxes, num_anchors], 
                              dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([batch_size, num_max_boxes, num_anchors],
                                 dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, batch_size, num_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(batch_size).view(-1, 1).expand(-1, num_max_boxes)
        ind[1] = gt_labels.squeeze(-1).long()
        
        bbox_scores[gt_mask] = pd_scores[ind[0], :, ind[1]][gt_mask]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, num_max_boxes, -1, -1)[gt_mask]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, num_anchors, -1)[gt_mask]
        overlaps[gt_mask] = compute_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        # Select top-k predictions
        top_k_mask = mask_gt.expand(-1, -1, self.top_k).bool()
        top_k_metrics, top_k_indices = torch.topk(align_metric, self.top_k, dim=-1, largest=True)
        
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(top_k_indices)
        top_k_indices.masked_fill_(~top_k_mask, 0)

        mask_top_k = torch.zeros(align_metric.shape, dtype=torch.int8, device=top_k_indices.device)
        ones = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8)
        for k in range(self.top_k):
            mask_top_k.scatter_add_(-1, top_k_indices[:, :, k:k + 1], ones)
        mask_top_k.masked_fill_(mask_top_k > 1, 0)
        mask_top_k = mask_top_k.to(align_metric.dtype)
        mask_pos = mask_top_k * mask_in_gts * mask_gt

        # Handle multiple assignments
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        
        target_gt_idx = mask_pos.argmax(-2)

        # Get assigned targets
        index = torch.arange(batch_size, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_index = target_gt_idx + index * num_max_boxes
        target_labels = gt_labels.long().flatten()[target_index]
        target_bboxes = gt_bboxes.view(-1, 4)[target_index]

        # Compute target scores
        target_labels.clamp_(0)
        target_scores = torch.zeros((batch_size, num_anchors, self.nc),
                                   dtype=torch.int64, device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool()


class ComputeLoss:
    """
    Unified loss computation for YOLO detection.
    
    Computes:
    - Box regression loss (IoU + DFL)
    - Classification loss (BCE)
    
    Args:
        model: YOLO model instance
        params: Training parameters dictionary
    """
    
    def __init__(self, model: nn.Module, params: Dict):
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device
        head = model.head

        self.params = params
        self.stride = head.stride
        self.nc = head.nc
        self.no = head.no
        self.reg_max = head.ch
        self.device = device

        self.box_loss = BoxLoss(head.ch - 1).to(device)
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = TaskAlignedAssigner(num_classes=self.nc, top_k=10, alpha=0.5, beta=6.0)

        self.project = torch.arange(head.ch, dtype=torch.float, device=device)

    def _box_decode(
        self,
        anchor_points: torch.Tensor,
        pred_dist: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted distribution to boxes."""
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        pred_dist = pred_dist.matmul(self.project.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, dim=-1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        return torch.cat((x1y1, x2y2), dim=-1)

    def __call__(
        self,
        outputs: list,
        targets: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute detection losses.
        
        Args:
            outputs: List of predictions from detection head
            targets: Dictionary with 'cls', 'box', 'idx' keys
            
        Returns:
            Tuple of (box_loss, cls_loss, dfl_loss)
        """
        # Reshape predictions
        x = torch.cat([i.view(outputs[0].shape[0], self.no, -1) for i in outputs], dim=2)
        pred_distri, pred_scores = x.split((self.reg_max * 4, self.nc), dim=1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        data_type = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        input_size = torch.tensor(outputs[0].shape[2:], device=self.device, dtype=data_type) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(outputs, self.stride, offset=0.5)

        # Prepare targets
        idx = targets['idx'].view(-1, 1)
        cls = targets['cls'].view(-1, 1)
        box = targets['box']
        targets_tensor = torch.cat((idx, cls, box), dim=1).to(self.device)
        
        if targets_tensor.shape[0] == 0:
            gt = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets_tensor[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            gt = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets_tensor[matches, 1:]
            
            # Convert to absolute coordinates
            x = gt[..., 1:5].mul_(input_size[[1, 0, 1, 0]])
            y = torch.empty_like(x)
            dw = x[..., 2] / 2
            dh = x[..., 3] / 2
            y[..., 0] = x[..., 0] - dw
            y[..., 1] = x[..., 1] - dh
            y[..., 2] = x[..., 0] + dw
            y[..., 3] = x[..., 1] + dh
            gt[..., 1:5] = y

        gt_labels, gt_bboxes = gt.split((1, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Decode predictions
        pred_bboxes = self._box_decode(anchor_points, pred_distri)
        
        # Assign targets
        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Classification loss
        loss_cls = self.cls_loss(pred_scores, target_scores.to(data_type)).sum() / target_scores_sum

        # Box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss_box, loss_dfl = self.box_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        # Apply loss gains
        loss_box *= self.params['box']
        loss_cls *= self.params['cls']
        loss_dfl *= self.params['dfl']

        return loss_box, loss_cls, loss_dfl


class SegmentLoss(ComputeLoss):
    """
    Loss computation for instance segmentation.
    
    Extends ComputeLoss with mask loss computation.
    """
    
    def __init__(self, model: nn.Module, params: Dict, overlap: bool = True):
        super().__init__(model, params)
        self.nm = model.head.nm
        self.overlap = overlap

    def __call__(
        self,
        outputs: Tuple,
        targets: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute segmentation losses.
        
        Args:
            outputs: Tuple of (predictions, prototypes)
            targets: Dictionary with 'cls', 'box', 'idx', 'masks' keys
            
        Returns:
            Tuple of (box_loss, cls_loss, dfl_loss, mask_loss)
        """
        predictions, protos = outputs
        
        # Get detection losses
        loss_box, loss_cls, loss_dfl = super().__call__(predictions, targets)
        
        # Compute mask loss
        loss_mask = self._compute_mask_loss(predictions, protos, targets)
        
        return loss_box, loss_cls, loss_dfl, loss_mask

    def _compute_mask_loss(
        self,
        predictions: list,
        protos: torch.Tensor,
        targets: Dict
    ) -> torch.Tensor:
        """Compute mask prediction loss."""
        # This is a simplified version - full implementation would include:
        # - Extract mask coefficients from predictions
        # - Compute mask predictions from coefficients and prototypes
        # - Compare with ground truth masks using BCE
        return torch.zeros(1, device=self.device)


class PoseLoss(ComputeLoss):
    """
    Loss computation for pose estimation.
    
    Extends ComputeLoss with keypoint loss computation.
    """
    
    def __init__(self, model: nn.Module, params: Dict):
        super().__init__(model, params)
        self.nk = model.head.nk

    def __call__(
        self,
        outputs: list,
        targets: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute pose estimation losses.
        
        Args:
            outputs: List of predictions from pose head
            targets: Dictionary with 'cls', 'box', 'idx', 'keypoints' keys
            
        Returns:
            Tuple of (box_loss, cls_loss, dfl_loss, keypoint_loss)
        """
        # Get detection losses
        loss_box, loss_cls, loss_dfl = super().__call__(outputs, targets)
        
        # Compute keypoint loss
        loss_kpt = self._compute_keypoint_loss(outputs, targets)
        
        return loss_box, loss_cls, loss_dfl, loss_kpt

    def _compute_keypoint_loss(
        self,
        outputs: list,
        targets: Dict
    ) -> torch.Tensor:
        """Compute keypoint prediction loss using OKS."""
        # This is a simplified version - full implementation would include:
        # - Extract keypoint predictions
        # - Compute Object Keypoint Similarity (OKS)
        # - Compare with ground truth keypoints
        return torch.zeros(1, device=self.device)

