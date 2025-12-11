"""
Training utilities for YOLO v11.

This module provides:
- Learning rate schedulers
- EMA (Exponential Moving Average)
- Checkpoint utilities
- Training helpers
"""

import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def setup_seed(seed: int = 0) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes() -> None:
    """Configure environment for multi-process training."""
    import cv2
    from os import environ
    from platform import system

    # Use fork for faster multiprocessing (not on Windows)
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # Disable OpenCV multithreading to avoid system overload
    cv2.setNumThreads(0)

    # Set thread limits
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a moving average of model weights for more stable predictions.
    
    Reference: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    
    Args:
        model: Model to track
        decay: EMA decay rate
        tau: Decay ramp-up period
        updates: Initial update count
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        tau: int = 2000,
        updates: int = 0
    ):
        self.ema = copy.deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        if hasattr(model, 'module'):
            model = model.module
            
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class LinearLR:
    """
    Linear learning rate scheduler with warmup.
    
    Args:
        args: Training arguments with epochs
        params: Parameters with max_lr, min_lr, warmup_epochs
        num_steps: Number of steps per epoch
    """
    
    def __init__(self, args, params: Dict, num_steps: int):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = np.linspace(min_lr, max_lr, warmup_steps, endpoint=False)
        decay_lr = np.linspace(max_lr, min_lr, decay_steps)

        self.total_lr = np.concatenate((warmup_lr, decay_lr))

    def step(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        """Update learning rate."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class CosineLR:
    """
    Cosine annealing learning rate scheduler with warmup.
    
    Args:
        args: Training arguments with epochs
        params: Parameters with max_lr, min_lr, warmup_epochs
        num_steps: Number of steps per epoch
    """
    
    def __init__(self, args, params: Dict, num_steps: int):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = np.linspace(min_lr, max_lr, warmup_steps)

        decay_lr = []
        for step in range(1, decay_steps + 1):
            alpha = math.cos(math.pi * step / decay_steps)
            decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))

        self.total_lr = np.concatenate((warmup_lr, decay_lr))

    def step(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        """Update learning rate."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class AverageMeter:
    """Track running average of a metric."""
    
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, val: float, n: int = 1) -> None:
        if not math.isnan(val):
            self.num += n
            self.sum += val * n
            self.avg = self.sum / self.num


def set_params(model: nn.Module, weight_decay: float) -> List[Dict]:
    """
    Separate model parameters for optimizer with different weight decay.
    
    Biases and normalization layers get zero weight decay.
    
    Args:
        model: Model to get parameters from
        weight_decay: Weight decay for regular parameters
        
    Returns:
        List of parameter groups
    """
    no_decay = []
    decay = []
    norm_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    
    for m in model.modules():
        for n, p in m.named_parameters(recurse=0):
            if not p.requires_grad:
                continue
            if n == "bias":
                no_decay.append(p)
            elif n == "weight" and isinstance(m, norm_types):
                no_decay.append(p)
            else:
                decay.append(p)
    
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def strip_optimizer(checkpoint_path: str) -> None:
    """
    Strip optimizer state from checkpoint for deployment.
    
    Also converts model to half precision.
    
    Args:
        checkpoint_path: Path to checkpoint file
    """
    x = torch.load(checkpoint_path, map_location="cpu")
    x['model'].half()
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, checkpoint_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False
) -> nn.Module:
    """
    Load weights from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        strict: Whether to require exact match
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model'].float().state_dict()
    else:
        state_dict = checkpoint
    
    # Filter matching keys
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    
    model.load_state_dict(pretrained_dict, strict=strict)
    
    return model

