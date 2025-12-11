"""
Feature Pyramid Network (FPN) Neck for YOLO v11.

The neck combines features from the backbone at different scales
to create a rich feature representation for detection.
"""

from typing import Tuple, List

import torch
import torch.nn as nn

from yolo.models.common import Conv, CSP


class DarkFPN(nn.Module):
    """
    Feature Pyramid Network (FPN) neck for YOLO v11.
    
    This module takes multi-scale features from the backbone and fuses them
    using a top-down pathway with lateral connections, followed by a
    bottom-up pathway for better feature aggregation.
    
    Architecture:
    - Top-down: P5 -> P4 -> P3 (with upsampling and concatenation)
    - Bottom-up: P3 -> P4 -> P5 (with downsampling and concatenation)
    
    Args:
        width: List of channel widths
        depth: List of depths for CSP blocks
        csp: List of booleans for CSP module type
    """
    
    def __init__(
        self,
        width: List[int],
        depth: List[int],
        csp: List[bool]
    ):
        super().__init__()
        
        # Upsampling layer for top-down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Top-down pathway
        self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp[0], r=2)  # P5 + P4 -> P4'
        self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp[0], r=2)  # P4' + P3 -> P3'
        
        # Bottom-up pathway
        self.h3 = Conv(width[3], width[3], nn.SiLU(), k=3, s=2, p=1)  # Downsample P3'
        self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp[0], r=2)  # P3' + P4' -> P4''
        
        self.h5 = Conv(width[4], width[4], nn.SiLU(), k=3, s=2, p=1)  # Downsample P4''
        self.h6 = CSP(width[4] + width[5], width[5], depth[5], csp[1], r=2)  # P4'' + P5 -> P5''

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fuse multi-scale features.
        
        Args:
            features: Tuple of (P3, P4, P5) from backbone
            
        Returns:
            Tuple of fused (P3', P4', P5') features
        """
        p3, p4, p5 = features
        
        # Top-down pathway
        p4 = self.h1(torch.cat([self.upsample(p5), p4], dim=1))
        p3 = self.h2(torch.cat([self.upsample(p4), p3], dim=1))
        
        # Bottom-up pathway
        p4 = self.h4(torch.cat([self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat([self.h5(p4), p5], dim=1))
        
        return p3, p4, p5

