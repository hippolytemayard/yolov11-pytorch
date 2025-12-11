"""
DarkNet Backbone for YOLO v11.

The backbone extracts multi-scale features from input images using
a series of convolutions, CSP blocks, SPP, and PSA modules.
"""

from typing import Tuple, List

import torch
import torch.nn as nn

from yolo.models.common import Conv, CSP, SPP, PSA


class DarkNet(nn.Module):
    """
    DarkNet backbone for YOLO v11.
    
    This backbone extracts features at multiple scales (P3, P4, P5)
    for use in the detection/segmentation head.
    
    Architecture:
    - P1 (stride 2): Initial convolution
    - P2 (stride 4): Convolution + CSP
    - P3 (stride 8): Convolution + CSP -> Output
    - P4 (stride 16): Convolution + CSP -> Output
    - P5 (stride 32): Convolution + CSP + SPP + PSA -> Output
    
    Args:
        width: List of channel widths for each stage
        depth: List of depths (number of blocks) for each stage
        csp: List of booleans indicating CSP module type per stage
    """
    
    def __init__(
        self,
        width: List[int],
        depth: List[int],
        csp: List[bool]
    ):
        super().__init__()
        
        # P1/2 - Initial downsampling
        self.p1 = nn.Sequential(
            Conv(width[0], width[1], nn.SiLU(), k=3, s=2, p=1)
        )
        
        # P2/4 - Second stage
        self.p2 = nn.Sequential(
            Conv(width[1], width[2], nn.SiLU(), k=3, s=2, p=1),
            CSP(width[2], width[3], depth[0], csp[0], r=4)
        )
        
        # P3/8 - Third stage (output)
        self.p3 = nn.Sequential(
            Conv(width[3], width[3], nn.SiLU(), k=3, s=2, p=1),
            CSP(width[3], width[4], depth[1], csp[0], r=4)
        )
        
        # P4/16 - Fourth stage (output)
        self.p4 = nn.Sequential(
            Conv(width[4], width[4], nn.SiLU(), k=3, s=2, p=1),
            CSP(width[4], width[4], depth[2], csp[1], r=2)
        )
        
        # P5/32 - Fifth stage with SPP and PSA (output)
        self.p5 = nn.Sequential(
            Conv(width[4], width[5], nn.SiLU(), k=3, s=2, p=1),
            CSP(width[5], width[5], depth[3], csp[1], r=2),
            SPP(width[5], width[5]),
            PSA(width[5], depth[4])
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (P3, P4, P5) feature maps at strides 8, 16, 32
        """
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5

