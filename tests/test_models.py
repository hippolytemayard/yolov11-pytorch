"""
Tests for YOLO v11 model architecture.
"""

import pytest
import torch

from yolo.models import (
    YOLO,
    YOLODetect,
    YOLOSegment,
    YOLOPose,
    yolo_v11_n,
    yolo_v11_s,
    yolo_v11_m,
    yolo_v11_l,
    yolo_v11_x,
)
from yolo.models.backbone import DarkNet
from yolo.models.neck import DarkFPN
from yolo.models.common import Conv, CSP, SPP, PSA


class TestCommonModules:
    """Tests for common building blocks."""

    def test_conv_module(self):
        """Test Conv module."""
        conv = Conv(64, 128, torch.nn.SiLU(), k=3, p=1)
        x = torch.randn(1, 64, 32, 32)
        y = conv(x)
        assert y.shape == (1, 128, 32, 32)

    def test_csp_module(self):
        """Test CSP module."""
        csp = CSP(64, 128, n=2, csp=True, r=2)
        x = torch.randn(1, 64, 32, 32)
        y = csp(x)
        assert y.shape == (1, 128, 32, 32)

    def test_spp_module(self):
        """Test SPP module."""
        spp = SPP(256, 256)
        x = torch.randn(1, 256, 16, 16)
        y = spp(x)
        assert y.shape == (1, 256, 16, 16)

    def test_psa_module(self):
        """Test PSA module."""
        psa = PSA(256, n=2)
        x = torch.randn(1, 256, 16, 16)
        y = psa(x)
        assert y.shape == (1, 256, 16, 16)


class TestBackbone:
    """Tests for DarkNet backbone."""

    def test_backbone_output_shapes(self):
        """Test backbone produces correct output shapes."""
        width = [3, 16, 32, 64, 128, 256]
        depth = [1, 1, 1, 1, 1, 1]
        csp = [False, True]
        
        backbone = DarkNet(width, depth, csp)
        x = torch.randn(1, 3, 256, 256)
        
        p3, p4, p5 = backbone(x)
        
        assert p3.shape == (1, 128, 32, 32)
        assert p4.shape == (1, 128, 16, 16)
        assert p5.shape == (1, 256, 8, 8)


class TestNeck:
    """Tests for DarkFPN neck."""

    def test_fpn_output_shapes(self):
        """Test FPN produces correct output shapes."""
        width = [3, 16, 32, 64, 128, 256]
        depth = [1, 1, 1, 1, 1, 1]
        csp = [False, True]
        
        fpn = DarkFPN(width, depth, csp)
        
        p3 = torch.randn(1, 128, 32, 32)
        p4 = torch.randn(1, 128, 16, 16)
        p5 = torch.randn(1, 256, 8, 8)
        
        out3, out4, out5 = fpn((p3, p4, p5))
        
        assert out3.shape == (1, 64, 32, 32)
        assert out4.shape == (1, 128, 16, 16)
        assert out5.shape == (1, 256, 8, 8)


class TestYOLOModels:
    """Tests for complete YOLO models."""

    @pytest.mark.parametrize("model_fn", [
        yolo_v11_n,
        yolo_v11_s,
    ])
    def test_detection_model_inference(self, model_fn):
        """Test detection model inference."""
        model = model_fn(num_classes=80, task='detect')
        model.eval()
        
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            y = model(x)
        
        assert isinstance(y, torch.Tensor)
        assert y.shape[0] == 1
        assert y.shape[1] == 84  # 80 classes + 4 box coords

    def test_detection_model_training(self):
        """Test detection model in training mode."""
        model = yolo_v11_n(num_classes=80, task='detect')
        model.train()
        
        x = torch.randn(1, 3, 640, 640)
        y = model(x)
        
        assert isinstance(y, list)
        assert len(y) == 3  # 3 scale outputs

    def test_model_fuse(self):
        """Test model layer fusion."""
        model = yolo_v11_n(num_classes=80)
        model.eval()
        
        # Count Conv modules with norm before fusion
        conv_with_norm = sum(1 for m in model.modules() if isinstance(m, Conv) and hasattr(m, 'norm'))
        assert conv_with_norm > 0
        
        # Fuse
        model = model.fuse()
        
        # Count after fusion
        conv_with_norm_after = sum(1 for m in model.modules() if isinstance(m, Conv) and hasattr(m, 'norm'))
        assert conv_with_norm_after == 0


class TestModelProperties:
    """Tests for model properties."""

    def test_num_classes(self):
        """Test num_classes property."""
        model = yolo_v11_n(num_classes=20)
        assert model.num_classes == 20

    def test_num_params(self):
        """Test num_params property."""
        model = yolo_v11_n(num_classes=80)
        assert model.num_params > 0
        assert isinstance(model.num_params, int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

