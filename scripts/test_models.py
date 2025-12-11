#!/usr/bin/env python3
"""
Test and compare converted YOLO v11 models with Ultralytics originals.

Usage:
    python test_models.py
"""

import sys
import types
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo.models import yolo_v11_n, yolo_v11_s, yolo_v11_m, yolo_v11_l, yolo_v11_x


def create_model(size: str, task: str):
    """Create our open-source model."""
    models = {'n': yolo_v11_n, 's': yolo_v11_s, 'm': yolo_v11_m, 'l': yolo_v11_l, 'x': yolo_v11_x}
    nc = 1 if task == 'pose' else 80
    return models[size](nc, task=task)


def load_our_weights(model: nn.Module, size: str, task: str) -> nn.Module:
    """Load our converted weights."""
    weights_dir = Path(__file__).parent.parent / "weights" / "open"
    
    if task == 'detect':
        filename = f"yolo11{size}_open.pt"
    elif task == 'segment':
        filename = f"yolo11{size}-seg_open.pt"
    else:
        filename = f"yolo11{size}-pose_open.pt"
    
    checkpoint = torch.load(weights_dir / filename, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model


def create_dummy_ultralytics():
    """Create fake ultralytics modules for loading checkpoints."""
    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__()
            except:
                pass
        def __reduce_ex__(self, protocol):
            return (self.__class__, ())
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
        def forward(self, x, *args, **kwargs):
            return x
    
    module_names = [
        'ultralytics', 'ultralytics.nn', 'ultralytics.nn.tasks',
        'ultralytics.nn.modules', 'ultralytics.nn.modules.head',
        'ultralytics.nn.modules.block', 'ultralytics.nn.modules.conv',
        'ultralytics.nn.modules.transformer',
    ]
    
    class Container(types.ModuleType):
        def __getattr__(self, name):
            return DummyModule
    
    for name in module_names:
        sys.modules[name] = Container(name)


def cleanup_dummy_ultralytics():
    """Remove fake ultralytics modules."""
    to_remove = [k for k in sys.modules if k.startswith('ultralytics')]
    for k in to_remove:
        del sys.modules[k]


def load_ultralytics_weights(size: str, task: str) -> Dict[str, torch.Tensor]:
    """Load Ultralytics checkpoint and extract state dict."""
    weights_dir = Path(__file__).parent.parent / "weights" / "ultralytics"
    
    if task == 'detect':
        filename = f"yolo11{size}.pt"
    elif task == 'segment':
        filename = f"yolo11{size}-seg.pt"
    else:
        filename = f"yolo11{size}-pose.pt"
    
    create_dummy_ultralytics()
    
    try:
        checkpoint = torch.load(weights_dir / filename, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
            if hasattr(model, 'state_dict'):
                return model.state_dict()
        return {}
    finally:
        cleanup_dummy_ultralytics()


def compare_weights(our_model: nn.Module, ultra_sd: Dict[str, torch.Tensor], size: str, task: str) -> Dict:
    """Compare weights between our model and Ultralytics."""
    our_sd = our_model.state_dict()
    
    # Build mapping (simplified)
    mapping = {}
    
    # Just compare total statistics for now
    our_total = sum(p.numel() for p in our_sd.values())
    ultra_total = sum(p.numel() for p in ultra_sd.values())
    
    # Compare backbone weights (should be identical)
    backbone_match = 0
    backbone_total = 0
    
    for our_key, our_val in our_sd.items():
        if 'net.' in our_key:  # Backbone
            backbone_total += 1
            # Find corresponding Ultralytics key
            for ultra_key, ultra_val in ultra_sd.items():
                if our_val.shape == ultra_val.shape:
                    diff = (our_val - ultra_val).abs().max().item()
                    if diff < 1e-6:
                        backbone_match += 1
                        break
    
    return {
        'our_params': len(our_sd),
        'ultra_params': len(ultra_sd),
        'our_elements': our_total,
        'ultra_elements': ultra_total,
        'backbone_matched': backbone_match,
        'backbone_total': backbone_total
    }


def test_forward_pass(model: nn.Module, task: str) -> Dict:
    """Test forward pass and return output info."""
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        out = model(x)
    
    if task == 'segment':
        preds, proto = out
        return {
            'success': True,
            'output_shape': list(preds.shape),
            'proto_shape': list(proto.shape)
        }
    else:
        return {
            'success': True,
            'output_shape': list(out.shape)
        }


def test_numerical_comparison(our_model: nn.Module, size: str, task: str) -> Dict:
    """Test that our model produces numerically similar outputs to Ultralytics."""
    try:
        # Try to import ultralytics
        from ultralytics import YOLO
        
        weights_dir = Path(__file__).parent.parent / "weights" / "ultralytics"
        
        if task == 'detect':
            filename = f"yolo11{size}.pt"
        elif task == 'segment':
            filename = f"yolo11{size}-seg.pt"
        else:
            filename = f"yolo11{size}-pose.pt"
        
        ultra_model = YOLO(str(weights_dir / filename))
        
        # Create test input
        x = torch.randn(1, 3, 640, 640)
        
        # Run both models
        our_model.eval()
        with torch.no_grad():
            our_out = our_model(x)
        
        # Get Ultralytics raw output
        ultra_out = ultra_model.model(x)
        
        # Compare
        if task == 'segment':
            our_preds = our_out[0]
            ultra_preds = ultra_out[0] if isinstance(ultra_out, (list, tuple)) else ultra_out
        else:
            our_preds = our_out
            ultra_preds = ultra_out[0] if isinstance(ultra_out, (list, tuple)) else ultra_out
        
        if our_preds.shape == ultra_preds.shape:
            diff = (our_preds - ultra_preds).abs()
            return {
                'available': True,
                'shapes_match': True,
                'max_diff': diff.max().item(),
                'mean_diff': diff.mean().item(),
                'matched': diff.max().item() < 0.01
            }
        else:
            return {
                'available': True,
                'shapes_match': False,
                'our_shape': list(our_preds.shape),
                'ultra_shape': list(ultra_preds.shape)
            }
            
    except ImportError:
        return {'available': False, 'reason': 'ultralytics not installed'}
    except Exception as e:
        return {'available': False, 'error': str(e)}


def main():
    print("=" * 70)
    print("YOLO v11 Open Source - Model Testing")
    print("=" * 70)
    
    models_to_test = [
        # Detection
        ('n', 'detect'), ('s', 'detect'), ('m', 'detect'), ('l', 'detect'), ('x', 'detect'),
        # Segmentation
        ('n', 'segment'), ('s', 'segment'), ('m', 'segment'), ('l', 'segment'), ('x', 'segment'),
        # Pose
        ('n', 'pose'), ('s', 'pose'), ('m', 'pose'), ('l', 'pose'), ('x', 'pose'),
    ]
    
    results = []
    
    for size, task in models_to_test:
        model_name = f"yolo11{size}" + (f"-{task}" if task != 'detect' else "")
        print(f"\n{'─'*70}")
        print(f"Testing: {model_name}")
        print("─" * 70)
        
        try:
            # Create and load model
            print("  Loading model...", end=" ")
            model = create_model(size, task)
            model = load_our_weights(model, size, task)
            print("✓")
            
            # Test forward pass
            print("  Forward pass...", end=" ")
            fwd_result = test_forward_pass(model, task)
            if fwd_result['success']:
                print(f"✓ Output: {fwd_result['output_shape']}")
                if 'proto_shape' in fwd_result:
                    print(f"           Proto: {fwd_result['proto_shape']}")
            else:
                print("✗")
            
            # Compare with Ultralytics numerically
            print("  Numerical comparison...", end=" ")
            num_result = test_numerical_comparison(model, size, task)
            if num_result.get('available'):
                if num_result.get('matched'):
                    print(f"✓ max_diff={num_result['max_diff']:.2e}")
                elif num_result.get('shapes_match'):
                    print(f"⚠ max_diff={num_result['max_diff']:.2e}")
                else:
                    print(f"✗ Shape mismatch")
            else:
                print(f"⚠ {num_result.get('reason', 'N/A')}")
            
            results.append({
                'model': model_name,
                'forward_pass': fwd_result['success'],
                'numerical': num_result
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'model': model_name,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    
    success = sum(1 for r in results if r.get('forward_pass', False))
    matched = sum(1 for r in results if r.get('numerical', {}).get('matched', False))
    
    for r in results:
        status = "✓" if r.get('forward_pass') else "✗"
        num_status = ""
        if r.get('numerical', {}).get('available'):
            if r.get('numerical', {}).get('matched'):
                num_status = " [MATCHED]"
            else:
                num_status = f" [diff={r['numerical'].get('max_diff', 'N/A'):.2e}]"
        print(f"  {status} {r['model']}{num_status}")
    
    print(f"\nForward pass: {success}/{len(results)}")
    print(f"Numerical match: {matched}/{len(results)} (with ultralytics)")


if __name__ == '__main__':
    main()
