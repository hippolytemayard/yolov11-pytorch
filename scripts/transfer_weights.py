#!/usr/bin/env python3
"""
Weight Transfer Utility for YOLO v11.

This script transfers weights from official Ultralytics YOLOv11 checkpoints
to the open-source implementation.

Usage:
    python transfer_weights.py --source yolo11n.pt --target weights/yolo11n_transferred.pt
    python transfer_weights.py --source yolo11x-seg.pt --target weights/yolo11x_seg.pt --task segment

Author: Open Source Community
License: Apache 2.0
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo.models import yolo_v11_n, yolo_v11_s, yolo_v11_m, yolo_v11_l, yolo_v11_x


class WeightMatcher:
    """
    Weight matching utility for transferring Ultralytics weights to open-source model.
    
    Handles the mapping between different naming conventions and structures.
    """
    
    def __init__(self):
        self.weight_mapping = {}
        self._build_mapping()

    def _build_mapping(self) -> None:
        """Build complete weight mapping."""
        
        # Backbone convolutions
        backbone_maps = [
            ("model.0", "net.p1.0"),
            ("model.1", "net.p2.0"),
            ("model.3", "net.p3.0"),
            ("model.5", "net.p4.0"),
            ("model.7", "net.p5.0"),
        ]
        
        for official, impl in backbone_maps:
            self._add_conv_mapping(official, impl)
        
        # CSP blocks
        csp_maps = [
            ("model.2", "net.p2.1"),
            ("model.4", "net.p3.1"),
            ("model.6", "net.p4.1"),
            ("model.8", "net.p5.1"),
        ]
        
        for official, impl in csp_maps:
            self._add_csp_mapping(official, impl)
        
        # SPP block
        self._add_spp_mapping("model.9", "net.p5.2")
        
        # PSA block
        self._add_psa_mapping("model.10", "net.p5.3")
        
        # FPN
        fpn_maps = [
            ("model.13", "fpn.h1"),
            ("model.16", "fpn.h2"),
            ("model.19", "fpn.h4"),
            ("model.22", "fpn.h6"),
        ]
        
        for official, impl in fpn_maps:
            self._add_csp_mapping(official, impl)
        
        self._add_conv_mapping("model.17", "fpn.h3")
        self._add_conv_mapping("model.20", "fpn.h5")
        
        # Detection head
        self._add_detection_head_mapping()

    def _add_conv_mapping(self, official: str, impl: str) -> None:
        """Add mapping for a convolution layer."""
        suffixes = [
            ".conv.weight", ".bn.weight", ".bn.bias",
            ".bn.running_mean", ".bn.running_var", ".bn.num_batches_tracked"
        ]
        
        for suffix in suffixes:
            official_key = official + suffix
            impl_key = impl + suffix.replace(".bn.", ".norm.")
            self.weight_mapping[official_key] = impl_key

    def _add_csp_mapping(self, official: str, impl: str) -> None:
        """Add mapping for CSP block."""
        self._add_conv_mapping(f"{official}.cv1", f"{impl}.conv1")
        self._add_conv_mapping(f"{official}.cv2", f"{impl}.conv2")
        
        for i in range(2):
            res = f"{official}.m.{i}"
            impl_res = f"{impl}.res_m.{i}"
            
            self._add_conv_mapping(f"{res}.cv1", f"{impl_res}.conv1")
            self._add_conv_mapping(f"{res}.cv2", f"{impl_res}.conv2")
            self._add_conv_mapping(f"{res}.cv3", f"{impl_res}.conv3")
            
            for j in range(2):
                self._add_conv_mapping(f"{res}.m.{j}.cv1", f"{impl_res}.res_m.{j}.conv1")
                self._add_conv_mapping(f"{res}.m.{j}.cv2", f"{impl_res}.res_m.{j}.conv2")

    def _add_spp_mapping(self, official: str, impl: str) -> None:
        """Add mapping for SPP block."""
        self._add_conv_mapping(f"{official}.cv1", f"{impl}.conv1")
        self._add_conv_mapping(f"{official}.cv2", f"{impl}.conv2")

    def _add_psa_mapping(self, official: str, impl: str) -> None:
        """Add mapping for PSA block."""
        self._add_conv_mapping(f"{official}.cv1", f"{impl}.conv1")
        self._add_conv_mapping(f"{official}.cv2", f"{impl}.conv2")
        
        for i in range(2):
            psa = f"{official}.m.{i}"
            impl_psa = f"{impl}.res_m.{i}"
            
            self._add_conv_mapping(f"{psa}.attn.qkv", f"{impl_psa}.conv1.qkv")
            self._add_conv_mapping(f"{psa}.attn.proj", f"{impl_psa}.conv1.conv2")
            self._add_conv_mapping(f"{psa}.attn.pe", f"{impl_psa}.conv1.conv1")
            
            self._add_conv_mapping(f"{psa}.ffn.0", f"{impl_psa}.conv2.0")
            self._add_conv_mapping(f"{psa}.ffn.1", f"{impl_psa}.conv2.1")

    def _add_detection_head_mapping(self) -> None:
        """Add mapping for detection head."""
        for i in range(3):
            # Box heads
            self._add_conv_mapping(f"model.23.cv2.{i}.0", f"head.box.{i}.0")
            self._add_conv_mapping(f"model.23.cv2.{i}.1", f"head.box.{i}.1")
            self.weight_mapping[f"model.23.cv2.{i}.2.weight"] = f"head.box.{i}.2.weight"
            self.weight_mapping[f"model.23.cv2.{i}.2.bias"] = f"head.box.{i}.2.bias"
            
            # Class heads
            self._add_conv_mapping(f"model.23.cv3.{i}.0.0", f"head.cls.{i}.0")
            self._add_conv_mapping(f"model.23.cv3.{i}.0.1", f"head.cls.{i}.1")
            self._add_conv_mapping(f"model.23.cv3.{i}.1.0", f"head.cls.{i}.2")
            self._add_conv_mapping(f"model.23.cv3.{i}.1.1", f"head.cls.{i}.3")
            self.weight_mapping[f"model.23.cv3.{i}.2.weight"] = f"head.cls.{i}.4.weight"
            self.weight_mapping[f"model.23.cv3.{i}.2.bias"] = f"head.cls.{i}.4.bias"
        
        # DFL
        self.weight_mapping["model.23.dfl.conv.weight"] = "head.dfl.conv.weight"

    def transfer_weights(
        self,
        source_path: str,
        target_model: torch.nn.Module,
        verbose: bool = True
    ) -> Dict[str, int]:
        """
        Transfer weights from source checkpoint to target model.
        
        Args:
            source_path: Path to source checkpoint
            target_model: Target model instance
            verbose: Whether to print transfer details
            
        Returns:
            Dictionary with transfer statistics
        """
        print(f"Loading weights from: {source_path}")
        
        # Load source checkpoint
        try:
            checkpoint = torch.load(source_path, map_location='cpu', weights_only=True)
        except Exception:
            checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint and checkpoint['model'] is not None:
                source_dict = checkpoint['model'].state_dict()
            elif 'EMA' in checkpoint and checkpoint['EMA'] is not None:
                source_dict = checkpoint['EMA'].state_dict()
            else:
                source_dict = checkpoint
        else:
            source_dict = checkpoint.state_dict()
        
        target_dict = target_model.state_dict()
        
        # Statistics
        stats = {
            'transferred': 0,
            'shape_mismatch': 0,
            'missing': 0,
            'ignored': 0,
        }
        
        if verbose:
            print(f"Source layers: {len(source_dict)}")
            print(f"Target layers: {len(target_dict)}")
            print(f"Mappings available: {len(self.weight_mapping)}")
            print("-" * 60)
        
        # Transfer weights
        for source_key, source_weight in source_dict.items():
            if source_key in self.weight_mapping:
                target_key = self.weight_mapping[source_key]
                
                if target_key in target_dict:
                    if source_weight.shape == target_dict[target_key].shape:
                        target_dict[target_key] = source_weight
                        stats['transferred'] += 1
                        
                        if verbose and stats['transferred'] <= 10:
                            print(f"✓ {source_key} -> {target_key}")
                    else:
                        stats['shape_mismatch'] += 1
                        if verbose:
                            print(f"⚠ Shape mismatch: {source_key} {source_weight.shape} "
                                  f"-> {target_key} {target_dict[target_key].shape}")
                else:
                    stats['missing'] += 1
            else:
                stats['ignored'] += 1
        
        # Load transferred weights
        target_model.load_state_dict(target_dict)
        
        if verbose:
            print("-" * 60)
            print(f"Transfer Summary:")
            print(f"  ✓ Transferred: {stats['transferred']}")
            print(f"  ⚠ Shape mismatch: {stats['shape_mismatch']}")
            print(f"  ✗ Missing: {stats['missing']}")
            print(f"  → Ignored: {stats['ignored']}")
            
            total = stats['transferred'] + stats['ignored']
            if total > 0:
                print(f"  Coverage: {stats['transferred'] / total * 100:.1f}%")
        
        return stats

    def analyze_coverage(
        self,
        source_path: str,
        target_model: torch.nn.Module
    ) -> None:
        """Analyze mapping coverage."""
        try:
            checkpoint = torch.load(source_path, map_location='cpu', weights_only=True)
        except Exception:
            checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            source_dict = checkpoint['model'].state_dict()
        else:
            source_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        
        target_dict = target_model.state_dict()
        
        mapped_keys = set(self.weight_mapping.keys())
        source_keys = set(source_dict.keys())
        target_keys = set(target_dict.keys())
        
        mapped_source = mapped_keys.intersection(source_keys)
        unmapped_source = source_keys - mapped_keys
        
        print("Coverage Analysis:")
        print(f"  Source keys: {len(source_keys)}")
        print(f"  Mapped keys: {len(mapped_keys)}")
        print(f"  Matched: {len(mapped_source)}")
        print(f"  Unmapped source keys: {len(unmapped_source)}")
        
        if unmapped_source:
            print("\nUnmapped source keys (first 20):")
            for key in list(unmapped_source)[:20]:
                print(f"  - {key}")


def get_model_from_size(size: str, task: str, num_classes: int = 80):
    """Get model based on size and task."""
    model_map = {
        'n': yolo_v11_n,
        's': yolo_v11_s,
        'm': yolo_v11_m,
        'l': yolo_v11_l,
        'x': yolo_v11_x,
    }
    
    if size not in model_map:
        raise ValueError(f"Unknown model size: {size}")
    
    return model_map[size](num_classes, task=task)


def main():
    parser = argparse.ArgumentParser(description='Transfer YOLO weights')
    
    parser.add_argument('--source', required=True, help='Source checkpoint path')
    parser.add_argument('--target', required=True, help='Target checkpoint path')
    parser.add_argument('--size', default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size')
    parser.add_argument('--task', default='detect', choices=['detect', 'segment', 'pose'],
                        help='Task type')
    parser.add_argument('--num-classes', type=int, default=80, help='Number of classes')
    parser.add_argument('--analyze', action='store_true', help='Analyze coverage only')
    
    args = parser.parse_args()
    
    # Create model
    model = get_model_from_size(args.size, args.task, args.num_classes)
    print(f"Created YOLOv11-{args.size.upper()} for {args.task}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Transfer weights
    matcher = WeightMatcher()
    
    if args.analyze:
        matcher.analyze_coverage(args.source, model)
    else:
        stats = matcher.transfer_weights(args.source, model)
        
        # Save transferred model
        checkpoint = {
            'model': model,
            'task': args.task,
            'num_classes': args.num_classes,
        }
        
        Path(args.target).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.target)
        print(f"\nSaved transferred weights to: {args.target}")


if __name__ == '__main__':
    main()

