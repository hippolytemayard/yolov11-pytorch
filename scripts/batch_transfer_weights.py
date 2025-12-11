#!/usr/bin/env python3
"""
Transfer Ultralytics YOLOv11 weights to open-source format.

Usage:
    python batch_transfer_weights.py
"""

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo.models import yolo_v11_n, yolo_v11_s, yolo_v11_m, yolo_v11_l, yolo_v11_x


def create_dummy_ultralytics_modules():
    """Create fake ultralytics modules for unpickling."""
    
    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__()
            except:
                pass
            self._custom_state = {}
        
        def __reduce_ex__(self, protocol):
            return (self.__class__, ())
        
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            self._custom_state = state if isinstance(state, dict) else {}
        
        def forward(self, x, *args, **kwargs):
            return x
    
    module_names = [
        'ultralytics', 'ultralytics.nn', 'ultralytics.nn.tasks',
        'ultralytics.nn.modules', 'ultralytics.nn.modules.head',
        'ultralytics.nn.modules.block', 'ultralytics.nn.modules.conv',
        'ultralytics.nn.modules.transformer', 'ultralytics.engine',
        'ultralytics.engine.model', 'ultralytics.engine.results',
        'ultralytics.utils', 'ultralytics.utils.torch_utils',
        'ultralytics.models', 'ultralytics.models.yolo',
        'ultralytics.models.yolo.detect', 'ultralytics.models.yolo.segment',
        'ultralytics.models.yolo.pose',
    ]
    
    class DummyModuleContainer(types.ModuleType):
        def __getattr__(self, name):
            return DummyModule
    
    for name in module_names:
        sys.modules[name] = DummyModuleContainer(name)
    
    return DummyModule


def extract_state_dict(obj, prefix=''):
    """Recursively extract tensors from object."""
    result = {}
    
    if isinstance(obj, torch.Tensor):
        key = prefix.rstrip('.')
        if key:
            result[key] = obj
        return result
    
    if isinstance(obj, nn.Module):
        try:
            return {f"{prefix}{k}".lstrip('.'): v for k, v in obj.state_dict().items()}
        except:
            pass
    
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}{k}." if prefix else f"{k}."
            result.update(extract_state_dict(v, new_prefix))
        return result
    
    if hasattr(obj, '__dict__'):
        obj_dict = obj.__dict__
        for attr in ['_modules', '_parameters', '_buffers']:
            if attr in obj_dict and isinstance(obj_dict[attr], dict):
                for name, item in obj_dict[attr].items():
                    if item is not None:
                        result.update(extract_state_dict(item, f"{prefix}{name}."))
        
        for k, v in obj_dict.items():
            if isinstance(v, torch.Tensor):
                result[f"{prefix}{k}".lstrip('.')] = v
            elif isinstance(v, nn.Module):
                result.update(extract_state_dict(v, f"{prefix}{k}."))
    
    return result


def load_checkpoint(filepath):
    """Load checkpoint and extract state dict."""
    create_dummy_ultralytics_modules()
    
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        state_dict = None
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
                if isinstance(model, nn.Module):
                    state_dict = model.state_dict()
                elif isinstance(model, dict):
                    state_dict = {k: v for k, v in model.items() if isinstance(v, torch.Tensor)}
                else:
                    state_dict = extract_state_dict(model, 'model.')
                    state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
            elif any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        elif isinstance(checkpoint, nn.Module):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = extract_state_dict(checkpoint)
        
        if state_dict:
            state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        
        return state_dict
        
    except Exception as e:
        print(f"    Error: {e}")
        return None
    finally:
        to_remove = [k for k in sys.modules if k.startswith('ultralytics')]
        for k in to_remove:
            del sys.modules[k]


class WeightMatcher:
    """Maps Ultralytics weights to open-source model."""
    
    def __init__(self, task='detect'):
        self.task = task
        self.mapping = {}
        self._build()
    
    def _conv(self, src, dst):
        for s in ['.conv.weight', '.bn.weight', '.bn.bias', 
                  '.bn.running_mean', '.bn.running_var', '.bn.num_batches_tracked']:
            self.mapping[src + s] = dst + s.replace('.bn.', '.norm.')
    
    def _csp(self, src, dst):
        self._conv(f"{src}.cv1", f"{dst}.conv1")
        self._conv(f"{src}.cv2", f"{dst}.conv2")
        for i in range(2):
            self._conv(f"{src}.m.{i}.cv1", f"{dst}.res_m.{i}.conv1")
            self._conv(f"{src}.m.{i}.cv2", f"{dst}.res_m.{i}.conv2")
            self._conv(f"{src}.m.{i}.cv3", f"{dst}.res_m.{i}.conv3")
            for j in range(2):
                self._conv(f"{src}.m.{i}.m.{j}.cv1", f"{dst}.res_m.{i}.res_m.{j}.conv1")
                self._conv(f"{src}.m.{i}.m.{j}.cv2", f"{dst}.res_m.{i}.res_m.{j}.conv2")
    
    def _build(self):
        # Backbone
        self._conv("model.0", "net.p1.0")
        self._conv("model.1", "net.p2.0")
        self._csp("model.2", "net.p2.1")
        self._conv("model.3", "net.p3.0")
        self._csp("model.4", "net.p3.1")
        self._conv("model.5", "net.p4.0")
        self._csp("model.6", "net.p4.1")
        self._conv("model.7", "net.p5.0")
        self._csp("model.8", "net.p5.1")
        
        # SPP
        self._conv("model.9.cv1", "net.p5.2.conv1")
        self._conv("model.9.cv2", "net.p5.2.conv2")
        
        # PSA
        self._conv("model.10.cv1", "net.p5.3.conv1")
        self._conv("model.10.cv2", "net.p5.3.conv2")
        for i in range(2):
            self._conv(f"model.10.m.{i}.attn.qkv", f"net.p5.3.res_m.{i}.conv1.qkv")
            self._conv(f"model.10.m.{i}.attn.pe", f"net.p5.3.res_m.{i}.conv1.conv1")
            self._conv(f"model.10.m.{i}.attn.proj", f"net.p5.3.res_m.{i}.conv1.conv2")
            self._conv(f"model.10.m.{i}.ffn.0", f"net.p5.3.res_m.{i}.conv2.0")
            self._conv(f"model.10.m.{i}.ffn.1", f"net.p5.3.res_m.{i}.conv2.1")
        
        # FPN
        self._csp("model.13", "fpn.h1")
        self._csp("model.16", "fpn.h2")
        self._conv("model.17", "fpn.h3")
        self._csp("model.19", "fpn.h4")
        self._conv("model.20", "fpn.h5")
        self._csp("model.22", "fpn.h6")
        
        # Detection head
        for i in range(3):
            self._conv(f"model.23.cv2.{i}.0", f"head.box.{i}.0")
            self._conv(f"model.23.cv2.{i}.1", f"head.box.{i}.1")
            self.mapping[f"model.23.cv2.{i}.2.weight"] = f"head.box.{i}.2.weight"
            self.mapping[f"model.23.cv2.{i}.2.bias"] = f"head.box.{i}.2.bias"
            
            self._conv(f"model.23.cv3.{i}.0.0", f"head.cls.{i}.0")
            self._conv(f"model.23.cv3.{i}.0.1", f"head.cls.{i}.1")
            self._conv(f"model.23.cv3.{i}.1.0", f"head.cls.{i}.2")
            self._conv(f"model.23.cv3.{i}.1.1", f"head.cls.{i}.3")
            self.mapping[f"model.23.cv3.{i}.2.weight"] = f"head.cls.{i}.4.weight"
            self.mapping[f"model.23.cv3.{i}.2.bias"] = f"head.cls.{i}.4.bias"
        
        self.mapping["model.23.dfl.conv.weight"] = "head.dfl.conv.weight"
        
        # Segmentation extras
        if self.task == 'segment':
            self._conv("model.23.proto.cv1", "head.proto.conv1")
            self.mapping["model.23.proto.upsample.weight"] = "head.proto.upsample.weight"
            self.mapping["model.23.proto.upsample.bias"] = "head.proto.upsample.bias"
            self._conv("model.23.proto.cv2", "head.proto.conv2")
            self._conv("model.23.proto.cv3", "head.proto.conv3")
            
            for i in range(3):
                self._conv(f"model.23.cv4.{i}.0", f"head.mask.{i}.0")
                self._conv(f"model.23.cv4.{i}.1", f"head.mask.{i}.1")
                self.mapping[f"model.23.cv4.{i}.2.weight"] = f"head.mask.{i}.2.weight"
                self.mapping[f"model.23.cv4.{i}.2.bias"] = f"head.mask.{i}.2.bias"
        
        # Pose extras
        if self.task == 'pose':
            for i in range(3):
                self._conv(f"model.23.cv4.{i}.0", f"head.kpt.{i}.0")
                self._conv(f"model.23.cv4.{i}.1", f"head.kpt.{i}.1")
                self.mapping[f"model.23.cv4.{i}.2.weight"] = f"head.kpt.{i}.2.weight"
                self.mapping[f"model.23.cv4.{i}.2.bias"] = f"head.kpt.{i}.2.bias"
    
    def transfer(self, src_dict, model):
        tgt_dict = model.state_dict()
        stats = {'ok': 0, 'shape': 0, 'missing': 0, 'unmapped': 0}
        mismatches = []
        
        for sk, sv in src_dict.items():
            if sk in self.mapping:
                tk = self.mapping[sk]
                if tk in tgt_dict:
                    if sv.shape == tgt_dict[tk].shape:
                        tgt_dict[tk] = sv.clone()
                        stats['ok'] += 1
                    else:
                        stats['shape'] += 1
                        mismatches.append((sk, tk, sv.shape, tgt_dict[tk].shape))
                else:
                    stats['missing'] += 1
            else:
                stats['unmapped'] += 1
        
        model.load_state_dict(tgt_dict)
        return stats, mismatches


def create_model(size, task):
    m = {'n': yolo_v11_n, 's': yolo_v11_s, 'm': yolo_v11_m, 'l': yolo_v11_l, 'x': yolo_v11_x}
    nc = 1 if task == 'pose' else 80
    return m[size](nc, task=task)


def main():
    src_dir = Path(__file__).parent.parent / "weights" / "ultralytics"
    out_dir = Path(__file__).parent.parent / "weights" / "open"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    models = [
        ('yolo11n.pt', 'n', 'detect'), ('yolo11s.pt', 's', 'detect'),
        ('yolo11m.pt', 'm', 'detect'), ('yolo11l.pt', 'l', 'detect'),
        ('yolo11x.pt', 'x', 'detect'),
        ('yolo11n-seg.pt', 'n', 'segment'), ('yolo11s-seg.pt', 's', 'segment'),
        ('yolo11m-seg.pt', 'm', 'segment'), ('yolo11l-seg.pt', 'l', 'segment'),
        ('yolo11x-seg.pt', 'x', 'segment'),
        ('yolo11n-pose.pt', 'n', 'pose'), ('yolo11s-pose.pt', 's', 'pose'),
        ('yolo11m-pose.pt', 'm', 'pose'), ('yolo11l-pose.pt', 'l', 'pose'),
        ('yolo11x-pose.pt', 'x', 'pose'),
    ]
    
    print("=" * 60)
    print("YOLO v11 Weight Transfer")
    print("=" * 60)
    
    results = []
    
    for fname, size, task in models:
        path = src_dir / fname
        if not path.exists():
            continue
        
        print(f"\n{fname} ({size.upper()}, {task})")
        
        try:
            src_dict = load_checkpoint(str(path))
            if not src_dict:
                print("  ❌ Load failed")
                results.append((fname, 0, False))
                continue
            
            print(f"  Source: {len(src_dict)} tensors")
            
            model = create_model(size, task)
            matcher = WeightMatcher(task)
            stats, mismatches = matcher.transfer(src_dict, model)
            
            print(f"  ✓ Transferred: {stats['ok']}")
            if stats['shape']:
                print(f"  ⚠ Shape mismatch: {stats['shape']}")
                for sk, tk, ss, ts in mismatches[:5]:
                    print(f"      {sk}: {list(ss)} vs {list(ts)}")
            
            out_path = out_dir / fname.replace('.pt', '_open.pt')
            torch.save({'state_dict': model.state_dict(), 'task': task, 'size': size}, out_path)
            print(f"  💾 {out_path.name}")
            
            results.append((fname, stats['ok'], stats['shape'] == 0))
            
        except Exception as e:
            print(f"  ❌ {e}")
            results.append((fname, 0, False))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for name, count, ok in results:
        print(f"  {'✓' if ok else '⚠'} {name}: {count} weights")
    print(f"\nPerfect: {sum(1 for _,_,ok in results if ok)}/{len(results)}")


if __name__ == '__main__':
    main()
