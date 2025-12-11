#!/usr/bin/env python3
"""
Visual comparison of YOLO v11 models on a real image.
"""

import sys
from pathlib import Path
import urllib.request

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo.models import yolo_v11_n


def download_test_image():
    """Download COCO test image."""
    url = "http://images.cocodataset.org/val2017/000000000139.jpg"
    save_path = Path(__file__).parent.parent / "data" / "coco_test.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not save_path.exists():
        print(f"Downloading test image...")
        urllib.request.urlretrieve(url, save_path)
    
    return save_path


def load_image(path, size=640):
    """Load and preprocess image with letterboxing."""
    from PIL import Image
    
    img = Image.open(path).convert('RGB')
    orig_size = img.size  # (W, H)
    orig_np = np.array(img)
    
    # Resize maintaining aspect ratio
    ratio = size / max(orig_size)
    new_size = (int(orig_size[0] * ratio), int(orig_size[1] * ratio))
    img_resized = img.resize(new_size, Image.BILINEAR)
    
    # Pad to square (letterboxing)
    padded = Image.new('RGB', (size, size), (114, 114, 114))
    pad_x = (size - new_size[0]) // 2
    pad_y = (size - new_size[1]) // 2
    padded.paste(img_resized, (pad_x, pad_y))
    
    tensor = torch.from_numpy(np.array(padded)).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    
    # Return padding info for coordinate transformation
    padding_info = {
        'pad_x': pad_x,
        'pad_y': pad_y,
        'ratio': ratio,
        'orig_size': orig_size,  # (W, H)
        'input_size': size
    }
    
    return tensor, orig_np, padding_info


def scale_boxes_to_original(boxes, padding_info):
    """
    Scale boxes from input tensor space (640x640) back to original image coordinates.
    
    Args:
        boxes: numpy array of shape (N, 4) in xyxy format, in 640x640 space
        padding_info: dict with pad_x, pad_y, ratio, orig_size
    
    Returns:
        boxes scaled to original image coordinates
    """
    if len(boxes) == 0:
        return boxes
    
    boxes = boxes.copy()
    
    # Remove padding offset
    boxes[:, [0, 2]] -= padding_info['pad_x']  # x coordinates
    boxes[:, [1, 3]] -= padding_info['pad_y']  # y coordinates
    
    # Scale back to original size
    boxes /= padding_info['ratio']
    
    # Clip to image bounds
    orig_w, orig_h = padding_info['orig_size']
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    
    return boxes


def run_detection(model, input_tensor, conf_thres=0.25, iou_thres=0.45):
    """Run detection inference."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # Output shape: [1, 84, 8400] -> [1, 8400, 84]
    output = output.permute(0, 2, 1)
    
    # Split box and class predictions
    # Format: [cx, cy, w, h, cls_scores...] - already decoded in inference mode
    boxes = output[..., :4]  # [1, 8400, 4] - cxcywh format
    scores = output[..., 4:]  # [1, 8400, 80]
    
    # Get max class score and index
    max_scores, class_ids = scores.max(dim=-1)
    
    # Filter by confidence
    mask = max_scores > conf_thres
    
    results = []
    for batch_idx in range(output.shape[0]):
        batch_mask = mask[batch_idx]
        
        if batch_mask.sum() == 0:
            results.append({'boxes': np.array([]), 'scores': np.array([]), 'classes': np.array([])})
            continue
        
        batch_boxes = boxes[batch_idx][batch_mask]
        batch_scores = max_scores[batch_idx][batch_mask]
        batch_classes = class_ids[batch_idx][batch_mask]
        
        # Convert cxcywh to xyxy
        x1 = batch_boxes[:, 0] - batch_boxes[:, 2] / 2
        y1 = batch_boxes[:, 1] - batch_boxes[:, 3] / 2
        x2 = batch_boxes[:, 0] + batch_boxes[:, 2] / 2
        y2 = batch_boxes[:, 1] + batch_boxes[:, 3] / 2
        batch_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # Simple NMS
        keep = torchvision_nms(batch_boxes_xyxy, batch_scores, iou_thres)
        
        results.append({
            'boxes': batch_boxes_xyxy[keep].numpy(),
            'scores': batch_scores[keep].numpy(),
            'classes': batch_classes[keep].numpy()
        })
    
    return results


def torchvision_nms(boxes, scores, iou_threshold):
    """Simple NMS implementation."""
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    _, order = scores.sort(descending=True)
    keep = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0].item()
        keep.append(i)
        
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long)


def visualize(image, our_results, ultra_results, save_path):
    """Create side-by-side visualization."""
    from PIL import Image, ImageDraw
    
    COCO_NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    h, w = image.shape[:2]
    
    # Create side-by-side image
    combined = Image.new('RGB', (w * 2 + 10, h + 40), (255, 255, 255))
    
    # Our results
    img_our = Image.fromarray(image.copy())
    draw_our = ImageDraw.Draw(img_our)
    
    if len(our_results['boxes']) > 0:
        for box, score, cls in zip(our_results['boxes'], our_results['scores'], our_results['classes']):
            x1, y1, x2, y2 = box
            label = f"{COCO_NAMES[int(cls)]}: {score:.2f}"
            draw_our.rectangle([x1, y1, x2, y2], outline='green', width=2)
            draw_our.text((x1, max(0, y1 - 15)), label, fill='green')
    
    # Ultra results
    img_ultra = Image.fromarray(image.copy())
    draw_ultra = ImageDraw.Draw(img_ultra)
    
    if ultra_results and len(ultra_results.get('boxes', [])) > 0:
        for box, score, cls in zip(ultra_results['boxes'], ultra_results['scores'], ultra_results['classes']):
            x1, y1, x2, y2 = box
            label = f"{COCO_NAMES[int(cls)]}: {score:.2f}"
            draw_ultra.rectangle([x1, y1, x2, y2], outline='blue', width=2)
            draw_ultra.text((x1, max(0, y1 - 15)), label, fill='blue')
    
    combined.paste(img_our, (0, 40))
    combined.paste(img_ultra, (w + 10, 40))
    
    # Add titles
    draw = ImageDraw.Draw(combined)
    draw.text((w // 2 - 50, 10), "Our Model", fill='green')
    draw.text((w + 10 + w // 2 - 50, 10), "Ultralytics", fill='blue')
    
    combined.save(save_path)
    print(f"Saved comparison to: {save_path}")


def main():
    print("=" * 70)
    print("Visual Comparison: Our Model vs Ultralytics")
    print("=" * 70)
    
    # Download test image
    img_path = download_test_image()
    print(f"Test image: {img_path}")
    
    # Load image
    input_tensor, orig_image, padding_info = load_image(img_path)
    print(f"Original size: {padding_info['orig_size']}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Padding: x={padding_info['pad_x']}, y={padding_info['pad_y']}, ratio={padding_info['ratio']:.4f}")
    
    # Load our model
    print("\nLoading our model...")
    our_model = yolo_v11_n(80, task='detect')
    checkpoint = torch.load(
        Path(__file__).parent.parent / "weights" / "open" / "yolo11n_open.pt",
        map_location='cpu'
    )
    our_model.load_state_dict(checkpoint['state_dict'])
    our_model.eval()
    
    # Run our model (boxes in 640x640 space)
    print("Running our model...")
    our_results_640 = run_detection(our_model, input_tensor)[0]
    print(f"  Detections (640x640): {len(our_results_640['boxes'])}")
    
    # Scale boxes to original image coordinates
    our_results = {
        'boxes': scale_boxes_to_original(our_results_640['boxes'], padding_info),
        'scores': our_results_640['scores'],
        'classes': our_results_640['classes']
    }
    print(f"  Detections (original): {len(our_results['boxes'])}")
    
    # Load Ultralytics model
    ultra_results = None
    try:
        from ultralytics import YOLO
        print("\nLoading Ultralytics model...")
        ultra_model = YOLO(str(Path(__file__).parent.parent / "weights" / "ultralytics" / "yolo11n.pt"))
        
        # Run on original image path (Ultralytics handles preprocessing internally)
        print("Running Ultralytics model...")
        results = ultra_model.predict(str(img_path), verbose=False, conf=0.25)[0]
        
        if results.boxes is not None and len(results.boxes) > 0:
            ultra_results = {
                'boxes': results.boxes.xyxy.cpu().numpy(),
                'scores': results.boxes.conf.cpu().numpy(),
                'classes': results.boxes.cls.cpu().numpy().astype(int)
            }
            print(f"  Detections: {len(ultra_results['boxes'])}")
    except ImportError:
        print("  Ultralytics not available")
    
    # Print detection comparison
    COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush']
    
    print("\n" + "=" * 70)
    print("Detection Results (Original Image Coordinates)")
    print("=" * 70)
    
    print("\nOur Model:")
    for i, (box, score, cls) in enumerate(zip(our_results['boxes'][:5], 
                                               our_results['scores'][:5], 
                                               our_results['classes'][:5])):
        print(f"  {COCO_NAMES[int(cls)]}: {score:.3f} @ [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    if ultra_results:
        print("\nUltralytics:")
        for i, (box, score, cls) in enumerate(zip(ultra_results['boxes'][:5], 
                                                   ultra_results['scores'][:5], 
                                                   ultra_results['classes'][:5])):
            print(f"  {COCO_NAMES[int(cls)]}: {score:.3f} @ [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Save visualization
    save_path = Path(__file__).parent.parent / "data" / "comparison.jpg"
    visualize(orig_image, our_results, ultra_results, save_path)
    
    # Save detection results as JSON
    import json
    results_path = Path(__file__).parent.parent / "data" / "detections.json"
    
    detection_data = {
        'image': str(img_path),
        'image_size': list(padding_info['orig_size']),
        'our_model': {
            'num_detections': len(our_results['boxes']),
            'detections': [
                {
                    'class': COCO_NAMES[int(cls)],
                    'score': float(score),
                    'box': [float(x) for x in box]
                }
                for box, score, cls in zip(our_results['boxes'], our_results['scores'], our_results['classes'])
            ]
        }
    }
    
    if ultra_results:
        detection_data['ultralytics'] = {
            'num_detections': len(ultra_results['boxes']),
            'detections': [
                {
                    'class': COCO_NAMES[int(cls)],
                    'score': float(score),
                    'box': [float(x) for x in box]
                }
                for box, score, cls in zip(ultra_results['boxes'], ultra_results['scores'], ultra_results['classes'])
            ]
        }
    
    with open(results_path, 'w') as f:
        json.dump(detection_data, f, indent=2)
    print(f"Saved detections to: {results_path}")


if __name__ == '__main__':
    main()
