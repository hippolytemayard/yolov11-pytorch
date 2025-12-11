#!/usr/bin/env python3
"""
YOLO v11 Inference Script.

Usage:
    python inference.py --source image.jpg --weights weights/best.pt
    python inference.py --source video.mp4 --weights weights/yolo11n.pt
    python inference.py --source 0 --weights weights/yolo11s.pt  # webcam
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo.models import yolo_v11_n, yolo_v11_s, yolo_v11_m, yolo_v11_l, yolo_v11_x
from yolo.utils.boxes import non_max_suppression, scale_boxes
from yolo.data.augmentations import letterbox


# COCO class names
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

# Color palette for visualization
COLORS = np.random.default_rng(42).uniform(0, 255, (len(COCO_NAMES), 3)).astype(np.uint8)


class YOLOPredictor:
    """
    YOLO predictor for inference.
    
    Args:
        weights: Path to model weights
        device: Device to run inference on
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        input_size: Input image size
    """
    
    def __init__(
        self,
        weights: str,
        device: str = 'cuda',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        # Load model
        self.model = self._load_model(weights)
        self.model.eval()
        
        # Warmup
        self._warmup()

    def _load_model(self, weights: str) -> torch.nn.Module:
        """Load model from weights file."""
        print(f"Loading model from {weights}")
        
        checkpoint = torch.load(weights, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model'].float()
        else:
            model = checkpoint.float()
        
        model = model.to(self.device)
        model = model.fuse() if hasattr(model, 'fuse') else model
        
        return model

    def _warmup(self) -> None:
        """Warmup model with dummy input."""
        dummy = torch.zeros(1, 3, self.input_size, self.input_size).to(self.device)
        for _ in range(3):
            self.model(dummy)
        print("Model warmed up")

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[float, float]]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Tuple of (preprocessed tensor, original shape, ratio/pad)
        """
        orig_shape = image.shape[:2]
        
        # Letterbox resize
        img, ratio, pad = letterbox(image, self.input_size)
        
        # Convert to tensor
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().to(self.device)
        img = img.unsqueeze(0) / 255.0
        
        return img, orig_shape, (ratio[0], pad)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> List[dict]:
        """
        Run prediction on an image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of detections with boxes, scores, and classes
        """
        # Preprocess
        img, orig_shape, (ratio, pad) = self.preprocess(image)
        
        # Inference
        t0 = time.time()
        outputs = self.model(img)
        t1 = time.time()
        
        # Post-process
        results = non_max_suppression(
            outputs,
            self.conf_threshold,
            self.iou_threshold
        )
        
        detections = []
        for det in results:
            if len(det):
                # Scale boxes back to original image
                det[:, :4] = scale_boxes(
                    det[:, :4],
                    (self.input_size, self.input_size),
                    orig_shape,
                    ratio_pad=(ratio, pad)
                )
                
                for *box, conf, cls in det.cpu().numpy():
                    detections.append({
                        'box': [int(x) for x in box],
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': COCO_NAMES[int(cls)] if int(cls) < len(COCO_NAMES) else f'class_{int(cls)}'
                    })
        
        inference_time = (t1 - t0) * 1000
        
        return detections, inference_time

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[dict],
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection results on image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            line_thickness: Line thickness for boxes
            
        Returns:
            Image with drawn detections
        """
        img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls = det['class']
            conf = det['confidence']
            name = det['class_name']
            
            color = tuple(int(c) for c in COLORS[cls % len(COLORS)])
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
            
            # Draw label
            label = f'{name} {conf:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img


def process_image(predictor: YOLOPredictor, image_path: str, output_path: Optional[str] = None) -> None:
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    detections, inference_time = predictor.predict(image)
    
    print(f"Image: {image_path}")
    print(f"Inference time: {inference_time:.1f}ms")
    print(f"Detections: {len(detections)}")
    
    for det in detections:
        print(f"  {det['class_name']}: {det['confidence']:.2f} at {det['box']}")
    
    # Draw and save
    result = predictor.draw_detections(image, detections)
    
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Saved to: {output_path}")
    else:
        cv2.imshow('YOLO Detection', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(predictor: YOLOPredictor, video_path: str, output_path: Optional[str] = None) -> None:
    """Process a video file or webcam."""
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections, inference_time = predictor.predict(frame)
        total_time += inference_time
        frame_count += 1
        
        result = predictor.draw_detections(frame, detections)
        
        # Add FPS counter
        avg_fps = 1000 * frame_count / total_time if total_time > 0 else 0
        cv2.putText(result, f'FPS: {avg_fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if writer:
            writer.write(result)
        else:
            cv2.imshow('YOLO Detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if writer:
        writer.release()
        print(f"Saved to: {output_path}")
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")
    print(f"Average FPS: {1000 * frame_count / total_time:.1f}")


def main():
    parser = argparse.ArgumentParser(description='YOLO v11 Inference')
    
    parser.add_argument('--source', required=True, help='Image/video path or camera index')
    parser.add_argument('--weights', required=True, help='Model weights path')
    parser.add_argument('--output', default=None, help='Output path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--size', type=int, default=640, help='Input size')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = YOLOPredictor(
        weights=args.weights,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.size
    )
    
    source = args.source
    
    # Determine source type
    if source.isdigit() or source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(predictor, source, args.output)
    else:
        process_image(predictor, source, args.output)


if __name__ == '__main__':
    main()

