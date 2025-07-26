"""
Utility functions for YOLO operations
"""

import torch
import numpy as np
from ultralytics import YOLO


def load_yolo_model(model_path):
    """Load YOLO model safely"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model from {model_path}: {str(e)}")


def filter_detections_by_class(detections, allowed_classes):
    """Filter detections by class names or IDs"""
    if not allowed_classes:
        return detections
    
    filtered = []
    for detection in detections:
        class_id = detection.get('class', -1)
        class_name = detection.get('class_name', '')
        
        if class_id in allowed_classes or class_name in allowed_classes:
            filtered.append(detection)
    
    return filtered


def filter_detections_by_confidence(detections, min_confidence):
    """Filter detections by confidence threshold"""
    return [d for d in detections if d.get('confidence', 0) >= min_confidence]


def filter_detections_by_area(detections, min_area=None, max_area=None):
    """Filter detections by bounding box area"""
    filtered = []
    
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
            
        filtered.append(detection)
    
    return filtered


def nms_detections(detections, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to detections"""
    if not detections:
        return detections
    
    # Convert to format for NMS
    boxes = []
    scores = []
    
    for detection in detections:
        bbox = detection['bbox']
        boxes.append(bbox)
        scores.append(detection.get('confidence', 1.0))
    
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    
    # Apply NMS
    keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    
    # Return filtered detections
    return [detections[i] for i in keep_indices]


def get_yolo_class_names():
    """Get standard YOLO class names"""
    return [
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


def convert_yolo_to_detection_format(results, img_width, img_height):
    """Convert YOLO results to standardized detection format"""
    detections = []
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        for i in range(len(boxes)):
            # Get box coordinates
            box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            
            # Get class name if available
            class_names = get_yolo_class_names()
            class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
            
            # Create detection dictionary
            detection = {
                'bbox': box.tolist(),
                'confidence': conf,
                'class': cls,
                'class_name': class_name,
                'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                'width': box[2] - box[0],
                'height': box[3] - box[1],
                'area': (box[2] - box[0]) * (box[3] - box[1])
            }
            
            detections.append(detection)
    
    return detections
