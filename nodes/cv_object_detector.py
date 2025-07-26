"""
CV Object Detector Node for ComfyUI
Performs object detection using YOLO models
"""

# Handle missing dependencies gracefully
try:
    import torch
    import numpy as np
    from PIL import Image
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)


class CV_ObjectDetector:
    """Detects objects in images using YOLO models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CV_MODEL",),
                "image": ("IMAGE",),
                "confidence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "CV_DETECTIONS")
    RETURN_NAMES = ("image", "detections")
    FUNCTION = "detect_objects"
    CATEGORY = "CV/Detection"
    
    def detect_objects(self, model, image, confidence, iou_threshold):
        """Perform object detection on input image"""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(f"Missing dependencies: {MISSING_DEPS}. Please install requirements: pip install torch ultralytics opencv-python")
        
        try:
            # Convert ComfyUI image tensor to PIL Image
            if isinstance(image, torch.Tensor):
                # ComfyUI images are in format (batch, height, width, channels)
                if image.dim() == 4:
                    image = image[0]  # Take first image from batch
                # Convert from tensor to numpy
                img_np = (image.cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
            else:
                img_pil = image
            
            # Run inference
            results = model(img_pil, conf=confidence, iou=iou_threshold)
            
            # Extract detection data
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class': cls,
                        'class_name': model.names[cls] if hasattr(model, 'names') else str(cls)
                    })
            
            # Convert back to ComfyUI tensor format
            if isinstance(image, torch.Tensor):
                output_image = image
            else:
                # Convert PIL to tensor if needed
                img_tensor = torch.from_numpy(np.array(img_pil)).float() / 255.0
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                output_image = img_tensor
            
            return (output_image, detections)
            
        except Exception as e:
            raise RuntimeError(f"Object detection failed: {str(e)}")
