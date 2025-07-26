"""
CV Object Detector Node for ComfyUI
Clean object detection that outputs detection data only
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
    """Clean object detection without visual rendering"""
    
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
            }
        }
    
    RETURN_TYPES = ("IMAGE", "CV_DETECTIONS")
    RETURN_NAMES = ("image", "detections")
    FUNCTION = "detect_objects"
    CATEGORY = "CV/Detection"
    
    def detect_objects(self, model, image, confidence):
        """Perform object detection and return data"""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(f"Missing dependencies: {MISSING_DEPS}")
        
        try:
            # Handle batch processing for video frames
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    # Batch of images [batch_size, height, width, channels]
                    batch_size = image.shape[0]
                    all_detections = []
                    
                    for i in range(batch_size):
                        # Process each frame in the batch
                        frame = image[i]  # Single frame [height, width, channels]
                        frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                        frame_pil = Image.fromarray(frame_np)
                        
                        # Run inference on this frame
                        results = model(frame_pil, conf=confidence)
                        
                        # Extract detection data for this frame
                        frame_detections = []
                        if len(results) > 0 and results[0].boxes is not None:
                            boxes = results[0].boxes
                            for j in range(len(boxes)):
                                box = boxes.xyxy[j].cpu().numpy()  # [x1, y1, x2, y2]
                                conf = boxes.conf[j].cpu().numpy()
                                cls = int(boxes.cls[j].cpu().numpy())
                                class_name = model.names[cls] if hasattr(model, 'names') else str(cls)
                                
                                frame_detections.append({
                                    'bbox': box.tolist(),
                                    'confidence': float(conf),
                                    'class': cls,
                                    'class_name': class_name
                                })
                        
                        all_detections.append(frame_detections)
                    
                    # Return original image batch + detections for all frames
                    return (image, all_detections)
                    
                else:
                    # Single image [height, width, channels]
                    img_np = (image.cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)
            else:
                img_pil = image
            
            # Process single image (non-batch case)
            results = model(img_pil, conf=confidence)
            
            # Extract detection data
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    class_name = model.names[cls] if hasattr(model, 'names') else str(cls)
                    
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class': cls,
                        'class_name': class_name
                    })
            
            # Return original image unchanged + detection data
            return (image, detections)
            
        except Exception as e:
            raise RuntimeError(f"Object detection failed: {str(e)}")


# Register the node
NODE_CLASS_MAPPINGS = {
    "CV_ObjectDetector": CV_ObjectDetector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CV_ObjectDetector": "CV Object Detector"
}
