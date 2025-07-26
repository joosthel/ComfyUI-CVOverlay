"""
CV Model Loader Node for ComfyUI
Loads YOLO models for object detection
"""

import os

# Handle missing dependencies gracefully
try:
    import torch
    from ultralytics import YOLO
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)


class CV_ModelLoader:
    """Loads YOLO models for computer vision tasks"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"], {
                    "default": "yolov8n.pt"
                }),
            },
            "optional": {
                "custom_model_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("CV_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "CV/Models"
    
    def load_model(self, model_name, custom_model_path=""):
        """Load YOLO model"""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(f"Missing dependencies: {MISSING_DEPS}. Please install requirements: pip install torch ultralytics")
        
        try:
            # Use custom path if provided, otherwise use default model
            model_path = custom_model_path if custom_model_path else model_name
            
            # Load YOLO model
            model = YOLO(model_path)
            
            return (model,)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
