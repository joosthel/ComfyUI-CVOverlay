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
            
            print(f"Loading YOLO model: {model_path}")
            
            # Load YOLO model - this will auto-download if needed
            if not custom_model_path:
                print(f"Note: {model_name} will be downloaded automatically if not present")
            
            model = YOLO(model_path)
            print(f"✓ YOLO model loaded successfully: {model_path}")
            
            return (model,)
            
        except Exception as e:
            error_msg = f"Failed to load model '{model_path}': {str(e)}"
            if "No such file or directory" in str(e) and not custom_model_path:
                error_msg += f"\nNote: The model {model_name} should download automatically. Check your internet connection."
            raise RuntimeError(error_msg)
