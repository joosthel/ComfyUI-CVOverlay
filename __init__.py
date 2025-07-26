"""
ComfyUI-CVOverlay: Minimal OpenCV and YOLOv8 integration for ComfyUI
Clean architecture with unified art direction
"""

from .nodes.cv_model_loader import CV_ModelLoader
from .nodes.cv_object_detector import CV_ObjectDetector
from .nodes.cv_blob_tracker import CV_BlobTracker
from .nodes.cv_aesthetic_overlay import CV_AestheticOverlay

NODE_CLASS_MAPPINGS = {
    "CV_ModelLoader": CV_ModelLoader,
    "CV_ObjectDetector": CV_ObjectDetector,
    "CV_BlobTracker": CV_BlobTracker,
    "CV_AestheticOverlay": CV_AestheticOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CV_ModelLoader": "CV Model Loader",
    "CV_ObjectDetector": "CV Object Detector",
    "CV_BlobTracker": "CV Blob Tracker",
    "CV_AestheticOverlay": "CV Aesthetic Overlay",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]