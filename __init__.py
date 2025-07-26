"""
ComfyUI OpenCV Overlays - Custom nodes for computer vision effects
"""

import os
import sys
import importlib.util

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
nodes_dir = os.path.join(current_dir, "nodes")

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if nodes_dir not in sys.path:
    sys.path.insert(0, nodes_dir)

# Import nodes with error handling using direct file imports
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def import_node_from_file(file_path, class_name):
    """Import a class from a specific file"""
    try:
        spec = importlib.util.spec_from_file_location("temp_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    except Exception as e:
        print(f"Failed to import {class_name} from {file_path}: {e}")
        return None

# Import CV_ModelLoader
cv_model_loader_path = os.path.join(nodes_dir, "cv_model_loader.py")
if os.path.exists(cv_model_loader_path):
    CV_ModelLoader = import_node_from_file(cv_model_loader_path, "CV_ModelLoader")
    if CV_ModelLoader:
        NODE_CLASS_MAPPINGS["CV_ModelLoader"] = CV_ModelLoader
        NODE_DISPLAY_NAME_MAPPINGS["CV_ModelLoader"] = "CV Model Loader"

# Import CV_ObjectDetector
cv_object_detector_path = os.path.join(nodes_dir, "cv_object_detector.py")
if os.path.exists(cv_object_detector_path):
    CV_ObjectDetector = import_node_from_file(cv_object_detector_path, "CV_ObjectDetector")
    if CV_ObjectDetector:
        NODE_CLASS_MAPPINGS["CV_ObjectDetector"] = CV_ObjectDetector
        NODE_DISPLAY_NAME_MAPPINGS["CV_ObjectDetector"] = "CV Object Detector"

# Import CV_BlobTracker
cv_blob_tracker_path = os.path.join(nodes_dir, "cv_blob_tracker.py")
if os.path.exists(cv_blob_tracker_path):
    CV_BlobTracker = import_node_from_file(cv_blob_tracker_path, "CV_BlobTracker")
    if CV_BlobTracker:
        NODE_CLASS_MAPPINGS["CV_BlobTracker"] = CV_BlobTracker
        NODE_DISPLAY_NAME_MAPPINGS["CV_BlobTracker"] = "CV Blob Tracker"

# Import CV_AestheticOverlay
cv_aesthetic_overlay_path = os.path.join(nodes_dir, "cv_aesthetic_overlay.py")
if os.path.exists(cv_aesthetic_overlay_path):
    CV_AestheticOverlay = import_node_from_file(cv_aesthetic_overlay_path, "CV_AestheticOverlay")
    if CV_AestheticOverlay:
        NODE_CLASS_MAPPINGS["CV_AestheticOverlay"] = CV_AestheticOverlay
        NODE_DISPLAY_NAME_MAPPINGS["CV_AestheticOverlay"] = "CV Aesthetic Overlay"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"ComfyUI-CVOverlay loaded {len(NODE_CLASS_MAPPINGS)} nodes successfully")