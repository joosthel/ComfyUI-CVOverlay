"""
CV Model Loader Node for ComfyUI
Loads YOLO models for object detection
Downloads to ComfyUI models folder following best practices
"""

import os
import folder_paths

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
            # Use custom path if provided
            if custom_model_path:
                model_path = custom_model_path
                print(f"Loading custom YOLO model: {model_path}")
                model = YOLO(model_path)
                print(f"✓ YOLO model loaded successfully: {model_path}")
                return (model,)
            
            # Use ComfyUI models folder structure
            models_dir = folder_paths.models_dir
            yolo_models_dir = os.path.join(models_dir, "yolo")
            
            # Create YOLO models directory if it doesn't exist
            os.makedirs(yolo_models_dir, exist_ok=True)
            
            model_path = os.path.join(yolo_models_dir, model_name)
            
            # If model doesn't exist in ComfyUI models folder, download and move it
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}")
                print(f"Downloading YOLO model {model_name}...")
                
                # Store current working directory
                original_cwd = os.getcwd()
                
                try:
                    # Change to ComfyUI models/yolo directory for download
                    os.chdir(yolo_models_dir)
                    
                    # Download model - it will go to current directory (models/yolo)
                    model = YOLO(model_name)
                    
                    # Check if model was downloaded to the current directory
                    if os.path.exists(model_name):
                        # Rename to full path
                        final_path = os.path.join(yolo_models_dir, model_name)
                        if not os.path.exists(final_path):
                            import shutil
                            shutil.move(model_name, final_path)
                        print(f"✓ Model downloaded to: {final_path}")
                    
                finally:
                    # Always restore original working directory
                    os.chdir(original_cwd)
                
                # If model still not in expected location, look for it and move it
                if not os.path.exists(model_path):
                    # Check common download locations
                    possible_locations = [
                        os.path.join(original_cwd, model_name),  # Current working directory
                        os.path.join(yolo_models_dir, model_name),  # Target directory
                    ]
                    
                    # Also check ultralytics cache directory
                    try:
                        from ultralytics.utils import WEIGHTS_DIR
                        possible_locations.append(os.path.join(WEIGHTS_DIR, model_name))
                    except:
                        pass
                    
                    source_found = None
                    for loc in possible_locations:
                        if os.path.exists(loc) and loc != model_path:
                            source_found = loc
                            break
                    
                    if source_found:
                        import shutil
                        shutil.move(source_found, model_path)
                        print(f"✓ Model moved from {source_found} to: {model_path}")
                    else:
                        print(f"Warning: Model downloaded but could not locate for moving to {model_path}")
                        # Return the model we already loaded
                        return (model,)
            else:
                print(f"Using existing YOLO model: {model_path}")
            
            # Load model from ComfyUI models directory
            model = YOLO(model_path)
            print(f"✓ YOLO model loaded successfully from: {model_path}")
            
            return (model,)
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(f"Error details: {error_msg}")
            raise RuntimeError(error_msg)
