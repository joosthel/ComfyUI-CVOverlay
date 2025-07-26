"""
CV Blob Tracker Node for ComfyUI
Clean blob tracking that outputs blob data only
"""

# Handle missing dependencies gracefully
try:
    import cv2
    import torch
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)


class CV_BlobTracker:
    """Clean blob tracking without visual rendering"""
    
    # Class-level storage for video tracking
    _video_states = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # TouchDesigner-style parameters
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "min_area": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 10000,
                    "step": 10
                }),
                "max_area": ("INT", {
                    "default": 5000,
                    "min": 100,
                    "max": 50000,
                    "step": 100
                }),
                "blur_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 21,
                    "step": 2
                }),
                "detection_mode": (["bright_blobs", "dark_blobs", "motion_blobs"], {
                    "default": "bright_blobs"
                }),
            },
            "optional": {
                "video_id": ("STRING", {
                    "default": "default",
                    "tooltip": "Unique ID for video tracking"
                }),
                "reset_tracking": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "CV_BLOBS")
    RETURN_NAMES = ("image", "blobs")
    FUNCTION = "track_blobs"
    CATEGORY = "CV/Tracking"
    
    def track_blobs(self, image, threshold, min_area, max_area, blur_size, detection_mode, 
                   video_id="default", reset_tracking=False):
        """Track blobs and return data"""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(f"Missing dependencies: {MISSING_DEPS}")
        
        # Reset tracking state if requested
        if reset_tracking and video_id in self._video_states:
            del self._video_states[video_id]
        
        try:
            # Handle batch processing for video frames
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    # Batch of images [batch_size, height, width, channels]
                    batch_size = image.shape[0]
                    all_blobs = []
                    
                    for i in range(batch_size):
                        # Process each frame in the batch
                        frame = image[i]  # Single frame [height, width, channels]
                        frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                        
                        # Convert to grayscale for processing
                        if len(frame_np.shape) == 3:
                            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
                        else:
                            gray = frame_np
                        
                        # Detect blobs for this frame
                        frame_blobs = self._detect_blobs(gray, threshold, min_area, max_area, 
                                                       blur_size, detection_mode, f"{video_id}_frame_{i}")
                        all_blobs.append(frame_blobs)
                    
                    # Return original image batch + blob data for all frames
                    return (image, all_blobs)
                    
                else:
                    # Single image [height, width, channels]
                    img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = np.array(image)
            
            # Process single image (non-batch case)
            # Convert to grayscale for processing
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            
            # Detect blobs
            blobs = self._detect_blobs(gray, threshold, min_area, max_area, blur_size, 
                                     detection_mode, video_id)
            
            # Return original image unchanged + blob data
            return (image, blobs)
            
        except Exception as e:
            raise RuntimeError(f"Blob tracking failed: {str(e)}")
    
    def _detect_blobs(self, gray, threshold, min_area, max_area, blur_size, mode, video_id):
        """Detect blobs using TouchDesigner-style parameters"""
        blobs = []
        
        # Ensure blur_size is odd
        if blur_size % 2 == 0:
            blur_size += 1
        
        # Apply blur
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        if mode == "bright_blobs":
            thresh_val = int(threshold * 255)
            _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
            
        elif mode == "dark_blobs":
            thresh_val = int(threshold * 255)
            _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
        elif mode == "motion_blobs":
            if video_id not in self._video_states:
                self._video_states[video_id] = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            
            back_sub = self._video_states[video_id]
            binary = back_sub.apply(blurred)
        
        # Clean up binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract blob data
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                # Calculate blob properties
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    center_x = int(moments['m10'] / moments['m00'])
                    center_y = int(moments['m01'] / moments['m00'])
                else:
                    center_x, center_y = 0, 0
                
                # Bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                blob_data = {
                    'id': i,
                    'center': [center_x, center_y],
                    'bbox': [x, y, x + w, y + h],
                    'area': area,
                    'contour': contour.tolist(),
                    'width': w,
                    'height': h
                }
                blobs.append(blob_data)
        
        return blobs


# Register the node
NODE_CLASS_MAPPINGS = {
    "CV_BlobTracker": CV_BlobTracker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CV_BlobTracker": "CV Blob Tracker"
}
