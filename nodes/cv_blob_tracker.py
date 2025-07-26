"""
CV Blob Tracker Node for ComfyUI
Tracks moving objects/blobs across video frames using OpenCV
"""

# Handle missing dependencies gracefully
try:
    import cv2
    import torch
    import numpy as np
    from PIL import Image
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)


class CV_BlobTracker:
    """Tracks blobs/objects across frames using OpenCV algorithms"""
    
    def __init__(self):
        self.prev_frame = None
        self.tracker_type = None
        self.trackers = []
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tracking_method": (["background_subtraction", "optical_flow", "contour_tracking"], {
                    "default": "background_subtraction"
                }),
                "min_area": ("INT", {
                    "default": 500,
                    "min": 50,
                    "max": 10000,
                    "step": 50
                }),
                "threshold": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 255,
                    "step": 5
                }),
            },
            "optional": {
                "previous_tracks": ("CV_TRACKS",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "CV_TRACKS")
    RETURN_NAMES = ("image", "tracks")
    FUNCTION = "track_blobs"
    CATEGORY = "CV/Tracking"
    
    def track_blobs(self, image, tracking_method, min_area, threshold, previous_tracks=None):
        """Track blobs in the current frame"""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(f"Missing dependencies: {MISSING_DEPS}. Please install requirements: pip install opencv-python torch")
        
        try:
            # Convert ComfyUI image tensor to OpenCV format
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image[0]  # Take first image from batch
                img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = np.array(image)
            
            # Convert to grayscale for processing
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            
            tracks = []
            
            if tracking_method == "background_subtraction":
                tracks = self._background_subtraction_tracking(gray, min_area, threshold)
            elif tracking_method == "optical_flow":
                tracks = self._optical_flow_tracking(gray, previous_tracks)
            elif tracking_method == "contour_tracking":
                tracks = self._contour_tracking(gray, min_area, threshold)
            
            # Store current frame for next iteration
            self.prev_frame = gray.copy()
            
            # Convert back to ComfyUI tensor format
            if isinstance(image, torch.Tensor):
                output_image = image
            else:
                img_tensor = torch.from_numpy(img_np).float() / 255.0
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                output_image = img_tensor
            
            return (output_image, tracks)
            
        except Exception as e:
            raise RuntimeError(f"Blob tracking failed: {str(e)}")
    
    def _background_subtraction_tracking(self, gray, min_area, threshold):
        """Simple background subtraction tracking"""
        tracks = []
        
        if self.prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(self.prev_frame, gray)
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    tracks.append({
                        'bbox': [x, y, x + w, y + h],
                        'center': [center_x, center_y],
                        'area': area,
                        'confidence': min(area / (min_area * 10), 1.0)
                    })
        
        return tracks
    
    def _optical_flow_tracking(self, gray, previous_tracks):
        """Lucas-Kanade optical flow tracking"""
        tracks = []
        
        if self.prev_frame is not None and previous_tracks:
            # Extract previous points
            prev_points = []
            for track in previous_tracks:
                if 'center' in track:
                    prev_points.append([track['center']])
            
            if prev_points:
                prev_points = np.array(prev_points, dtype=np.float32)
                
                # Calculate optical flow
                new_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame, gray, prev_points, None
                )
                
                # Keep only good points
                good_new = new_points[status == 1]
                
                for i, point in enumerate(good_new):
                    x, y = point.ravel()
                    tracks.append({
                        'center': [int(x), int(y)],
                        'confidence': 0.8,
                        'bbox': [int(x-20), int(y-20), int(x+20), int(y+20)]  # Approximate bbox
                    })
        
        return tracks
    
    def _contour_tracking(self, gray, min_area, threshold):
        """Simple contour-based tracking"""
        tracks = []
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                tracks.append({
                    'bbox': [x, y, x + w, y + h],
                    'center': [center_x, center_y],
                    'area': area,
                    'confidence': min(area / (min_area * 5), 1.0)
                })
        
        return tracks
