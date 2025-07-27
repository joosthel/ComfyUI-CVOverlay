"""
CV Blob Tracker Node for ComfyUI
Professional TouchDesigner-style blob tracking for surveillance aesthetic
"""

# Handle missing dependencies gracefully
try:
    import cv2
    import torch
    import numpy as np
    from scipy.spatial.distance import cdist
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)


class CV_BlobTracker:
    """Professional TouchDesigner-style blob tracking with persistent tracking"""
    
    # Class-level storage for video tracking
    _video_states = {}
    _tracked_blobs = {}
    _next_blob_id = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Core detection parameters
                "detection_mode": (["motion", "bright_regions", "dark_regions", "edge_density", "color_variance"], {
                    "default": "bright_regions"  # More reliable default than motion
                }),
                "sensitivity": ("FLOAT", {
                    "default": 0.5,  # Higher default for better detection
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Detection sensitivity"
                }),
                # Size filtering
                "min_size": ("INT", {
                    "default": 100,  # Larger minimum for better stability
                    "min": 10,
                    "max": 5000,
                    "step": 10
                }),
                "max_size": ("INT", {
                    "default": 5000,  # Larger maximum for more flexibility
                    "min": 100,
                    "max": 20000,
                    "step": 50
                }),
                # Processing parameters
                "blur_amount": ("INT", {
                    "default": 5,  # Better noise reduction
                    "min": 1,
                    "max": 15,
                    "step": 2
                }),
                "noise_reduction": ("FLOAT", {
                    "default": 0.3,  # More noise reduction by default
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                # Tracking parameters
                "max_tracking_distance": ("INT", {
                    "default": 80,  # Larger tracking distance
                    "min": 10,
                    "max": 200,
                    "step": 5,
                    "tooltip": "Max pixel distance for tracking same blob"
                }),
                "track_persistence": ("INT", {
                    "default": 15,  # Longer persistence
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "tooltip": "Frames to keep tracking lost blobs"
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
    
    RETURN_TYPES = ("IMAGE", "CV_BLOBS", "CV_TRACKS")
    RETURN_NAMES = ("image", "blobs", "tracks")
    FUNCTION = "track_blobs"
    CATEGORY = "CV/Tracking"
    
    def track_blobs(self, image, detection_mode, sensitivity, min_size, max_size, 
                   blur_amount, noise_reduction, max_tracking_distance, track_persistence,
                   video_id="default", reset_tracking=False):
        """Advanced blob tracking with persistence and trajectory analysis"""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(f"Missing dependencies: {MISSING_DEPS}")
        
        # Reset tracking state if requested
        if reset_tracking:
            if video_id in self._video_states:
                del self._video_states[video_id]
            if video_id in self._tracked_blobs:
                del self._tracked_blobs[video_id]
            if video_id in self._next_blob_id:
                del self._next_blob_id[video_id]
        
        # Initialize tracking state
        if video_id not in self._tracked_blobs:
            self._tracked_blobs[video_id] = {}
            self._next_blob_id[video_id] = 0
        
        try:
            # Handle batch processing for video frames
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    # Batch of images [batch_size, height, width, channels]
                    batch_size = image.shape[0]
                    all_blobs = []
                    all_tracks = []
                    
                    for i in range(batch_size):
                        # Process each frame in the batch
                        frame = image[i]  # Single frame [height, width, channels]
                        frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                        
                        # Convert to grayscale for processing
                        if len(frame_np.shape) == 3:
                            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
                        else:
                            gray = frame_np
                        
                        # Detect and track blobs for this frame
                        frame_blobs, frame_tracks = self._process_frame(
                            gray, detection_mode, sensitivity, min_size, max_size,
                            blur_amount, noise_reduction, max_tracking_distance, 
                            track_persistence, f"{video_id}_frame_{i}"
                        )
                        all_blobs.append(frame_blobs)
                        all_tracks.append(frame_tracks)
                    
                    # Return original image batch + blob data + track data for all frames
                    return (image, all_blobs, all_tracks)
                    
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
            
            # Detect and track blobs
            blobs, tracks = self._process_frame(
                gray, detection_mode, sensitivity, min_size, max_size,
                blur_amount, noise_reduction, max_tracking_distance, 
                track_persistence, video_id
            )
            
            # Return original image unchanged + blob data + track data
            return (image, blobs, tracks)
            
        except Exception as e:
            raise RuntimeError(f"Blob tracking failed: {str(e)}")
            
    def _process_frame(self, gray, detection_mode, sensitivity, min_size, max_size,
                      blur_amount, noise_reduction, max_tracking_distance, 
                      track_persistence, video_id):
        """Process a single frame for blob detection and tracking"""
        
        # Ensure blur_amount is odd
        if blur_amount % 2 == 0:
            blur_amount += 1
        
        # Pre-processing
        processed = cv2.GaussianBlur(gray, (blur_amount, blur_amount), 0)
        
        # Advanced detection based on mode
        if detection_mode == "motion":
            binary = self._detect_motion(processed, sensitivity, video_id)
        elif detection_mode == "bright_regions":
            binary = self._detect_bright_regions(processed, sensitivity)
        elif detection_mode == "dark_regions":
            binary = self._detect_dark_regions(processed, sensitivity)
        elif detection_mode == "edge_density":
            binary = self._detect_edge_density(processed, sensitivity)
        elif detection_mode == "color_variance":
            binary = self._detect_color_variance(gray, sensitivity)  # Use original for color
        
        # Noise reduction
        if noise_reduction > 0:
            kernel_size = max(3, int(noise_reduction * 10))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract current frame blobs
        current_blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_size <= area <= max_size:
                # Calculate enhanced blob properties
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    center_x = int(moments['m10'] / moments['m00'])
                    center_y = int(moments['m01'] / moments['m00'])
                else:
                    continue
                
                # Bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate additional properties for better tracking
                perimeter = cv2.arcLength(contour, True)
                aspect_ratio = w / h if h > 0 else 1.0
                extent = area / (w * h) if w * h > 0 else 0.0
                solidity = area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0.0
                
                blob_data = {
                    'center': [center_x, center_y],
                    'bbox': [x, y, x + w, y + h],
                    'area': area,
                    'perimeter': perimeter,
                    'aspect_ratio': aspect_ratio,
                    'extent': extent,
                    'solidity': solidity,
                    'contour': contour.tolist(),
                    'width': w,
                    'height': h,
                    'confidence': min(1.0, area / max_size)  # Normalized confidence
                }
                current_blobs.append(blob_data)
        
        # Track blobs across frames
        tracked_blobs, tracks = self._update_tracking(
            current_blobs, max_tracking_distance, track_persistence, video_id
        )
        
        return tracked_blobs, tracks
    
    def _detect_motion(self, gray, sensitivity, video_id):
        """Advanced motion detection for surveillance-style tracking"""
        if video_id not in self._video_states:
            self._video_states[video_id] = {
                'bg_subtractor': cv2.createBackgroundSubtractorMOG2(
                    detectShadows=False, 
                    varThreshold=16 * sensitivity,
                    history=100
                ),
                'prev_frame': None
            }
        
        state = self._video_states[video_id]
        
        # Background subtraction
        fg_mask = state['bg_subtractor'].apply(gray)
        
        # Frame differencing for additional motion info
        if state['prev_frame'] is not None:
            frame_diff = cv2.absdiff(gray, state['prev_frame'])
            _, diff_thresh = cv2.threshold(frame_diff, int(30 * sensitivity), 255, cv2.THRESH_BINARY)
            
            # Combine background subtraction with frame differencing
            combined = cv2.bitwise_or(fg_mask, diff_thresh)
        else:
            combined = fg_mask
        
        state['prev_frame'] = gray.copy()
        return combined
    
    def _detect_bright_regions(self, gray, sensitivity):
        """Detect bright regions with adaptive thresholding"""
        # Calculate adaptive threshold based on image statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        threshold_val = min(255, max(0, int(mean_val + std_val * (2.0 - sensitivity))))
        
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        return binary
    
    def _detect_dark_regions(self, gray, sensitivity):
        """Detect dark regions with adaptive thresholding"""
        # Calculate adaptive threshold based on image statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        threshold_val = max(0, min(255, int(mean_val - std_val * (2.0 - sensitivity))))
        
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
        return binary
    
    def _detect_edge_density(self, gray, sensitivity):
        """Detect regions with high edge density"""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Create kernel for edge density calculation
        kernel_size = max(3, int(20 * (1.0 - sensitivity)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Calculate edge density
        edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
        
        # Threshold based on sensitivity
        threshold_val = int(255 * sensitivity * 0.3)
        _, binary = cv2.threshold(edge_density.astype(np.uint8), threshold_val, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def _detect_color_variance(self, gray, sensitivity):
        """Detect regions with high local variance"""
        # Calculate local variance using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance_map = np.abs(laplacian)
        
        # Normalize and threshold
        variance_norm = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        threshold_val = int(255 * sensitivity * 0.4)
        _, binary = cv2.threshold(variance_norm, threshold_val, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def _update_tracking(self, current_blobs, max_distance, persistence, video_id):
        """Update blob tracking with persistence and trajectory calculation"""
        # Ensure video_id is initialized in tracking dictionaries
        if video_id not in self._tracked_blobs:
            self._tracked_blobs[video_id] = {}
        if video_id not in self._next_blob_id:
            self._next_blob_id[video_id] = 0
            
        tracked_blobs = self._tracked_blobs[video_id]
        
        # Create distance matrix between current blobs and tracked blobs
        assigned_current = set()
        assigned_tracked = set()
        
        if tracked_blobs and current_blobs:
            current_centers = np.array([blob['center'] for blob in current_blobs])
            tracked_centers = np.array([blob['center'] for blob in tracked_blobs.values() if blob['active']])
            
            if len(tracked_centers) > 0:
                distances = cdist(current_centers, tracked_centers)
                
                # Hungarian algorithm would be ideal here, but using greedy matching for simplicity
                tracked_list = [(blob_id, blob) for blob_id, blob in tracked_blobs.items() if blob['active']]
                
                # Find best matches
                for i in range(len(current_blobs)):
                    if i in assigned_current:
                        continue
                    
                    best_match = None
                    best_distance = float('inf')
                    best_j = None
                    
                    for j, (blob_id, tracked_blob) in enumerate(tracked_list):
                        if j in assigned_tracked:
                            continue
                        
                        if distances[i, j] < max_distance and distances[i, j] < best_distance:
                            best_match = blob_id
                            best_distance = distances[i, j]
                            best_j = j
                    
                    if best_match is not None:
                        # Update existing blob
                        current_blobs[i]['id'] = best_match
                        current_blobs[i]['age'] = tracked_blobs[best_match]['age'] + 1
                        
                        # Update trajectory
                        if 'trajectory' not in tracked_blobs[best_match]:
                            tracked_blobs[best_match]['trajectory'] = []
                        tracked_blobs[best_match]['trajectory'].append(current_blobs[i]['center'])
                        
                        # Keep trajectory length manageable
                        if len(tracked_blobs[best_match]['trajectory']) > 30:
                            tracked_blobs[best_match]['trajectory'] = tracked_blobs[best_match]['trajectory'][-30:]
                        
                        # Update blob data
                        tracked_blobs[best_match].update(current_blobs[i])
                        tracked_blobs[best_match]['active'] = True
                        tracked_blobs[best_match]['lost_frames'] = 0
                        
                        assigned_current.add(i)
                        assigned_tracked.add(best_j)
        
        # Create new blobs for unassigned current detections
        for i, blob in enumerate(current_blobs):
            if i not in assigned_current:
                blob_id = self._next_blob_id[video_id]
                self._next_blob_id[video_id] += 1
                
                blob['id'] = blob_id
                blob['age'] = 1
                blob['active'] = True
                blob['lost_frames'] = 0
                blob['trajectory'] = [blob['center']]
                
                tracked_blobs[blob_id] = blob
        
        # Update lost blobs
        active_blob_ids = set([blob['id'] for blob in current_blobs if 'id' in blob])
        for blob_id, blob in tracked_blobs.items():
            if blob['active'] and blob_id not in active_blob_ids:
                blob['lost_frames'] += 1
                if blob['lost_frames'] > persistence:
                    blob['active'] = False
        
        # Prepare output
        active_blobs = [blob for blob in tracked_blobs.values() if blob['active']]
        
        # Generate track connections
        tracks = self._generate_tracks(tracked_blobs)
        
        return active_blobs, tracks
    
    def _generate_tracks(self, tracked_blobs):
        """Generate track lines between nearby blobs"""
        tracks = []
        active_blobs = [blob for blob in tracked_blobs.values() if blob['active'] and blob['age'] > 2]
        
        # Generate connections between nearby blobs
        for i, blob1 in enumerate(active_blobs):
            for j, blob2 in enumerate(active_blobs[i+1:], i+1):
                center1 = blob1['center']
                center2 = blob2['center']
                
                # Calculate distance
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Connect if within reasonable distance and similar properties
                max_connection_distance = 150
                if distance < max_connection_distance:
                    # Consider blob properties for connection strength
                    area_ratio = min(blob1['area'], blob2['area']) / max(blob1['area'], blob2['area'])
                    
                    if area_ratio > 0.3:  # Similar sized blobs
                        track = {
                            'start': center1,
                            'end': center2,
                            'distance': distance,
                            'strength': area_ratio * (1.0 - distance / max_connection_distance),
                            'blob_ids': [blob1['id'], blob2['id']]
                        }
                        tracks.append(track)
        
        return tracks


# Register the node
NODE_CLASS_MAPPINGS = {
    "CV_BlobTracker": CV_BlobTracker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CV_BlobTracker": "CV Blob Tracker"
}
