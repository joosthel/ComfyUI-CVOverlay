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
                
                # Simplified Controls - Bright Spots Only
                "detection_threshold": ("FLOAT", {
                    "default": 0.75,  
                    "min": 0.01,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Lower = more sensitive, Higher = only brightest spots"
                }),
                
                "min_blob_size": ("INT", {
                    "default": 300,
                    "min": 10,
                    "max": 5000,
                    "step": 10,
                    "tooltip": "Minimum blob size in pixels - filters out noise"
                }),
                
                "max_blob_size": ("INT", {
                    "default": 12000,
                    "min": 500,
                    "max": 100000,
                    "step": 100,
                    "tooltip": "Maximum blob size in pixels - prevents huge regions"
                }),
                
                "blur_radius": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 2,
                    "tooltip": "Blur radius for stability - higher values extend tracking borders and reduce noise"
                }),
                
                "max_blobs": ("INT", {
                    "default": 50,  
                    "min": 1,
                    "max": 1000, 
                    "step": 1,
                    "tooltip": "Maximum number of tracked blobs"
                }),
                
                # Plexus Controls
                "enable_plexus": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable plexus connections between blobs"
                }),
                
                "max_connections": ("INT", {
                    "default": 15,  # Lower default for cleaner look
                    "min": 5,
                    "max": 100,  # Reduced max
                    "step": 1,
                    "tooltip": "Maximum plexus connections"
                }),
                
                "plexus_distance": ("INT", {
                    "default": 200,  # Increased for better connections
                    "min": 50,
                    "max": 5000,
                    "step": 10,
                    "tooltip": "Maximum distance for plexus connections"
                }),
            },
            "optional": {
                "video_id": ("STRING", {
                    "default": "default",
                    "tooltip": "Unique ID for this video sequence"
                }),
                "reset_tracking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reset all tracking memory"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "CV_BLOBS", "CV_TRACKS")
    RETURN_NAMES = ("image", "blobs", "tracks")
    FUNCTION = "track_blobs"
    CATEGORY = "CV/Tracking"
    
    def track_blobs(self, image, detection_threshold, min_blob_size, max_blob_size, blur_radius, max_blobs,
                   enable_plexus, max_connections, plexus_distance,
                   video_id="default", reset_tracking=False):
        """Simplified bright spot tracking with relaxed/stable behavior"""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(f"Missing dependencies: {MISSING_DEPS}")
        
        # Relaxed defaults for stable, non-twitchy tracking (other parameters)
        noise_cleanup = 0.6  # More aggressive noise cleanup
        max_movement_per_frame = 200  # Allow more movement for stability
        memory_frames = 60  # Much longer memory (2+ seconds at 30fps)
        chaos_reduction = True
        stability_filter = 0.1  # Very permissive for stable tracking
        tracking_confidence = 0.1  # Very low for easier connections
        
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
                        
                        # Detect and track blobs for this frame (bright spots only)
                        frame_blobs, frame_tracks = self._process_frame(
                            gray, detection_threshold, min_blob_size, max_blob_size, blur_radius,
                            noise_cleanup, max_movement_per_frame, memory_frames, 
                            chaos_reduction, stability_filter, tracking_confidence, enable_plexus,
                            plexus_distance, max_connections, max_blobs,
                            f"{video_id}_frame_{i}"
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
            
            # Detect and track blobs (bright spots only)
            blobs, tracks = self._process_frame(
                gray, detection_threshold, min_blob_size, max_blob_size, blur_radius,
                noise_cleanup, max_movement_per_frame, memory_frames, 
                chaos_reduction, stability_filter, tracking_confidence, enable_plexus,
                plexus_distance, max_connections, max_blobs, video_id
            )
            
            # Return original image unchanged + blob data + track data
            return (image, blobs, tracks)
            
        except Exception as e:
            raise RuntimeError(f"Blob tracking failed: {str(e)}")
            
    def _process_frame(self, gray, detection_threshold, min_blob_size, max_blob_size,
                      blur_radius, noise_cleanup, max_movement_per_frame, memory_frames, 
                      chaos_reduction, stability_filter, tracking_confidence, enable_plexus,
                      plexus_distance, max_connections, max_blobs, video_id):
        """Process frame with bright spot detection and relaxed tracking"""
        
        # Ensure blur radius is odd
        if blur_radius % 2 == 0:
            blur_radius += 1
        
        # More aggressive pre-processing for stability
        processed = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)
        
        # Apply chaos reduction preprocessing (always enabled for stability)
        processed = cv2.medianBlur(processed, 5)  # Stronger median filter
        processed = cv2.bilateralFilter(processed, 9, 75, 75)  # Stronger bilateral filter
        
        # Bright spot detection only
        binary = self._detect_bright_spots(processed, detection_threshold)
        
        # More aggressive noise cleanup for stability
        cleanup_size = max(5, int(noise_cleanup * 10))  # Larger cleanup
        if cleanup_size % 2 == 0:
            cleanup_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cleanup_size, cleanup_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Additional smoothing pass for very stable blobs
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find and filter contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract blobs with stability metrics
        current_blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_blob_size <= area <= max_blob_size:
                # Calculate blob properties
                moments = cv2.moments(contour)
                if moments['m00'] == 0:
                    continue
                
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate stability metrics
                perimeter = cv2.arcLength(contour, True)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Circularity (how round the blob is)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Aspect ratio stability
                aspect_ratio = w / h if h > 0 else 1
                aspect_stability = min(aspect_ratio, 1/aspect_ratio)  # 0-1, 1 = square
                
                # Overall stability score
                stability_score = (solidity * 0.4 + circularity * 0.3 + aspect_stability * 0.3)
                
                # Apply stability filter
                if stability_score >= (1.0 - stability_filter):
                    blob_data = {
                        'center': [center_x, center_y],
                        'bbox': [x, y, x + w, y + h],
                        'area': area,
                        'perimeter': perimeter,
                        'solidity': solidity,
                        'circularity': circularity,
                        'aspect_ratio': aspect_ratio,
                        'stability_score': stability_score,
                        'contour': contour.tolist(),
                        'width': w,
                        'height': h,
                        'confidence': min(1.0, stability_score)
                    }
                    current_blobs.append(blob_data)
        
        # Update tracking with new parameters
        tracked_blobs, tracks = self._update_tracking_improved(
            current_blobs, max_movement_per_frame, memory_frames, 
            tracking_confidence, enable_plexus, plexus_distance,
            max_connections, max_blobs, video_id
        )
        
        return tracked_blobs, tracks
    
    def _detect_motion_analysis(self, gray, threshold, learning_rate, video_id):
        """Motion detection with controllable background learning"""
        if video_id not in self._video_states:
            self._video_states[video_id] = {
                'bg_subtractor': cv2.createBackgroundSubtractorMOG2(
                    detectShadows=False,
                    varThreshold=threshold * 100,  # Convert to MOG2 scale
                    history=200
                ),
                'prev_frame': None
            }
        
        state = self._video_states[video_id]
        
        # Update learning rate
        fg_mask = state['bg_subtractor'].apply(gray, learningRate=learning_rate)
        
        # Additional frame differencing for immediate motion
        if state['prev_frame'] is not None:
            frame_diff = cv2.absdiff(gray, state['prev_frame'])
            diff_threshold = int(threshold * 255)
            _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)
            
            # Combine both methods
            combined = cv2.bitwise_or(fg_mask, diff_mask)
        else:
            combined = fg_mask
        
        state['prev_frame'] = gray.copy()
        return combined

    def _detect_bright_spots(self, gray, threshold):
        """Detect bright spots with enhanced stability for reduced twitching"""
        # More conservative percentile-based thresholding
        high_percentile = 100 - (threshold * 60)  # Less aggressive than before
        threshold_val = np.percentile(gray, high_percentile)
        
        # Ensure minimum threshold to avoid noise
        threshold_val = max(threshold_val, 100)  # Minimum brightness threshold
        
        _, binary = cv2.threshold(gray, int(threshold_val), 255, cv2.THRESH_BINARY)
        
        # Additional stability: only keep blobs that are significantly brighter
        # Create a more conservative mask
        very_bright_threshold = min(255, threshold_val + 30)
        _, very_bright = cv2.threshold(gray, int(very_bright_threshold), 255, cv2.THRESH_BINARY)
        
        # Combine both thresholds for more stable detection
        # Use the very bright areas as "seeds" and grow them to the regular threshold
        combined = cv2.bitwise_and(binary, cv2.dilate(very_bright, np.ones((5,5), np.uint8), iterations=1))
        
        return combined

    def _detect_dark_spots(self, gray, threshold):
        """Detect dark spots with precise control"""
        # Use percentile-based thresholding for consistency
        low_percentile = threshold * 80  # threshold 0.1 = 8th percentile
        threshold_val = np.percentile(gray, low_percentile)
        
        _, binary = cv2.threshold(gray, int(threshold_val), 255, cv2.THRESH_BINARY_INV)
        return binary

    def _detect_high_contrast_areas(self, gray, threshold):
        """Detect areas with high local contrast"""
        # Calculate local contrast using standard deviation
        kernel_size = 9
        mean_filter = cv2.boxFilter(gray.astype(np.float32), -1, (kernel_size, kernel_size))
        sqr_filter = cv2.boxFilter((gray.astype(np.float32))**2, -1, (kernel_size, kernel_size))
        local_std = np.sqrt(sqr_filter - mean_filter**2)
        
        # Threshold based on standard deviation
        std_threshold = np.percentile(local_std, 100 - (threshold * 70))
        _, binary = cv2.threshold(local_std, std_threshold, 255, cv2.THRESH_BINARY)
        
        return binary.astype(np.uint8)

    def _detect_texture_regions(self, gray, threshold):
        """Detect regions with high texture (edge density)"""
        # Use Sobel operators for edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Smooth edge magnitude
        edge_smooth = cv2.GaussianBlur(edge_magnitude, (5, 5), 0)
        
        # Threshold
        edge_threshold = np.percentile(edge_smooth, 100 - (threshold * 60))
        _, binary = cv2.threshold(edge_smooth, edge_threshold, 255, cv2.THRESH_BINARY)
        
        return binary.astype(np.uint8)
    
    def _update_tracking_improved(self, current_blobs, max_movement, memory_frames, 
                                 tracking_confidence, enable_plexus, plexus_distance,
                                 max_connections, max_blobs, video_id):
        """Update blob tracking with improved stability and confidence metrics"""
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
            tracked_list = [(blob_id, blob) for blob_id, blob in tracked_blobs.items() if blob['active']]
            
            if tracked_list:
                tracked_centers = np.array([blob['center'] for _, blob in tracked_list])
                distances = cdist(current_centers, tracked_centers)
                
                # Enhanced matching with confidence scoring
                for i in range(len(current_blobs)):
                    if i in assigned_current:
                        continue
                    
                    best_match = None
                    best_confidence = 0.0
                    best_j = None
                    
                    for j, (blob_id, tracked_blob) in enumerate(tracked_list):
                        if j in assigned_tracked:
                            continue
                        
                        distance = distances[i, j]
                        if distance < max_movement:
                            # Calculate matching confidence based on multiple factors
                            distance_score = 1.0 - (distance / max_movement)
                            
                            # Size similarity
                            current_area = current_blobs[i]['area']
                            tracked_area = tracked_blob['area']
                            size_ratio = min(current_area, tracked_area) / max(current_area, tracked_area)
                            
                            # Shape similarity
                            current_solidity = current_blobs[i]['solidity']
                            tracked_solidity = tracked_blob['solidity']
                            shape_similarity = 1.0 - abs(current_solidity - tracked_solidity)
                            
                            # Overall confidence
                            match_confidence = (distance_score * 0.5 + size_ratio * 0.3 + shape_similarity * 0.2)
                            
                            if match_confidence > (tracking_confidence * 0.5) and match_confidence > best_confidence:  # Lower threshold
                                best_match = blob_id
                                best_confidence = match_confidence
                                best_j = j
                    
                    if best_match is not None:
                        # Update existing blob
                        current_blobs[i]['id'] = best_match
                        current_blobs[i]['age'] = tracked_blobs[best_match]['age'] + 1
                        current_blobs[i]['tracking_confidence'] = best_confidence
                        
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
                blob['tracking_confidence'] = blob['stability_score']
                
                tracked_blobs[blob_id] = blob
        
        # Update lost blobs
        active_blob_ids = set([blob['id'] for blob in current_blobs if 'id' in blob])
        for blob_id, blob in tracked_blobs.items():
            if blob['active'] and blob_id not in active_blob_ids:
                blob['lost_frames'] += 1
                if blob['lost_frames'] > memory_frames:
                    blob['active'] = False
        
        # Prepare output - Apply blob limiting based on certainty
        active_blobs = [blob for blob in tracked_blobs.values() if blob['active']]
        
        # Sort by confidence and limit if max_blobs > 0
        if max_blobs > 0 and len(active_blobs) > max_blobs:
            # Sort by tracking confidence (certainty) in descending order
            active_blobs.sort(key=lambda b: b.get('tracking_confidence', 0), reverse=True)
            active_blobs = active_blobs[:max_blobs]
        
        # Generate track connections with plexus animation
        if enable_plexus:
            tracks = self._generate_plexus_connections(
                active_blobs, plexus_distance, max_connections, tracking_confidence
            )
        else:
            tracks = self._generate_tracks_improved(tracked_blobs, tracking_confidence)
        
        return active_blobs, tracks
    
    def _generate_tracks_improved(self, tracked_blobs, confidence_threshold):
        """Generate track lines between nearby blobs with improved confidence scoring"""
        tracks = []
        active_blobs = [blob for blob in tracked_blobs.values() 
                       if blob['active'] and blob['age'] > 1 and  # Reduced age requirement
                       blob.get('tracking_confidence', 0) > (confidence_threshold * 0.3)]  # Much lower threshold
        
        # Generate connections between nearby blobs
        for i, blob1 in enumerate(active_blobs):
            for j, blob2 in enumerate(active_blobs[i+1:], i+1):
                center1 = blob1['center']
                center2 = blob2['center']
                
                # Calculate distance
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # More generous connection distance
                avg_size = (blob1['area'] + blob2['area']) / 2
                max_connection_distance = min(300, max(150, np.sqrt(avg_size) * 4))  # Increased
                
                if distance < max_connection_distance:
                    # Enhanced connection strength calculation
                    size_similarity = min(blob1['area'], blob2['area']) / max(blob1['area'], blob2['area'])
                    stability_avg = (blob1['stability_score'] + blob2['stability_score']) / 2
                    confidence_avg = (blob1.get('tracking_confidence', 0.5) + 
                                    blob2.get('tracking_confidence', 0.5)) / 2
                    distance_factor = 1.0 - (distance / max_connection_distance)
                    
                    # Overall connection strength
                    connection_strength = (size_similarity * 0.3 + 
                                         stability_avg * 0.3 + 
                                         confidence_avg * 0.2 + 
                                         distance_factor * 0.2)
                    
                    if connection_strength > (confidence_threshold * 0.3):  # Much lower threshold
                        track = {
                            'start': center1,
                            'end': center2,
                            'distance': distance,
                            'strength': connection_strength,
                            'blob_ids': [blob1['id'], blob2['id']],
                            'confidence': confidence_avg,
                            'alpha': min(1.0, distance_factor * 2.0)  # More visible
                        }
                        tracks.append(track)
        
        return tracks
    
    def _generate_plexus_connections(self, active_blobs, max_distance, max_connections, confidence_threshold):
        """Generate plexus-style connections between nearby blobs"""
        tracks = []
        
        # Much more permissive filtering for plexus
        qualified_blobs = [blob for blob in active_blobs 
                          if blob.get('tracking_confidence', 0) > 0.1]  # Very low threshold
        
        # Calculate all possible connections with distances
        potential_connections = []
        
        for i, blob1 in enumerate(qualified_blobs):
            for j, blob2 in enumerate(qualified_blobs[i+1:], i+1):
                center1 = blob1['center']
                center2 = blob2['center']
                
                # Calculate distance
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance <= max_distance:
                    # Simplified connection strength for better visibility
                    distance_factor = 1.0 - (distance / max_distance)
                    confidence_avg = (blob1.get('tracking_confidence', 0.5) + 
                                    blob2.get('tracking_confidence', 0.5)) / 2
                    
                    # More generous connection strength
                    connection_strength = distance_factor * 0.7 + confidence_avg * 0.3
                    
                    potential_connections.append({
                        'start': center1,
                        'end': center2,
                        'distance': distance,
                        'strength': connection_strength,
                        'blob_ids': [blob1['id'], blob2['id']],
                        'confidence': confidence_avg,
                        'alpha': min(1.0, distance_factor * 1.2 + 0.3),  # More visible with minimum alpha
                        'plexus': True  # Mark as plexus connection
                    })
        
        # Sort by connection strength and limit
        potential_connections.sort(key=lambda x: x['strength'], reverse=True)
        tracks = potential_connections[:max_connections]
        
        # Ensure we always have some connections if blobs exist
        if len(qualified_blobs) > 1 and len(tracks) == 0:
            # Force at least one connection between closest blobs
            min_distance = float('inf')
            closest_pair = None
            
            for i, blob1 in enumerate(qualified_blobs):
                for j, blob2 in enumerate(qualified_blobs[i+1:], i+1):
                    center1 = blob1['center']
                    center2 = blob2['center']
                    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    
                    if distance < min_distance and distance <= max_distance:
                        min_distance = distance
                        closest_pair = (blob1, blob2, distance)
            
            if closest_pair:
                blob1, blob2, distance = closest_pair
                tracks = [{
                    'start': blob1['center'],
                    'end': blob2['center'],
                    'distance': distance,
                    'strength': 0.8,
                    'blob_ids': [blob1['id'], blob2['id']],
                    'confidence': 0.8,
                    'alpha': 0.8,
                    'plexus': True
                }]
        
        return tracks


# Register the node
NODE_CLASS_MAPPINGS = {
    "CV_BlobTracker": CV_BlobTracker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CV_BlobTracker": "CV Blob Tracker"
}
