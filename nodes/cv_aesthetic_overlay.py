"""
CV Aesthetic Overlay Node for ComfyUI
Simplified unified art direction for object detection and blob tracking
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


class CV_AestheticOverlay:
    """Simplified art direction overlay for detection and blob data"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Simple Art Direction Controls
                "border_color": ("STRING", {
                    "default": "#00FF00",
                    "tooltip": "Border color in hex (e.g., #00FF00)"
                }),
                "border_thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "text_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Text color in hex (e.g., #FFFFFF)"
                }),
                "text_background_opacity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Black background opacity behind text"
                }),
                "text_size": ("INT", {
                    "default": 16,
                    "min": 8,
                    "max": 48,
                    "step": 2
                }),
                # Track connection controls
                "show_tracks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show connecting lines between tracked blobs"
                }),
                "track_color": ("STRING", {
                    "default": "#FF0080",
                    "tooltip": "Track line color in hex (e.g., #FF0080)"
                }),
                "track_thickness": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
                }),
                "track_opacity": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05
                }),
            },
            "optional": {
                "detections": ("CV_DETECTIONS",),
                "blobs": ("CV_BLOBS",),
                "tracks": ("CV_TRACKS",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_overlay"
    CATEGORY = "CV/Overlay"
    
    def apply_overlay(self, image, border_color, border_thickness, text_color, 
                     text_background_opacity, text_size, show_tracks, track_color,
                     track_thickness, track_opacity, detections=None, blobs=None, tracks=None):
        """Apply unified art direction overlay with surveillance-style tracking"""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(f"Missing dependencies: {MISSING_DEPS}")
        
        try:
            # Handle batch processing for video frames
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    # Batch of images [batch_size, height, width, channels]
                    batch_size = image.shape[0]
                    processed_frames = []
                    
                    for i in range(batch_size):
                        # Process each frame in the batch
                        frame = image[i]  # Single frame [height, width, channels]
                        frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                        
                        # Create working copy
                        overlay = frame_np.copy()
                        
                        # Parse colors
                        border_bgr = self._hex_to_bgr(border_color)
                        text_bgr = self._hex_to_bgr(text_color)
                        track_bgr = self._hex_to_bgr(track_color)
                        
                        # Get detections/blobs/tracks for this frame
                        frame_detections = detections[i] if detections and i < len(detections) else None
                        frame_blobs = blobs[i] if blobs and i < len(blobs) else None
                        frame_tracks = tracks[i] if tracks and i < len(tracks) else None
                        
                        # Draw tracks first (behind other elements)
                        if show_tracks and frame_tracks:
                            self._draw_tracks(overlay, frame_tracks, track_bgr, track_thickness, track_opacity)
                        
                        # Draw detections if provided
                        if frame_detections:
                            self._draw_detections(overlay, frame_detections, border_bgr, text_bgr,
                                                border_thickness, text_size, text_background_opacity)
                        
                        # Draw blobs if provided (with enhanced surveillance style)
                        if frame_blobs:
                            self._draw_surveillance_blobs(overlay, frame_blobs, border_bgr, text_bgr,
                                                        border_thickness, text_size, text_background_opacity)
                        
                        # Convert processed frame back to tensor
                        processed_frame = torch.from_numpy(overlay).float() / 255.0
                        processed_frames.append(processed_frame)
                    
                    # Stack all processed frames back into a batch
                    output_tensor = torch.stack(processed_frames, dim=0)
                    return (output_tensor,)
                    
                else:
                    # Single image [height, width, channels]
                    img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = np.array(image)
            
            # Process single image (non-batch case)
            # Create working copy
            overlay = img_np.copy()
            
            # Parse colors
            border_bgr = self._hex_to_bgr(border_color)
            text_bgr = self._hex_to_bgr(text_color)
            track_bgr = self._hex_to_bgr(track_color)
            
            # Draw tracks first (behind other elements)
            if show_tracks and tracks:
                self._draw_tracks(overlay, tracks, track_bgr, track_thickness, track_opacity)
            
            # Draw detections if provided
            if detections:
                self._draw_detections(overlay, detections, border_bgr, text_bgr,
                                    border_thickness, text_size, text_background_opacity)
            
            # Draw blobs if provided with surveillance style
            if blobs:
                self._draw_surveillance_blobs(overlay, blobs, border_bgr, text_bgr,
                               border_thickness, text_size, text_background_opacity)
            
            # Convert back to ComfyUI tensor format
            output_tensor = torch.from_numpy(overlay).float() / 255.0
            if output_tensor.dim() == 3:
                output_tensor = output_tensor.unsqueeze(0)
            
            return (output_tensor,)
            
        except Exception as e:
            raise RuntimeError(f"Overlay application failed: {str(e)}")
    
    def _draw_detections(self, img, detections, border_color, text_color,
                        thickness, text_size, text_bg_opacity):
        """Draw object detection overlays"""
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, thickness)
            
            # Draw label with background
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            self._draw_text_with_background(img, label, (x1, y1 - 5), text_color,
                                          text_size, text_bg_opacity)
    
    def _draw_blobs(self, img, blobs, border_color, text_color,
                   thickness, text_size, text_bg_opacity):
        """Draw blob tracking overlays"""
        for i, blob in enumerate(blobs):
            x, y, w, h = [int(coord) for coord in blob['bbox']]
            center_x, center_y = blob['center']
            area = blob['area']
            
            # Draw blob outline
            cv2.rectangle(img, (x, y), (x + w, y + h), border_color, thickness)
            
            # Draw center point
            cv2.circle(img, (int(center_x), int(center_y)), 3, text_color, -1)
            
            # Draw blob info
            label = f"Blob {i} ({area}px)"
            self._draw_text_with_background(img, label, (x, y - 5), text_color,
                                          text_size, text_bg_opacity)
    
    def _draw_tracks(self, img, tracks, track_color, thickness, opacity):
        """Draw surveillance-style track connections between blobs"""
        if not tracks:
            return
        
        # Create overlay for tracks with opacity
        overlay = img.copy()
        
        for track in tracks:
            start_point = tuple(map(int, track['start']))
            end_point = tuple(map(int, track['end']))
            
            # Vary line style based on connection strength
            strength = track.get('strength', 1.0)
            
            if strength > 0.7:
                # Strong connection - solid line
                cv2.line(overlay, start_point, end_point, track_color, thickness)
            elif strength > 0.4:
                # Medium connection - dashed line
                self._draw_dashed_line(overlay, start_point, end_point, track_color, thickness)
            else:
                # Weak connection - dotted line
                self._draw_dotted_line(overlay, start_point, end_point, track_color, thickness)
            
            # Add small connection indicators at endpoints
            cv2.circle(overlay, start_point, 2, track_color, -1)
            cv2.circle(overlay, end_point, 2, track_color, -1)
        
        # Apply opacity to tracks
        cv2.addWeighted(img, 1 - opacity, overlay, opacity, 0, img)
    
    def _draw_dashed_line(self, img, start, end, color, thickness):
        """Draw a dashed line"""
        dash_length = 10
        x1, y1 = start
        x2, y2 = end
        
        total_length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        if total_length == 0:
            return
        
        dx = (x2 - x1) / total_length
        dy = (y2 - y1) / total_length
        
        for i in range(0, total_length, dash_length * 2):
            dash_start_x = int(x1 + dx * i)
            dash_start_y = int(y1 + dy * i)
            dash_end_x = int(x1 + dx * min(i + dash_length, total_length))
            dash_end_y = int(y1 + dy * min(i + dash_length, total_length))
            
            cv2.line(img, (dash_start_x, dash_start_y), (dash_end_x, dash_end_y), color, thickness)
    
    def _draw_dotted_line(self, img, start, end, color, thickness):
        """Draw a dotted line"""
        dot_spacing = 8
        x1, y1 = start
        x2, y2 = end
        
        total_length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        if total_length == 0:
            return
        
        dx = (x2 - x1) / total_length
        dy = (y2 - y1) / total_length
        
        for i in range(0, total_length, dot_spacing):
            dot_x = int(x1 + dx * i)
            dot_y = int(y1 + dy * i)
            cv2.circle(img, (dot_x, dot_y), thickness, color, -1)
    
    def _draw_surveillance_blobs(self, img, blobs, border_color, text_color, thickness, text_size, text_bg_opacity):
        """Draw blobs with professional surveillance aesthetic"""
        for blob in blobs:
            x, y, x2, y2 = blob['bbox']
            w = x2 - x
            h = y2 - y
            center_x, center_y = blob['center']
            area = blob['area']
            blob_id = blob.get('id', 'N/A')
            age = blob.get('age', 1)
            confidence = blob.get('confidence', 1.0)
            
            # Color intensity based on tracking age and confidence
            alpha = min(1.0, age / 10.0) * confidence
            adjusted_border = tuple(int(c * alpha) for c in border_color)
            adjusted_text = tuple(int(c * alpha) for c in text_color)
            
            # Main bounding box
            cv2.rectangle(img, (x, y), (x2, y2), adjusted_border, thickness)
            
            # Corner markers for technical look
            corner_size = min(w, h) // 8
            corners = [(x, y), (x2, y), (x, y2), (x2, y2)]
            for corner_x, corner_y in corners:
                # Draw corner brackets
                if corner_x == x and corner_y == y:  # Top-left
                    cv2.line(img, (corner_x, corner_y), (corner_x + corner_size, corner_y), adjusted_border, thickness)
                    cv2.line(img, (corner_x, corner_y), (corner_x, corner_y + corner_size), adjusted_border, thickness)
                elif corner_x == x2 and corner_y == y:  # Top-right
                    cv2.line(img, (corner_x, corner_y), (corner_x - corner_size, corner_y), adjusted_border, thickness)
                    cv2.line(img, (corner_x, corner_y), (corner_x, corner_y + corner_size), adjusted_border, thickness)
                elif corner_x == x and corner_y == y2:  # Bottom-left
                    cv2.line(img, (corner_x, corner_y), (corner_x + corner_size, corner_y), adjusted_border, thickness)
                    cv2.line(img, (corner_x, corner_y), (corner_x, corner_y - corner_size), adjusted_border, thickness)
                elif corner_x == x2 and corner_y == y2:  # Bottom-right
                    cv2.line(img, (corner_x, corner_y), (corner_x - corner_size, corner_y), adjusted_border, thickness)
                    cv2.line(img, (corner_x, corner_y), (corner_x, corner_y - corner_size), adjusted_border, thickness)
            
            # Center crosshair
            crosshair_size = 5
            cv2.line(img, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), adjusted_text, 1)
            cv2.line(img, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), adjusted_text, 1)
            
            # Trajectory trail if available
            if 'trajectory' in blob and len(blob['trajectory']) > 1:
                trajectory = blob['trajectory']
                for i in range(1, len(trajectory)):
                    start_pt = tuple(map(int, trajectory[i-1]))
                    end_pt = tuple(map(int, trajectory[i]))
                    trail_alpha = (i / len(trajectory)) * alpha * 0.7
                    trail_color = tuple(int(c * trail_alpha) for c in border_color)
                    cv2.line(img, start_pt, end_pt, trail_color, 1)
            
            # Technical info display
            info_lines = [
                f"ID: {blob_id}",
                f"AGE: {age}",
                f"SIZE: {area}px",
                f"CONF: {confidence:.2f}"
            ]
            
            # Calculate info panel position
            info_x = x
            info_y = y - 10
            
            for i, line in enumerate(info_lines):
                line_y = info_y - (i * (text_size + 2))
                self._draw_text_with_background(img, line, (info_x, line_y), adjusted_text,
                                              text_size, text_bg_opacity * alpha)

    def _draw_text_with_background(self, img, text, pos, text_color, text_size, bg_opacity):
        """Draw text with semi-transparent black background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = text_size / 30.0  # Scale to reasonable size
        
        # Get text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 1)
        
        # Draw black background rectangle
        x, y = pos
        bg_color = (0, 0, 0)  # Black background
        
        # Create background rectangle
        overlay = img.copy()
        cv2.rectangle(overlay, (x - 2, y - text_height - 2), 
                     (x + text_width + 2, y + baseline + 2), bg_color, -1)
        
        # Apply background opacity
        cv2.addWeighted(img, 1 - bg_opacity, overlay, bg_opacity, 0, img)
        
        # Draw text
        cv2.putText(img, text, (x, y), font, font_scale, text_color, 1)
    
    def _hex_to_bgr(self, hex_color):
        """Convert hex color to BGR tuple"""
        try:
            # Remove # if present
            hex_color = hex_color.lstrip('#')
            # Convert to RGB then BGR for OpenCV
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return (rgb[2], rgb[1], rgb[0])  # BGR format
        except:
            return (0, 255, 0)  # Default green


# Register the node
NODE_CLASS_MAPPINGS = {
    "CV_AestheticOverlay": CV_AestheticOverlay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CV_AestheticOverlay": "CV Aesthetic Overlay"
}
