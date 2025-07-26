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
            },
            "optional": {
                "detections": ("CV_DETECTIONS",),
                "blobs": ("CV_BLOBS",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_overlay"
    CATEGORY = "CV/Overlay"
    
    def apply_overlay(self, image, border_color, border_thickness, text_color, 
                     text_background_opacity, text_size, detections=None, blobs=None):
        """Apply unified art direction overlay"""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(f"Missing dependencies: {MISSING_DEPS}")
        
        try:
            # Convert ComfyUI tensor to OpenCV format
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image[0]  # Take first image from batch
                img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = np.array(image)
            
            # Create working copy
            overlay = img_np.copy()
            
            # Parse colors
            border_bgr = self._hex_to_bgr(border_color)
            text_bgr = self._hex_to_bgr(text_color)
            
            # Draw detections if provided
            if detections:
                self._draw_detections(overlay, detections, border_bgr, text_bgr,
                                    border_thickness, text_size, text_background_opacity)
            
            # Draw blobs if provided  
            if blobs:
                self._draw_blobs(overlay, blobs, border_bgr, text_bgr,
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
