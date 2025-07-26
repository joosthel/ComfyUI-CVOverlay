"""
CV Aesthetic Overlay Node for ComfyUI
Applies technical/surveillance-style visual overlays to images with detection data
"""

# Handle missing dependencies gracefully
try:
    import cv2
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)


class CV_AestheticOverlay:
    """Applies aesthetic overlays to images with detection/tracking data"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "overlay_style": (["minimal", "technical", "surveillance", "cyberpunk"], {
                    "default": "technical"
                }),
                "line_thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "opacity": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "color_scheme": (["green", "blue", "red", "white", "orange", "purple"], {
                    "default": "green"
                }),
                "show_coordinates": ("BOOLEAN", {"default": True}),
                "show_confidence": ("BOOLEAN", {"default": True}),
                "show_grid": ("BOOLEAN", {"default": False}),
                "show_crosshairs": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "detections": ("CV_DETECTIONS",),
                "tracks": ("CV_TRACKS",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_overlay"
    CATEGORY = "CV/Overlay"
    
    def apply_overlay(self, image, overlay_style, line_thickness, opacity, color_scheme, 
                     show_coordinates, show_confidence, show_grid, show_crosshairs,
                     detections=None, tracks=None):
        """Apply aesthetic overlay to image"""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError(f"Missing dependencies: {MISSING_DEPS}. Please install requirements: pip install opencv-python torch Pillow")
        
        try:
            # Convert ComfyUI image tensor to PIL Image
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image[0]  # Take first image from batch
                img_np = (image.cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
            else:
                img_pil = image
            
            # Create overlay
            overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Color schemes
            colors = {
                'green': (0, 255, 0),
                'blue': (0, 150, 255),
                'red': (255, 50, 50),
                'white': (255, 255, 255),
                'orange': (255, 165, 0),
                'purple': (180, 0, 255)
            }
            
            primary_color = colors[color_scheme]
            alpha = int(opacity * 255)
            color_with_alpha = (*primary_color, alpha)
            
            # Draw grid if enabled
            if show_grid:
                self._draw_grid(draw, img_pil.size, color_with_alpha, line_thickness)
            
            # Draw detections
            if detections:
                for detection in detections:
                    self._draw_detection(draw, detection, overlay_style, color_with_alpha, 
                                       line_thickness, show_coordinates, show_confidence)
            
            # Draw tracks
            if tracks:
                for track in tracks:
                    self._draw_track(draw, track, overlay_style, color_with_alpha, 
                                   line_thickness, show_coordinates, show_confidence)
            
            # Draw crosshairs if enabled
            if show_crosshairs:
                self._draw_crosshairs(draw, img_pil.size, color_with_alpha, line_thickness)
            
            # Composite overlay onto original image
            if img_pil.mode != 'RGBA':
                img_pil = img_pil.convert('RGBA')
            
            result = Image.alpha_composite(img_pil, overlay)
            result = result.convert('RGB')
            
            # Convert back to ComfyUI tensor format
            result_np = np.array(result).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_np).unsqueeze(0)  # Add batch dimension
            
            return (result_tensor,)
            
        except Exception as e:
            raise RuntimeError(f"Overlay application failed: {str(e)}")
    
    def _draw_grid(self, draw, size, color, thickness):
        """Draw coordinate grid"""
        width, height = size
        grid_spacing = min(width, height) // 20
        
        # Vertical lines
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill=color, width=max(1, thickness//2))
        
        # Horizontal lines
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill=color, width=max(1, thickness//2))
    
    def _draw_crosshairs(self, draw, size, color, thickness):
        """Draw center crosshairs"""
        width, height = size
        center_x, center_y = width // 2, height // 2
        crosshair_size = min(width, height) // 20
        
        # Horizontal crosshair
        draw.line([(center_x - crosshair_size, center_y), 
                  (center_x + crosshair_size, center_y)], fill=color, width=thickness)
        
        # Vertical crosshair
        draw.line([(center_x, center_y - crosshair_size), 
                  (center_x, center_y + crosshair_size)], fill=color, width=thickness)
    
    def _draw_detection(self, draw, detection, style, color, thickness, show_coords, show_conf):
        """Draw detection overlay"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        if style == "minimal":
            # Just corner markers
            corner_size = 10
            self._draw_corners(draw, bbox, color, thickness, corner_size)
        
        elif style == "technical":
            # Rectangle + corner markers + info
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
            self._draw_corners(draw, bbox, color, thickness, 15)
            
        elif style == "surveillance":
            # Thick rectangle + crosshair at center
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness + 1)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cross_size = 8
            draw.line([(center_x - cross_size, center_y), 
                      (center_x + cross_size, center_y)], fill=color, width=thickness)
            draw.line([(center_x, center_y - cross_size), 
                      (center_x, center_y + cross_size)], fill=color, width=thickness)
        
        elif style == "cyberpunk":
            # Glitch-style fragmented rectangle
            self._draw_glitch_box(draw, bbox, color, thickness)
        
        # Add text information
        if show_coords or show_conf:
            text_y = max(y1 - 20, 10)
            text_parts = []
            
            if show_coords:
                text_parts.append(f"({x1},{y1})")
            
            if show_conf and 'confidence' in detection:
                conf = detection['confidence']
                text_parts.append(f"{conf:.2f}")
            
            if 'class_name' in detection:
                text_parts.append(detection['class_name'])
            
            if text_parts:
                text = " | ".join(text_parts)
                draw.text((x1, text_y), text, fill=color)
    
    def _draw_track(self, draw, track, style, color, thickness, show_coords, show_conf):
        """Draw tracking overlay"""
        if 'bbox' in track:
            # Draw like detection but with different style
            fake_detection = {
                'bbox': track['bbox'],
                'confidence': track.get('confidence', 0.0)
            }
            self._draw_detection(draw, fake_detection, style, color, thickness, show_coords, show_conf)
        
        if 'center' in track:
            # Draw center point
            center_x, center_y = track['center']
            point_size = thickness + 2
            draw.ellipse([center_x - point_size, center_y - point_size,
                         center_x + point_size, center_y + point_size], 
                        fill=color, outline=color)
    
    def _draw_corners(self, draw, bbox, color, thickness, corner_size):
        """Draw corner markers on bounding box"""
        x1, y1, x2, y2 = bbox
        
        # Top-left corner
        draw.line([(x1, y1), (x1 + corner_size, y1)], fill=color, width=thickness)
        draw.line([(x1, y1), (x1, y1 + corner_size)], fill=color, width=thickness)
        
        # Top-right corner
        draw.line([(x2 - corner_size, y1), (x2, y1)], fill=color, width=thickness)
        draw.line([(x2, y1), (x2, y1 + corner_size)], fill=color, width=thickness)
        
        # Bottom-left corner
        draw.line([(x1, y2 - corner_size), (x1, y2)], fill=color, width=thickness)
        draw.line([(x1, y2), (x1 + corner_size, y2)], fill=color, width=thickness)
        
        # Bottom-right corner
        draw.line([(x2 - corner_size, y2), (x2, y2)], fill=color, width=thickness)
        draw.line([(x2, y2 - corner_size), (x2, y2)], fill=color, width=thickness)
    
    def _draw_glitch_box(self, draw, bbox, color, thickness):
        """Draw cyberpunk-style glitched bounding box"""
        x1, y1, x2, y2 = bbox
        
        # Main rectangle with gaps
        segments = [
            [(x1, y1), (x1 + (x2-x1)//3, y1)],  # Top left segment
            [(x1 + 2*(x2-x1)//3, y1), (x2, y1)],  # Top right segment
            [(x2, y1), (x2, y1 + (y2-y1)//3)],  # Right top segment
            [(x2, y1 + 2*(y2-y1)//3), (x2, y2)],  # Right bottom segment
            [(x2, y2), (x1 + 2*(x2-x1)//3, y2)],  # Bottom right segment
            [(x1 + (x2-x1)//3, y2), (x1, y2)],  # Bottom left segment
            [(x1, y2), (x1, y1 + 2*(y2-y1)//3)],  # Left bottom segment
            [(x1, y1 + (y2-y1)//3), (x1, y1)],  # Left top segment
        ]
        
        for segment in segments:
            draw.line(segment, fill=color, width=thickness)
