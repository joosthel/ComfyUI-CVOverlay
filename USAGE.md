# Usage Examples

## Installation & Setup

1. Install via ComfyUI-Manager (search for "ComfyUI-CVOverlay") 
2. Restart ComfyUI
3. Dependencies are installed automatically
4. Find nodes under "CV/" category in the node menu

## Basic Object Detection Workflow

1. **CV Model Loader** → Load a YOLO model (start with `yolov8n.pt` for lightweight processing)
2. **Load Image** (ComfyUI default node) → Your input image
3. **CV Object Detector** → Connect model and image, adjust confidence threshold
4. **CV Aesthetic Overlay** → Connect image and detections, choose style and colors
5. **Save Image** (ComfyUI default node) → Output the result

## Basic Blob Tracking Workflow

1. **Load Image** → Your input image/video frame
2. **CV Blob Tracker** → Process for moving objects
3. **CV Aesthetic Overlay** → Visualize tracked objects
4. **Save Image** → Output the result

## Node Parameters Quick Reference

### CV Model Loader
- `model_name`: Choose from yolov8n.pt (fastest) to yolov8x.pt (most accurate)
- `custom_model_path`: Use your own trained model (optional)

### CV Object Detector
- `confidence`: 0.5 = balanced, 0.3 = more detections, 0.7 = fewer but more confident
- `iou_threshold`: 0.45 = default overlap filtering

### CV Aesthetic Overlay
- `overlay_style`: 
  - `minimal` = Just corner markers
  - `technical` = Full boxes with corners
  - `surveillance` = Thick boxes with crosshairs
  - `cyberpunk` = Glitched fragmented boxes
- `color_scheme`: green (classic), blue, red, white, orange, purple
- `opacity`: 0.8 = semi-transparent, 1.0 = fully opaque

### CV Blob Tracker
- `tracking_method`:
  - `background_subtraction` = Good for stationary camera
  - `optical_flow` = Good for moving camera
  - `contour_tracking` = Simple shape-based tracking
- `min_area`: Minimum size of objects to track (500 = medium sized objects)

## Tips

1. **Start Small**: Use `yolov8n.pt` model first, it's fast and good for testing
2. **Adjust Confidence**: Lower confidence (0.3-0.4) for more detections, higher (0.6-0.8) for precision
3. **Style Mixing**: Try different overlay styles with different color schemes for unique looks
4. **Video Processing**: For video, connect multiple frames through the same workflow
5. **Performance**: Disable unnecessary overlays (grid, crosshairs) for faster processing

## Troubleshooting

- **No detections**: Lower confidence threshold or try different YOLO model
- **Too many false positives**: Raise confidence threshold or IOU threshold
- **Tracking not working**: Try different tracking method or adjust min_area
- **Overlay too bright**: Reduce opacity or try different color scheme
