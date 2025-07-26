# ComfyUI-OpenCV-Overlays

This project integrates OpenCV and YOLOv8 into ComfyUI, providing custom nodes for object detection, blob tracking, and aesthetic video overlays.

## Overview

The `ComfyUI-CVOverlay` package provides four main custom nodes:

- **CV Model Loader**: Loads YOLO models for object detection
- **CV Object Detector**: Performs real-time object detection using YOLOv8
- **CV Blob Tracker**: Implements blob tracking algorithms using OpenCV
- **CV Aesthetic Overlay**: Applies customizable technical/surveillance-style overlays

## Features

✨ **Lightweight Integration**: Minimal dependencies, optimized for ComfyUI workflows  
🎯 **Multiple Detection Models**: Support for YOLOv8n through YOLOv8x models  
🎨 **Aesthetic Overlays**: Technical, surveillance, minimal, and cyberpunk styles  
📹 **Video Processing**: Frame-by-frame processing for video workflows  
🎛️ **Full Control**: Confidence thresholds, colors, opacity, line thickness  

## Installation

### Method 1: Using ComfyUI-Manager (Recommended)
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) if you haven't already
2. Open ComfyUI-Manager in ComfyUI 
3. Go to "Install Custom Nodes"
4. Search for "ComfyUI-CVOverlay" and click Install
5. Restart ComfyUI

### Method 2: Manual Installation
1. Navigate to your ComfyUI `custom_nodes` directory
2. Clone this repository:
   ```bash
   git clone https://github.com/joosthel/ComfyUI-CVOverlay.git
   ```
3. Dependencies will be automatically installed by ComfyUI-Manager on next startup
4. Restart ComfyUI

**Note**: ComfyUI-Manager will automatically handle the installation of required dependencies (`opencv-python`, `ultralytics`, `torch`, etc.) when you restart ComfyUI.

## Quick Start

1. **Load Model**: Use `CV Model Loader` with `yolov8n.pt` (lightweight)
2. **Detect Objects**: Connect your image to `CV Object Detector`
3. **Apply Style**: Use `CV Aesthetic Overlay` to visualize detections
4. **Save Result**: Connect to ComfyUI's Save Image node

See [USAGE.md](USAGE.md) for detailed examples and workflows.

## Node Reference

### CV Model Loader
- Loads YOLO models (n/s/m/l/x variants)
- Supports custom trained models
- Output: CV_MODEL

### CV Object Detector  
- Input: CV_MODEL, IMAGE
- Configurable confidence and IOU thresholds
- Output: IMAGE, CV_DETECTIONS

### CV Blob Tracker
- Input: IMAGE, (optional) CV_TRACKS
- Multiple tracking algorithms
- Output: IMAGE, CV_TRACKS

### CV Aesthetic Overlay
- Input: IMAGE, (optional) CV_DETECTIONS, CV_TRACKS
- Multiple overlay styles and color schemes
- Output: IMAGE

## Project Structure

```
ComfyUI-CVOverlay/
├── __init__.py              # ComfyUI node registration
├── requirements.txt         # Dependencies
├── USAGE.md                # Usage examples and workflows
├── nodes/                  # Custom nodes
│   ├── cv_model_loader.py
│   ├── cv_object_detector.py
│   ├── cv_blob_tracker.py
│   └── cv_aesthetic_overlay.py
└── utils/                  # Helper functions
    ├── opencv_helpers.py
    └── yolo_utils.py
```

## Requirements

- Python ≥ 3.8
- ComfyUI
- OpenCV ≥ 4.8.0
- Ultralytics (YOLOv8) ≥ 8.0.0
- PyTorch ≥ 1.11.0

## Contributing

Contributions welcome! Please submit pull requests or open issues for enhancements and bug fixes.

## License

MIT License - see LICENSE file for details.