# ComfyUI-CVOverlay

Minimal OpenCV and YOLOv8 integration for ComfyUI with clean architecture and TouchDesigner-style blob tracking.

## 🎯 Features

- **4 Essential Nodes** - Clean separation of detection and visualization
- **YOLOv8 Object Detection** - Automatic model downloading  
- **TouchDesigner Blob Tracking** - Bright/dark/motion blob detection
- **Unified Art Direction** - Single overlay node handles all styling
- **Standard ComfyUI Integration** - Works with existing image/video nodes

## 📦 Nodes

### 1. CV Model Loader
Load YOLOv8 models (yolov8n.pt → yolov8x.pt) with automatic downloading to `ComfyUI/models/yolo/` folder

### 2. CV Object Detector  
Pure object detection returning detection data (no visualization)

### 3. CV Blob Tracker
Professional TouchDesigner-style blob tracking with 5 detection modes:
- **Motion** - Background subtraction + frame differencing
- **Bright Regions** - Adaptive bright area detection  
- **Dark Regions** - Adaptive dark area detection
- **Edge Density** - High-detail region detection
- **Color Variance** - Local texture variation detection

**Enhanced Features:**
- Persistent blob tracking with unique IDs
- Trajectory trails showing movement history
- Technical surveillance aesthetic with corner brackets and crosshairs
- Smart connection lines between related blobs

### 4. CV Aesthetic Overlay
Unified art direction for both detection and blob data with surveillance-style controls:
- **border_color** - Hex color for boxes/outlines (#00FF00)
- **border_thickness** - Line thickness (1-10)
- **text_color** - Hex color for text (#FFFFFF)
- **text_background_opacity** - Black background behind text (0.0-1.0)
- **text_size** - Text size (8-48)
- **show_tracks** - Enable connecting lines between blobs
- **track_color** - Track line color (#FF0080)
- **track_thickness** - Track line thickness (1-5)
- **track_opacity** - Track transparency (0.1-1.0)

## 🔄 Workflow

### **For Images:**
```
Load Image → Detection Node → CV Aesthetic Overlay → Preview Image
```

### **For Videos:**
```
VHS Load Video → Detection Node → CV Aesthetic Overlay → VHS Video Combine
```

**Use standard ComfyUI nodes** for input/output: Load Image, VHS Video nodes, Preview Image, etc.

### **Video Batch Processing:**
- **Automatic batch handling** - All CV nodes process every frame in video batches
- **Frame-by-frame detection** - Object detection and blob tracking on each frame
- **Consistent styling** - Art direction applied to all frames uniformly
- **Memory efficient** - Processes video frames in batches without loading entire video

## 🔧 TouchDesigner Parameters

Blob tracker maintains TouchDesigner compatibility with advanced controls:
- `detection_mode` - Motion, bright/dark regions, edge density, color variance
- `sensitivity` - Detection threshold (0.01-1.0)
- `min_size/max_size` - Size filtering (10-20000px)
- `blur_amount` - Gaussian preprocessing (1-15, odd numbers)
- `noise_reduction` - Morphological cleaning (0.0-1.0)
- `max_tracking_distance` - Blob tracking range (10-200px)
- `track_persistence` - Lost blob retention (1-60 frames)

## ⚡ Installation

1. Clone to ComfyUI custom_nodes folder
2. Restart ComfyUI (dependencies handled automatically)
3. Find nodes under "CV" category

### **Model Storage:**
- **YOLO models** download to `ComfyUI/models/yolo/`
- **Follows ComfyUI conventions** for clean organization
- **Reusable across sessions** - models persist in proper location

## 📄 License

MIT License
