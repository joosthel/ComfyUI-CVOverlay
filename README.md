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
TouchDesigner-style blob tracking with 3 modes: bright_blobs, dark_blobs, motion_blobs

### 4. CV Aesthetic Overlay
Unified art direction for both detection and blob data with simple controls:
- **border_color** - Hex color for boxes/outlines (#00FF00)
- **border_thickness** - Line thickness (1-10)
- **text_color** - Hex color for text (#FFFFFF)
- **text_background_opacity** - Black background behind text (0.0-1.0)
- **text_size** - Text size (8-48)

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

Blob tracker maintains TouchDesigner compatibility:
- `threshold` - Brightness cutoff (0.0-1.0)
- `min_area` - Minimum blob size (10-10000px) 
- `max_area` - Maximum blob size (100-50000px)
- `blur_size` - Gaussian blur (1-21, odd numbers)
- `detection_mode` - bright/dark/motion blobs

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
