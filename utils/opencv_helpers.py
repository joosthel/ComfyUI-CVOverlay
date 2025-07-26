"""
Utility functions for OpenCV operations
"""

import cv2
import numpy as np
import torch
from PIL import Image


def tensor_to_opencv(tensor):
    """Convert ComfyUI tensor to OpenCV format"""
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first image from batch
        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_np
    return tensor


def opencv_to_tensor(img_np):
    """Convert OpenCV format to ComfyUI tensor"""
    # Convert BGR to RGB
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    # Normalize and convert to tensor
    img_tensor = torch.from_numpy(img_np).float() / 255.0
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    return img_tensor


def tensor_to_pil(tensor):
    """Convert ComfyUI tensor to PIL Image"""
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first image from batch
        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)
    return tensor


def pil_to_tensor(img_pil):
    """Convert PIL Image to ComfyUI tensor"""
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor


def normalize_bbox(bbox, img_width, img_height):
    """Normalize bounding box coordinates to 0-1 range"""
    x1, y1, x2, y2 = bbox
    return [
        x1 / img_width,
        y1 / img_height,
        x2 / img_width,
        y2 / img_height
    ]


def denormalize_bbox(bbox, img_width, img_height):
    """Denormalize bounding box coordinates from 0-1 range to pixel coordinates"""
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * img_width),
        int(y1 * img_height),
        int(x2 * img_width),
        int(y2 * img_height)
    ]


def resize_image_keep_aspect(image, target_size):
    """Resize image while keeping aspect ratio"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    result = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    
    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result, scale, (x_offset, y_offset)
