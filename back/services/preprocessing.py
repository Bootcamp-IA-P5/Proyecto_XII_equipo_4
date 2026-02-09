"""
Image preprocessing module for the Logo Detection Pipeline.
Provides functions for resizing, normalizing, and preparing images for model inference.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path

from . import config


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = None,
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize an image to the target size.
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        target_size: Target size as (width, height). Defaults to config.DEFAULT_IMAGE_SIZE
        keep_aspect_ratio: If True, maintains aspect ratio with padding
        
    Returns:
        Resized image as numpy array
    """
    if target_size is None:
        target_size = config.DEFAULT_IMAGE_SIZE
    
    target_w, target_h = target_size
    h, w = image.shape[:2]
    
    if not keep_aspect_ratio:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image with gray background
    padded = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    
    # Calculate padding offsets to center the image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return padded


def normalize_image(
    image: np.ndarray,
    mean: list = None,
    std: list = None
) -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        image: Input image as numpy array (BGR format)
        mean: Mean values for normalization (RGB order)
        std: Standard deviation values for normalization (RGB order)
        
    Returns:
        Normalized image as float32 numpy array
    """
    if mean is None:
        mean = config.NORMALIZE_MEAN
    if std is None:
        std = config.NORMALIZE_STD
    
    # Convert to RGB and float
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_float = image_rgb.astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    normalized = (image_float - mean) / std
    
    return normalized


def preprocess_for_model(
    image: np.ndarray,
    target_size: Tuple[int, int] = None,
    normalize: bool = False
) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image: Input image as numpy array
        target_size: Target size for resizing
        normalize: Whether to apply normalization
        
    Returns:
        Preprocessed image ready for model inference
    """
    # Resize with aspect ratio preservation
    processed = resize_image(image, target_size, keep_aspect_ratio=True)
    
    # Optionally normalize (YOLO typically handles this internally)
    if normalize:
        processed = normalize_image(processed)
    
    return processed


def load_and_preprocess(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = None
) -> Optional[np.ndarray]:
    """
    Load an image from disk and preprocess it.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image or None if loading fails
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return None
    
    # Check if format is supported
    if image_path.suffix.lower() not in config.SUPPORTED_FORMATS:
        print(f"Error: Unsupported format: {image_path.suffix}")
        return None
    
    # Load image (use np.fromfile + imdecode for Unicode path safety on Windows)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return None
    
    # Preprocess
    return preprocess_for_model(image, target_size)


def apply_augmentation(
    image: np.ndarray,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0
) -> np.ndarray:
    """
    Apply basic augmentations to an image.
    
    Args:
        image: Input image
        flip_horizontal: Whether to flip horizontally
        flip_vertical: Whether to flip vertically
        brightness_factor: Brightness adjustment factor (1.0 = no change)
        contrast_factor: Contrast adjustment factor (1.0 = no change)
        
    Returns:
        Augmented image
    """
    result = image.copy()
    
    # Flips
    if flip_horizontal:
        result = cv2.flip(result, 1)
    if flip_vertical:
        result = cv2.flip(result, 0)
    
    # Brightness and contrast
    if brightness_factor != 1.0 or contrast_factor != 1.0:
        result = cv2.convertScaleAbs(
            result,
            alpha=contrast_factor,
            beta=(brightness_factor - 1.0) * 127
        )
    
    return result


def get_image_info(image: np.ndarray) -> dict:
    """
    Get information about an image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary with image information
    """
    return {
        "height": image.shape[0],
        "width": image.shape[1],
        "channels": image.shape[2] if len(image.shape) > 2 else 1,
        "dtype": str(image.dtype),
        "size_bytes": image.nbytes
    }
