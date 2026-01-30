"""
Configuration settings for the Logo Detection Pipeline.
Centralized configuration for paths, model settings, and visualization parameters.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base directory of the project
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Models directory
MODELS_DIR = BASE_DIR / "models"

# =============================================================================
# IMAGE SETTINGS
# =============================================================================

# Supported image formats
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]

# Supported video formats
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

# Default image size for preprocessing (width, height)
DEFAULT_IMAGE_SIZE = (640, 640)

# Normalization parameters
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# =============================================================================
# MODEL SETTINGS
# =============================================================================

# Default YOLO model (will be downloaded automatically if not present)
DEFAULT_MODEL = "yolov8n.pt"

# Detection confidence threshold (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.5

# IOU threshold for Non-Maximum Suppression
IOU_THRESHOLD = 0.45

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Bounding box settings
BBOX_COLOR = (0, 255, 0)  # Green in BGR format
BBOX_THICKNESS = 2

# Label settings
LABEL_FONT_SCALE = 0.6
LABEL_THICKNESS = 2
LABEL_COLOR = (255, 255, 255)  # White
LABEL_BG_COLOR = (0, 255, 0)  # Green background

# Color palette for multiple classes (BGR format)
COLOR_PALETTE = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (255, 165, 0),    # Orange
    (0, 128, 128),    # Teal
    (128, 128, 0),    # Olive
]

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [DATA_DIR, INPUT_DIR, OUTPUT_DIR, MODELS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_color_for_class(class_id: int) -> tuple:
    """Get a consistent color for a given class ID."""
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]
