"""
Tests for the Logo Detection Pipeline.
Run with: python -m pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from src import config
from src.preprocessing import resize_image, normalize_image, get_image_info
from src.image_loader import ImageLoader
from src.visualization import draw_bounding_box, add_label, annotate_image


class TestPreprocessing:
    """Tests for preprocessing module."""
    
    def test_resize_image_basic(self):
        """Test basic image resizing."""
        # Create a test image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Resize to target
        resized = resize_image(image, target_size=(320, 320))
        
        assert resized.shape == (320, 320, 3)
    
    def test_resize_image_keeps_aspect_ratio(self):
        """Test that resize maintains aspect ratio with padding."""
        # Create a wide image
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        # Resize with aspect ratio
        resized = resize_image(image, target_size=(200, 200), keep_aspect_ratio=True)
        
        assert resized.shape == (200, 200, 3)
        # Check that there's padding (gray pixels at top/bottom)
        assert resized[0, 100, 0] == 128  # Padding should be gray
    
    def test_normalize_image(self):
        """Test image normalization."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 127
        
        normalized = normalize_image(image)
        
        assert normalized.dtype == np.float32
        assert normalized.shape == (100, 100, 3)
    
    def test_get_image_info(self):
        """Test getting image information."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        info = get_image_info(image)
        
        assert info['height'] == 480
        assert info['width'] == 640
        assert info['channels'] == 3


class TestImageLoader:
    """Tests for image loader module."""
    
    def test_loader_initialization(self):
        """Test ImageLoader initialization."""
        loader = ImageLoader()
        
        assert loader.supported_formats == config.SUPPORTED_FORMATS
    
    def test_is_valid_image_nonexistent(self):
        """Test validation of non-existent file."""
        loader = ImageLoader()
        
        assert loader.is_valid_image("nonexistent.jpg") == False
    
    def test_get_images_from_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        loader = ImageLoader()
        
        images = loader.get_images_from_directory(tmp_path)
        
        assert images == []


class TestVisualization:
    """Tests for visualization module."""
    
    def test_draw_bounding_box(self):
        """Test drawing bounding box."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        result = draw_bounding_box(image, (50, 50, 150, 150))
        
        # Check that box was drawn (pixels should be non-zero at box edges)
        assert result[50, 50, 1] == 255  # Green channel at corner
    
    def test_add_label(self):
        """Test adding label to image."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        result = add_label(image, "Test Label", (50, 50))
        
        # Check that something was drawn (image should be modified)
        assert not np.array_equal(result, np.zeros((200, 200, 3), dtype=np.uint8))
    
    def test_annotate_image(self):
        """Test annotating image with detections."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = [
            {'box': (10, 10, 50, 50), 'label': 'Test', 'confidence': 0.9, 'class_id': 0}
        ]
        
        result = annotate_image(image, detections)
        
        # Result should be modified
        assert not np.array_equal(result, image)


class TestConfig:
    """Tests for configuration module."""
    
    def test_ensure_directories(self):
        """Test that directories are created."""
        config.ensure_directories()
        
        assert config.DATA_DIR.exists()
        assert config.INPUT_DIR.exists()
        assert config.OUTPUT_DIR.exists()
    
    def test_get_color_for_class(self):
        """Test color palette function."""
        color = config.get_color_for_class(0)
        
        assert isinstance(color, tuple)
        assert len(color) == 3
    
    def test_color_palette_wrapping(self):
        """Test that color palette wraps for large class IDs."""
        color1 = config.get_color_for_class(0)
        color2 = config.get_color_for_class(len(config.COLOR_PALETTE))
        
        assert color1 == color2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
