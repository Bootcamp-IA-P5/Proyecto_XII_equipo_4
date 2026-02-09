"""
Image loader module for the Logo Detection Pipeline.
Handles loading images from files and directories with validation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Generator, Tuple, Union
from tqdm import tqdm

from . import config


class ImageLoader:
    """
    Class for loading and managing images for the detection pipeline.
    Supports loading single images, multiple images, or entire directories.
    """
    
    def __init__(self, supported_formats: List[str] = None):
        """
        Initialize the ImageLoader.
        
        Args:
            supported_formats: List of supported file extensions.
                             Defaults to config.SUPPORTED_FORMATS
        """
        self.supported_formats = supported_formats or config.SUPPORTED_FORMATS
    
    def is_valid_image(self, path: Union[str, Path]) -> bool:
        """
        Check if a file is a valid image.
        
        Args:
            path: Path to the file
            
        Returns:
            True if the file is a valid image, False otherwise
        """
        path = Path(path)
        
        if not path.exists():
            return False
        
        if not path.is_file():
            return False
        
        if path.suffix.lower() not in self.supported_formats:
            return False
        
        return True
    
    def load_image(self, path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load a single image from disk.
        
        Args:
            path: Path to the image file
            
        Returns:
            Image as numpy array (BGR format) or None if loading fails
        """
        path = Path(path)
        
        if not self.is_valid_image(path):
            print(f"Warning: Invalid image path: {path}")
            return None
        
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"Warning: Could not read image: {path}")
            return None
        
        return image
    
    def load_image_rgb(self, path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load a single image and convert to RGB format.
        
        Args:
            path: Path to the image file
            
        Returns:
            Image as numpy array (RGB format) or None if loading fails
        """
        image = self.load_image(path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_with_metadata(
        self, 
        path: Union[str, Path]
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Load an image along with its metadata.
        
        Args:
            path: Path to the image file
            
        Returns:
            Tuple of (image, metadata) or None if loading fails
        """
        path = Path(path)
        image = self.load_image(path)
        
        if image is None:
            return None
        
        metadata = {
            "filename": path.name,
            "filepath": str(path.absolute()),
            "extension": path.suffix.lower(),
            "height": image.shape[0],
            "width": image.shape[1],
            "channels": image.shape[2] if len(image.shape) > 2 else 1,
        }
        
        return image, metadata
    
    def get_images_from_directory(
        self, 
        directory: Union[str, Path],
        recursive: bool = False
    ) -> List[Path]:
        """
        Get list of valid image paths from a directory.
        
        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of Path objects for valid images
        """
        directory = Path(directory)
        
        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            return []
        
        if not directory.is_dir():
            print(f"Error: Not a directory: {directory}")
            return []
        
        image_paths = []
        
        if recursive:
            pattern_func = directory.rglob
        else:
            pattern_func = directory.glob
        
        for ext in self.supported_formats:
            # Search for both lowercase and uppercase extensions
            image_paths.extend(pattern_func(f"*{ext}"))
            image_paths.extend(pattern_func(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_paths = sorted(set(image_paths))
        
        return image_paths
    
    def load_batch(
        self, 
        paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> List[Tuple[Path, np.ndarray]]:
        """
        Load multiple images from a list of paths.
        
        Args:
            paths: List of image paths
            show_progress: Whether to show a progress bar
            
        Returns:
            List of tuples (path, image) for successfully loaded images
        """
        results = []
        
        iterator = tqdm(paths, desc="Loading images") if show_progress else paths
        
        for path in iterator:
            path = Path(path)
            image = self.load_image(path)
            
            if image is not None:
                results.append((path, image))
        
        print(f"Successfully loaded {len(results)}/{len(paths)} images")
        return results
    
    def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
        show_progress: bool = True
    ) -> List[Tuple[Path, np.ndarray]]:
        """
        Load all valid images from a directory.
        
        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories
            show_progress: Whether to show a progress bar
            
        Returns:
            List of tuples (path, image) for successfully loaded images
        """
        image_paths = self.get_images_from_directory(directory, recursive)
        
        if not image_paths:
            print(f"No valid images found in: {directory}")
            return []
        
        return self.load_batch(image_paths, show_progress)
    
    def iterate_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False
    ) -> Generator[Tuple[Path, np.ndarray], None, None]:
        """
        Generator that yields images one at a time from a directory.
        Memory-efficient for large directories.
        
        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories
            
        Yields:
            Tuples of (path, image) for each valid image
        """
        image_paths = self.get_images_from_directory(directory, recursive)
        
        for path in image_paths:
            image = self.load_image(path)
            if image is not None:
                yield path, image


def load_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Convenience function to load a single image.
    
    Args:
        path: Path to the image file
        
    Returns:
        Image as numpy array or None if loading fails
    """
    loader = ImageLoader()
    return loader.load_image(path)


def load_images_from_directory(
    directory: Union[str, Path],
    recursive: bool = False
) -> List[Tuple[Path, np.ndarray]]:
    """
    Convenience function to load all images from a directory.
    
    Args:
        directory: Path to the directory
        recursive: Whether to search subdirectories
        
    Returns:
        List of tuples (path, image)
    """
    loader = ImageLoader()
    return loader.load_directory(directory, recursive)
