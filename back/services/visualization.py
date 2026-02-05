"""
Visualization module for the Logo Detection Pipeline.
Provides functions for drawing bounding boxes, labels, and annotations on images.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path

from . import config

def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """
    Returns a color based on the confidence level.
    Verde: >80%
    Amarillo: 50-80%
    Rojo: <50%

    Args:
        confidence: Confidence level (0.0 - 1.0)
    Returns:
        Color in BGR format
    """
    if confidence > 0.8:
        return (0, 255, 0)  # Verde
    elif confidence > 0.5:
        return (0, 255, 255)  # Amarillo
    else:
        return (0, 0, 255)  # Rojo

def draw_bounding_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = None,
    thickness: int = None
) -> np.ndarray:
    """
    Draw a bounding box on an image.
    
    Args:
        image: Input image (will be modified in place)
        box: Bounding box coordinates as (x1, y1, x2, y2)
        color: Box color in BGR format. Defaults to config.BBOX_COLOR
        thickness: Line thickness. Defaults to config.BBOX_THICKNESS
        
    Returns:
        Image with bounding box drawn
    """
    if color is None:
        color = config.BBOX_COLOR
    if thickness is None:
        thickness = config.BBOX_THICKNESS
    
    x1, y1, x2, y2 = [int(coord) for coord in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    return image


def add_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = None,
    text_color: Tuple[int, int, int] = None,
    bg_color: Tuple[int, int, int] = None,
    thickness: int = None
) -> np.ndarray:
    """
    Add a text label with background to an image.
    
    Args:
        image: Input image (will be modified in place)
        text: Text to display
        position: Position (x, y) for the label (top-left corner)
        font_scale: Font scale. Defaults to config.LABEL_FONT_SCALE
        text_color: Text color in BGR. Defaults to config.LABEL_COLOR
        bg_color: Background color in BGR. Defaults to config.LABEL_BG_COLOR
        thickness: Text thickness. Defaults to config.LABEL_THICKNESS
        
    Returns:
        Image with label added
    """
    if font_scale is None:
        font_scale = config.LABEL_FONT_SCALE
    if text_color is None:
        text_color = config.LABEL_COLOR
    if bg_color is None:
        bg_color = config.LABEL_BG_COLOR
    if thickness is None:
        thickness = config.LABEL_THICKNESS
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = int(position[0]), int(position[1])
    
    # Ensure label stays within image bounds
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w - text_width - 4))
    y = max(text_height + 4, min(y, img_h))
    
    # Draw background rectangle
    padding = 4
    cv2.rectangle(
        image,
        (x, y - text_height - padding),
        (x + text_width + padding, y + padding),
        bg_color,
        -1  # Filled rectangle
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x + padding // 2, y),
        font,
        font_scale,
        text_color,
        thickness
    )
    
    return image


def annotate_detection(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float = None,
    color: Tuple[int, int, int] = None,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Annotate an image with a detection (bounding box + label).
    
    Args:
        image: Input image (will be modified in place)
        box: Bounding box as (x1, y1, x2, y2)
        label: Class label
        confidence: Detection confidence (0.0 - 1.0)
        show_confidence: Whether to show confidence percentage
        
    Returns:
        Annotated image
    """
    # Usar color basado en confianza si está disponible, sino color por defecto
    if confidence is not None:
        color = get_confidence_color(confidence)
    else:
        color = color if color is not None else config.BBOX_COLOR
    
    # Draw bounding box
    draw_bounding_box(image, box, color)
    
    # Prepare label text
    if show_confidence and confidence is not None:
        text = f"{label}: {confidence:.1%}"
    else:
        text = label
    
    # Add label above the box
    x1, y1, _, _ = box
    label_position = (x1, y1 - 5)
    add_label(image, text, label_position, bg_color=color)
    
    return image


def annotate_image(
    image: np.ndarray,
    detections: List[dict],
    show_confidence: bool = True
) -> np.ndarray:
    """
    Annotate an image with multiple detections.
    
    Args:
        image: Input image
        detections: List of detection dictionaries, each containing:
            - 'box': (x1, y1, x2, y2)
            - 'label': class name
            - 'confidence': confidence score (optional)
            - 'class_id': class ID for color selection (optional)
        show_confidence: Whether to show confidence percentages
        
    Returns:
        Annotated image (copy of original)
    """
    # Work on a copy to preserve original
    annotated = image.copy()
    
    for detection in detections:
        box = detection['box']
        label = detection.get('label', 'Unknown')
        confidence = detection.get('confidence', None)
        class_id = detection.get('class_id', 0)
        
        # Get color based on class ID
        color = config.get_color_for_class(class_id)
        
        annotate_detection(
            annotated,
            box,
            label,
            confidence,
            color,
            show_confidence
        )
    
    return annotated


def save_annotated_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95
) -> bool:
    """
    Save an annotated image to disk.
    
    Args:
        image: Image to save
        output_path: Path for the output file
        quality: JPEG quality (0-100)
        
    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_path)
    
    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set compression parameters
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif output_path.suffix.lower() == '.png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        params = []
    
    try:
        success = cv2.imwrite(str(output_path), image, params)
        if success:
            print(f"Saved: {output_path}")
        else:
            print(f"Error: Could not save image to {output_path}")
        return success
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def create_detection_summary(
    image: np.ndarray,
    detections: List[dict],
    title: str = "Detection Summary"
) -> np.ndarray:
    """
    Create a summary visualization showing the image and detection statistics.
    
    Args:
        image: Original image
        detections: List of detections
        title: Title for the summary
        
    Returns:
        Summary image
    """
    # Annotate the image
    annotated = annotate_image(image, detections)
    
    # Add title bar at top
    title_height = 40
    h, w = annotated.shape[:2]
    
    # Create output image with title bar
    output = np.zeros((h + title_height, w, 3), dtype=np.uint8)
    output[:title_height] = (50, 50, 50)  # Dark gray title bar
    output[title_height:] = annotated
    
    # Add title text
    cv2.putText(
        output,
        f"{title} - {len(detections)} detection(s)",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    
    return output


def show_image(
    image: np.ndarray,
    window_name: str = "Detection Result",
    wait: bool = True
) -> None:
    """
    Display an image in a window.
    
    Args:
        image: Image to display
        window_name: Name for the display window
        wait: Whether to wait for a key press
    """
    cv2.imshow(window_name, image)
    
    if wait:
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
