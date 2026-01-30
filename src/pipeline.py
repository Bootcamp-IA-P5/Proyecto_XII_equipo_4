"""
Detection Pipeline for the Logo Detection Project.
Orchestrates the complete workflow: loading, preprocessing, detection, and visualization.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Run: pip install ultralytics")
    YOLO = None

from . import config
from .image_loader import ImageLoader
from .preprocessing import preprocess_for_model, get_image_info
from .visualization import annotate_image, save_annotated_image, show_image


class DetectionPipeline:
    """
    Main pipeline class for logo/object detection.
    Handles the complete workflow from image loading to result visualization.
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = None,
        iou_threshold: float = None
    ):
        """
        Initialize the detection pipeline.
        
        Args:
            model_path: Path to YOLO model file (.pt). Defaults to config.DEFAULT_MODEL
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
        """
        self.model_path = model_path or config.DEFAULT_MODEL
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or config.IOU_THRESHOLD
        
        self.model = None
        self.image_loader = ImageLoader()
        
        # Ensure directories exist
        config.ensure_directories()
    
    def load_model(self) -> bool:
        """
        Load the YOLO model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if YOLO is None:
            print("Error: ultralytics package not available")
            return False
        
        try:
            print(f"Loading model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect(
        self,
        image: np.ndarray,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Run detection on a single image.
        
        Args:
            image: Input image as numpy array (BGR format)
            verbose: Whether to print detection details
            
        Returns:
            List of detection dictionaries with keys:
                - box: (x1, y1, x2, y2)
                - label: class name
                - confidence: confidence score
                - class_id: class ID
        """
        if self.model is None:
            if not self.load_model():
                return []
        
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=verbose
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                # Get box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = result.names[class_id]
                
                detection = {
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'label': class_name,
                    'confidence': conf,
                    'class_id': class_id
                }
                detections.append(detection)
        
        return detections
    
    def process_image(
        self,
        image_path: Union[str, Path],
        save_output: bool = True,
        show_result: bool = False,
        output_dir: Union[str, Path] = None
    ) -> Dict:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the input image
            save_output: Whether to save the annotated image
            show_result: Whether to display the result
            output_dir: Directory for output files. Defaults to config.OUTPUT_DIR
            
        Returns:
            Dictionary with processing results
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        
        result = {
            'input_path': str(image_path),
            'success': False,
            'detections': [],
            'output_path': None,
            'processing_time': 0
        }
        
        # Load image
        image = self.image_loader.load_image(image_path)
        if image is None:
            result['error'] = f"Could not load image: {image_path}"
            return result
        
        # Get image info
        result['image_info'] = get_image_info(image)
        
        # Run detection
        start_time = datetime.now()
        detections = self.detect(image)
        result['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        result['detections'] = detections
        result['detection_count'] = len(detections)
        
        # Annotate image
        annotated = annotate_image(image, detections)
        
        # Save output
        if save_output:
            output_path = output_dir / f"detected_{image_path.name}"
            if save_annotated_image(annotated, output_path):
                result['output_path'] = str(output_path)
        
        # Show result
        if show_result:
            show_image(annotated, f"Detections: {image_path.name}")
        
        result['success'] = True
        return result
    
    def process_directory(
        self,
        input_dir: Union[str, Path] = None,
        output_dir: Union[str, Path] = None,
        recursive: bool = False,
        save_output: bool = True
    ) -> List[Dict]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory. Defaults to config.INPUT_DIR
            output_dir: Output directory. Defaults to config.OUTPUT_DIR
            recursive: Whether to search subdirectories
            save_output: Whether to save annotated images
            
        Returns:
            List of result dictionaries for each processed image
        """
        input_dir = Path(input_dir) if input_dir else config.INPUT_DIR
        output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        
        # Get list of images
        image_paths = self.image_loader.get_images_from_directory(input_dir, recursive)
        
        if not image_paths:
            print(f"No images found in: {input_dir}")
            return []
        
        print(f"Found {len(image_paths)} images to process")
        
        results = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            result = self.process_image(
                image_path,
                save_output=save_output,
                show_result=False,
                output_dir=output_dir
            )
            results.append(result)
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        total_detections = sum(r.get('detection_count', 0) for r in results)
        
        print(f"\n{'='*50}")
        print(f"Processing Complete!")
        print(f"{'='*50}")
        print(f"Images processed: {successful}/{len(results)}")
        print(f"Total detections: {total_detections}")
        print(f"Output directory: {output_dir}")
        
        return results
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path] = None,
        show_result: bool = False
    ) -> Dict:
        """
        Process a video file through the detection pipeline.
        
        Args:
            video_path: Path to the input video
            output_dir: Directory for output files. Defaults to config.OUTPUT_DIR
            show_result: Whether to display frames during processing
            
        Returns:
            Dictionary with processing results
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        
        result = {
            'input_path': str(video_path),
            'success': False,
            'frame_count': 0,
            'detection_count': 0,
            'output_path': None,
            'processing_time': 0
        }
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            result['error'] = f"Could not open video: {video_path}"
            return result
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output video
        output_path = output_dir / f"detected_{video_path.name}"
        # Use a common codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        start_time = datetime.now()
        frame_idx = 0
        total_detections = 0
        
        print(f"Processing video: {video_path.name} ({total_frames} frames)")
        
        try:
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run detection on frame
                    detections = self.detect(frame)
                    total_detections += len(detections)
                    
                    # Annotate frame
                    annotated = annotate_image(frame, detections)
                    
                    # Write frame
                    out.write(annotated)
                    
                    if show_result:
                        cv2.imshow(f"Processing: {video_path.name}", annotated)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    frame_idx += 1
                    pbar.update(1)
        finally:
            cap.release()
            out.release()
            if show_result:
                cv2.destroyAllWindows()
        
        result['processing_time'] = (datetime.now() - start_time).total_seconds()
        result['frame_count'] = frame_idx
        result['detection_count'] = total_detections
        result['output_path'] = str(output_path)
        result['success'] = True
        
        return result

    def run(
        self,
        source: Union[str, Path],
        output_dir: Union[str, Path] = None,
        show: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        Run the pipeline on an image or directory.
        
        Args:
            source: Path to image file or directory
            output_dir: Output directory for results
            show: Whether to display results
            
        Returns:
            Result dictionary for single image, or list for directory
        """
        source = Path(source)
        
        if source.is_file():
            # Check if it's a video
            if source.suffix.lower() in config.SUPPORTED_VIDEO_FORMATS:
                return self.process_video(source, output_dir=output_dir, show_result=show)
            return self.process_image(source, save_output=True, show_result=show, output_dir=output_dir)
        elif source.is_dir():
            return self.process_directory(source, output_dir=output_dir)
        else:
            print(f"Error: Source not found: {source}")
            return {'error': f'Source not found: {source}', 'success': False}


def create_pipeline(
    model: str = None,
    confidence: float = None
) -> DetectionPipeline:
    """
    Factory function to create a detection pipeline.
    
    Args:
        model: Path to model file
        confidence: Confidence threshold
        
    Returns:
        Configured DetectionPipeline instance
    """
    return DetectionPipeline(
        model_path=model,
        confidence_threshold=confidence
    )
