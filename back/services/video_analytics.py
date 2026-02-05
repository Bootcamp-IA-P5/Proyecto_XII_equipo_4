"""
Video Analytics Module for Logo Detection.
Provides metrics calculation and frame extraction for video analysis.

Features:
- Calculate total appearance time per logo/class
- Calculate percentage of frames with detections
- Extract and save frames with detections
- Generate detailed analytics report
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Run: pip install ultralytics")
    YOLO = None

from . import config
from .visualization import annotate_image


@dataclass
class DetectionMetrics:
    """Data class for storing detection metrics."""
    class_name: str
    class_id: int
    total_frames_detected: int = 0
    total_appearances: int = 0  # Total bounding boxes across all frames
    first_frame: int = -1
    last_frame: int = -1
    frame_indices: List[int] = field(default_factory=list)
    
    # Calculated after processing
    appearance_time_seconds: float = 0.0
    percentage_of_video: float = 0.0
    avg_confidence: float = 0.0
    confidences: List[float] = field(default_factory=list)


@dataclass
class VideoAnalyticsResult:
    """Complete analytics result for a video."""
    video_path: str
    total_frames: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    
    # Detection stats
    frames_with_detections: int = 0
    frames_without_detections: int = 0
    total_detections: int = 0
    
    # Per-class metrics
    class_metrics: Dict[str, DetectionMetrics] = field(default_factory=dict)
    
    # Processing info
    processing_time_seconds: float = 0.0
    extracted_frames_dir: Optional[str] = None
    output_video_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'duration_seconds': self.duration_seconds,
            'resolution': f"{self.width}x{self.height}",
            'frames_with_detections': self.frames_with_detections,
            'frames_without_detections': self.frames_without_detections,
            'detection_rate_percent': round(
                (self.frames_with_detections / self.total_frames * 100) if self.total_frames > 0 else 0, 2
            ),
            'total_detections': self.total_detections,
            'processing_time_seconds': round(self.processing_time_seconds, 2),
            'extracted_frames_dir': self.extracted_frames_dir,
            'output_video_path': self.output_video_path,
            'class_metrics': {}
        }
        
        for class_name, metrics in self.class_metrics.items():
            result['class_metrics'][class_name] = {
                'class_id': metrics.class_id,
                'total_frames_detected': metrics.total_frames_detected,
                'total_appearances': metrics.total_appearances,
                'appearance_time_seconds': round(metrics.appearance_time_seconds, 2),
                'percentage_of_video': round(metrics.percentage_of_video, 2),
                'avg_confidence': round(metrics.avg_confidence, 3),
                'first_appearance_frame': metrics.first_frame,
                'last_appearance_frame': metrics.last_frame,
                'first_appearance_time': round(metrics.first_frame / self.fps, 2) if metrics.first_frame >= 0 else None,
                'last_appearance_time': round(metrics.last_frame / self.fps, 2) if metrics.last_frame >= 0 else None,
            }
        
        return result


class VideoAnalyzer:
    """
    Analyzes videos for logo/object detection with detailed metrics.
    
    Features:
    - Time-based metrics (total appearance time per class)
    - Frame-based metrics (percentage of frames with detections)
    - Frame extraction (save frames with detections)
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = None,
        iou_threshold: float = None
    ):
        """
        Initialize the video analyzer.
        
        Args:
            model_path: Path to YOLO model file (.pt)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
        """
        self.model_path = model_path or config.DEFAULT_MODEL
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or config.IOU_THRESHOLD
        self.model = None
        
        config.ensure_directories()
    
    def load_model(self) -> bool:
        """Load the YOLO model."""
        if YOLO is None:
            print("Error: ultralytics package not available")
            return False
        
        try:
            print(f"Loading model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a single frame.
        
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            if not self.load_model():
                return []
        
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = result.names[class_id]
                
                detections.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'label': class_name,
                    'confidence': conf,
                    'class_id': class_id
                })
        
        return detections
    
    def analyze_video(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path] = None,
        extract_frames: bool = True,
        save_annotated_video: bool = True,
        extract_every_n_frames: int = 1,
        max_extracted_frames: int = None
    ) -> VideoAnalyticsResult:
        """
        Analyze a video and compute detailed metrics.
        
        Args:
            video_path: Path to the input video
            output_dir: Directory for output files
            extract_frames: Whether to save frames with detections
            save_annotated_video: Whether to save annotated video
            extract_every_n_frames: Extract every N frames with detections (1 = all)
            max_extracted_frames: Maximum number of frames to extract (None = no limit)
            
        Returns:
            VideoAnalyticsResult with complete metrics
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Initialize result
        result = VideoAnalyticsResult(
            video_path=str(video_path),
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            width=width,
            height=height
        )
        
        # Prepare frame extraction directory
        frames_dir = None
        if extract_frames:
            frames_dir = output_dir / f"extracted_frames_{video_path.stem}"
            frames_dir.mkdir(parents=True, exist_ok=True)
            result.extracted_frames_dir = str(frames_dir)
        
        # Prepare output video
        video_writer = None
        if save_annotated_video:
            output_video_path = output_dir / f"analyzed_{video_path.name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            result.output_video_path = str(output_video_path)
        
        # Track metrics per class
        class_metrics: Dict[str, DetectionMetrics] = {}
        
        # Processing
        start_time = datetime.now()
        frame_idx = 0
        extracted_count = 0
        detection_frame_count = 0
        
        print(f"\nAnalyzing video: {video_path.name}")
        print(f"Duration: {duration:.2f}s | Frames: {total_frames} | FPS: {fps:.2f}")
        print("-" * 50)
        
        try:
            with tqdm(total=total_frames, desc="Analyzing frames", unit="frame") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Detect objects in frame
                    detections = self.detect_frame(frame)
                    
                    # Track frame-level detection
                    if detections:
                        result.frames_with_detections += 1
                        detection_frame_count += 1
                        classes_in_frame = set()
                        
                        for det in detections:
                            class_name = det['label']
                            class_id = det['class_id']
                            confidence = det['confidence']
                            
                            # Initialize metrics for new class
                            if class_name not in class_metrics:
                                class_metrics[class_name] = DetectionMetrics(
                                    class_name=class_name,
                                    class_id=class_id
                                )
                            
                            metrics = class_metrics[class_name]
                            metrics.total_appearances += 1
                            metrics.confidences.append(confidence)
                            
                            # Track unique frames per class
                            if class_name not in classes_in_frame:
                                classes_in_frame.add(class_name)
                                metrics.total_frames_detected += 1
                                metrics.frame_indices.append(frame_idx)
                                
                                # First/last frame tracking
                                if metrics.first_frame == -1:
                                    metrics.first_frame = frame_idx
                                metrics.last_frame = frame_idx
                        
                        result.total_detections += len(detections)
                        
                        # Extract frame if enabled
                        if extract_frames:
                            should_extract = (detection_frame_count % extract_every_n_frames == 0)
                            if max_extracted_frames is not None:
                                should_extract = should_extract and (extracted_count < max_extracted_frames)
                            
                            if should_extract:
                                annotated_frame = annotate_image(frame.copy(), detections)
                                frame_filename = f"frame_{frame_idx:06d}.jpg"
                                cv2.imwrite(str(frames_dir / frame_filename), annotated_frame)
                                extracted_count += 1
                    else:
                        result.frames_without_detections += 1
                    
                    # Write annotated frame to video
                    if video_writer is not None:
                        annotated = annotate_image(frame.copy(), detections) if detections else frame
                        video_writer.write(annotated)
                    
                    frame_idx += 1
                    pbar.update(1)
        
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
        
        # Calculate final metrics
        result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
        
        for class_name, metrics in class_metrics.items():
            # Time-based metrics
            metrics.appearance_time_seconds = metrics.total_frames_detected / fps if fps > 0 else 0
            metrics.percentage_of_video = (metrics.total_frames_detected / total_frames * 100) if total_frames > 0 else 0
            
            # Average confidence
            if metrics.confidences:
                metrics.avg_confidence = sum(metrics.confidences) / len(metrics.confidences)
        
        result.class_metrics = class_metrics
        
        # Print summary
        self._print_summary(result, extracted_count)
        
        return result
    
    def _print_summary(self, result: VideoAnalyticsResult, extracted_count: int):
        """Print analysis summary."""
        print("\n" + "=" * 60)
        print("VIDEO ANALYSIS COMPLETE")
        print("=" * 60)
        
        print(f"\n📹 Video: {Path(result.video_path).name}")
        print(f"   Duration: {result.duration_seconds:.2f}s | Frames: {result.total_frames}")
        
        print(f"\n📊 Detection Summary:")
        print(f"   Frames with detections: {result.frames_with_detections}/{result.total_frames} "
              f"({result.frames_with_detections/result.total_frames*100:.1f}%)")
        print(f"   Total detections: {result.total_detections}")
        
        if result.class_metrics:
            print(f"\n🏷️  Per-Class Metrics:")
            print("-" * 60)
            print(f"{'Class':<25} {'Frames':<10} {'Time (s)':<10} {'% Video':<10} {'Avg Conf':<10}")
            print("-" * 60)
            
            # Sort by appearance time
            sorted_metrics = sorted(
                result.class_metrics.values(),
                key=lambda x: x.appearance_time_seconds,
                reverse=True
            )
            
            for m in sorted_metrics:
                print(f"{m.class_name:<25} {m.total_frames_detected:<10} "
                      f"{m.appearance_time_seconds:<10.2f} {m.percentage_of_video:<10.1f} "
                      f"{m.avg_confidence:<10.3f}")
        
        print("-" * 60)
        print(f"\n⏱️  Processing time: {result.processing_time_seconds:.2f}s")
        
        if result.extracted_frames_dir:
            print(f"📁 Extracted frames ({extracted_count}): {result.extracted_frames_dir}")
        if result.output_video_path:
            print(f"🎬 Annotated video: {result.output_video_path}")
    
    def save_report(
        self,
        result: VideoAnalyticsResult,
        output_path: Union[str, Path] = None
    ) -> str:
        """
        Save analytics report to JSON file.
        
        Args:
            result: VideoAnalyticsResult object
            output_path: Path for JSON file (auto-generated if None)
            
        Returns:
            Path to saved report
        """
        if output_path is None:
            video_name = Path(result.video_path).stem
            output_path = config.OUTPUT_DIR / f"report_{video_name}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_dict = result.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Report saved: {output_path}")
        return str(output_path)


def analyze_video(
    video_path: Union[str, Path],
    model_path: str = None,
    output_dir: Union[str, Path] = None,
    extract_frames: bool = True,
    save_video: bool = True,
    confidence: float = None,
    save_report: bool = True
) -> VideoAnalyticsResult:
    """
    Convenience function to analyze a video with default settings.
    
    Args:
        video_path: Path to video file
        model_path: Path to YOLO model (uses default if None)
        output_dir: Output directory
        extract_frames: Whether to extract frames with detections
        save_video: Whether to save annotated video
        confidence: Confidence threshold
        save_report: Whether to save JSON report
        
    Returns:
        VideoAnalyticsResult object
    """
    analyzer = VideoAnalyzer(
        model_path=model_path,
        confidence_threshold=confidence
    )
    
    result = analyzer.analyze_video(
        video_path=video_path,
        output_dir=output_dir,
        extract_frames=extract_frames,
        save_annotated_video=save_video
    )
    
    if save_report:
        analyzer.save_report(result)
    
    return result
