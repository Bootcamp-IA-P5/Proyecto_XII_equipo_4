"""
Video Processing Module - Handles video analysis for brand detection
Extracts frames, performs detection, and generates reports
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sqlite3
from datetime import datetime
import os

from .pipeline import DetectionPipeline
from .visualization import annotate_image
from . import config


class VideoProcessor:
    """Processes videos and detects brands/logos in each frame."""
    
    def __init__(self, pipeline: DetectionPipeline = None):
        """
        Initialize the video processor.
        
        Args:
            pipeline: DetectionPipeline instance for brand detection
        """
        self.pipeline = pipeline or DetectionPipeline()
        
    def process_video(
        self,
        video_path: str,
        confidence_threshold: float = None,
        frame_skip: int = 5,
        progress_callback=None
    ) -> Dict:
        """
        Process a video and detect brands in frames.
        
        Args:
            video_path: Path to video file
            confidence_threshold: Detection confidence threshold
            frame_skip: Process every Nth frame (default 5 for performance)
            progress_callback: Function to report progress
            
        Returns:
            Dictionary with detection results and statistics
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        results = {
            'video_path': video_path,
            'video_name': Path(video_path).name,
            'duration_seconds': duration,
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'detections': [],
            'frame_detections': {},  # frame_number -> list of detections
            'class_statistics': {},  # class_name -> stats
            'processing_timestamp': datetime.now().isoformat()
        }
        
        frame_count = 0
        detected_frames = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_skip == 0:
                detections = self.pipeline.detect_objects(frame, confidence_threshold)
                
                if detections:
                    detected_frames += 1
                    frame_time = frame_count / fps
                    
                    frame_detections_list = []
                    
                    for detection in detections:
                        detection_info = {
                            'frame_number': frame_count,
                            'timestamp': frame_time,
                            'class': detection.get('class'),
                            'confidence': float(detection.get('confidence', 0)),
                            'bbox': detection.get('bbox'),  # [x1, y1, x2, y2]
                        }
                        
                        # Update class statistics
                        class_name = detection.get('class', 'Unknown')
                        if class_name not in results['class_statistics']:
                            results['class_statistics'][class_name] = {
                                'detections_count': 0,
                                'frames_detected': 0,
                                'total_time': 0,
                                'avg_confidence': 0,
                                'max_confidence': 0,
                                'confidences': []
                            }
                        
                        stats = results['class_statistics'][class_name]
                        stats['detections_count'] += 1
                        stats['confidences'].append(detection_info['confidence'])
                        stats['max_confidence'] = max(stats['max_confidence'], detection_info['confidence'])
                        
                        frame_detections_list.append(detection_info)
                        results['detections'].append(detection_info)
                    
                    results['frame_detections'][frame_count] = frame_detections_list
            
            frame_count += 1
            
            # Progress callback
            if progress_callback and frame_count % 30 == 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_callback(progress, f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        
        # Calculate final statistics
        for class_name, stats in results['class_statistics'].items():
            if stats['confidences']:
                stats['avg_confidence'] = np.mean(stats['confidences'])
                stats['frames_detected'] = len([f for f in results['frame_detections'].values() 
                                               if any(d['class'] == class_name for d in f)])
                stats['total_time'] = stats['frames_detected'] * (frame_skip / fps)
                stats['percentage'] = (stats['frames_detected'] / (total_frames // frame_skip)) * 100
            del stats['confidences']  # Remove temporary list
        
        results['detected_frames'] = detected_frames
        results['frame_skip'] = frame_skip
        
        return results
    
    def extract_cropped_detections(
        self,
        video_path: str,
        results: Dict,
        output_dir: str = None
    ) -> Dict:
        """
        Extract cropped images of detected objects.
        
        Args:
            video_path: Path to video file
            results: Results from process_video
            output_dir: Directory to save cropped images
            
        Returns:
            Updated results with paths to cropped images
        """
        if output_dir is None:
            output_dir = Path(config.OUTPUT_DIR) / "crops"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return results
        
        results['cropped_images'] = []
        
        for frame_num, detections in results['frame_detections'].items():
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ensure valid coordinates
                x1, x2 = max(0, x1), min(frame.shape[1], x2)
                y1, y2 = max(0, y1), min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                cropped = frame[y1:y2, x1:x2]
                
                # Save cropped image
                class_name = detection['class']
                crop_name = f"{Path(video_path).stem}_f{frame_num}_{class_name}_{i}.jpg"
                crop_path = output_dir / crop_name
                
                cv2.imencode('.jpg', cropped)[1].tofile(str(crop_path))
                
                results['cropped_images'].append({
                    'path': str(crop_path),
                    'frame_number': frame_num,
                    'class': class_name,
                    'confidence': detection['confidence']
                })
        
        cap.release()
        return results
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate a text report from detection results.
        
        Args:
            results: Results from process_video
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("BRAND DETECTION REPORT")
        report.append("=" * 70)
        report.append(f"Video: {results['video_name']}")
        report.append(f"Duration: {results['duration_seconds']:.2f} seconds")
        report.append(f"Total Frames: {results['total_frames']}")
        report.append(f"FPS: {results['fps']:.2f}")
        report.append(f"Resolution: {results['width']}x{results['height']}")
        report.append(f"Processed: {results['processing_timestamp']}")
        report.append("")
        
        report.append("DETECTION SUMMARY")
        report.append("-" * 70)
        report.append(f"Total Detections: {len(results['detections'])}")
        report.append(f"Frames with Detections: {results['detected_frames']}")
        report.append("")
        
        if results['class_statistics']:
            report.append("BRAND STATISTICS")
            report.append("-" * 70)
            
            for class_name, stats in results['class_statistics'].items():
                report.append(f"\n{class_name}:")
                report.append(f"  Detections: {stats['detections_count']}")
                report.append(f"  Frames Detected: {stats['frames_detected']}")
                report.append(f"  Duration: {stats['total_time']:.2f} seconds")
                report.append(f"  Percentage: {stats['percentage']:.2f}%")
                report.append(f"  Avg Confidence: {stats['avg_confidence']:.2%}")
                report.append(f"  Max Confidence: {stats['max_confidence']:.2%}")
        else:
            report.append("No brands detected")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
