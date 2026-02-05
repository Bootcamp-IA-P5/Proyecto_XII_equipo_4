"""
Videos Router - Endpoints for video processing and analysis
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from typing import Optional
import tempfile
import os
from pathlib import Path
import uuid

from back.services.config import CONFIDENCE_THRESHOLD, OUTPUT_DIR

router = APIRouter()

# Store for background task results
_task_results = {}


def _get_video_analyzer():
    """Lazy load VideoAnalyzer to avoid heavy imports at startup."""
    from back.services.video_analytics import VideoAnalyzer
    return VideoAnalyzer()


def _get_video_downloader():
    """Lazy load VideoDownloader."""
    from back.services.video_downloader import VideoDownloader
    return VideoDownloader()


@router.post("/upload")
async def process_uploaded_video(
    file: UploadFile = File(...),
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
    frame_skip: int = Query(default=5, ge=1, le=30)
):
    """
    Process an uploaded video for logo detection.
    
    Args:
        file: Video file (mp4, avi, etc.)
        confidence: Detection confidence threshold
        frame_skip: Process every Nth frame
    
    Returns:
        Analysis results with metrics
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Analyze video
        analyzer = _get_video_analyzer()
        results = analyzer.analyze_video(
            video_path=tmp_path,
            conf_threshold=confidence,
            frame_skip=frame_skip
        )
        
        # Clean up
        os.unlink(tmp_path)
        
        return {
            "filename": file.filename,
            "analysis": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download")
async def download_and_process(
    url: str,
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
    frame_skip: int = Query(default=5, ge=1, le=30)
):
    """
    Download a video from URL and process it.
    
    Args:
        url: Video URL (YouTube, TikTok, Instagram, etc.)
        confidence: Detection confidence threshold
        frame_skip: Process every Nth frame
    
    Returns:
        Analysis results with metrics
    """
    try:
        # Download video
        downloader = _get_video_downloader()
        video_path = downloader.download(url)
        
        if not video_path:
            raise HTTPException(status_code=400, detail="Could not download video")
        
        # Analyze video
        analyzer = _get_video_analyzer()
        results = analyzer.analyze_video(
            video_path=video_path,
            conf_threshold=confidence,
            frame_skip=frame_skip
        )
        
        return {
            "url": url,
            "analysis": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-frames")
async def extract_frames_with_detections(
    file: UploadFile = File(...),
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
    max_frames: int = Query(default=10, ge=1, le=100)
):
    """
    Extract frames with logo detections from a video.
    
    Args:
        file: Video file
        confidence: Detection confidence threshold
        max_frames: Maximum number of frames to extract
    
    Returns:
        Paths to extracted frames
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Extract frames
        analyzer = _get_video_analyzer()
        output_dir = OUTPUT_DIR / "extracted_frames" / str(uuid.uuid4())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frames = analyzer.extract_frames_with_detections(
            video_path=tmp_path,
            output_dir=str(output_dir),
            conf_threshold=confidence,
            max_frames=max_frames
        )
        
        # Clean up video
        os.unlink(tmp_path)
        
        return {
            "frames_extracted": len(frames),
            "output_directory": str(output_dir),
            "frames": frames
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{video_id}")
async def get_video_metrics(video_id: str):
    """Get stored metrics for a processed video."""
    if video_id not in _task_results:
        raise HTTPException(status_code=404, detail="Video not found")
    return _task_results[video_id]
