"""
Detection Router - Endpoints for logo detection
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import tempfile
import os
from pathlib import Path

from back.services.config import CONFIDENCE_THRESHOLD

router = APIRouter()

# Initialize pipeline lazily
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from back.services.pipeline import DetectionPipeline
        _pipeline = DetectionPipeline()
    return _pipeline


@router.post("/image")
async def detect_in_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
):
    """
    Detect logos in an uploaded image.
    
    Args:
        file: Image file (jpg, png, etc.)
        confidence: Detection confidence threshold (0.0-1.0)
    
    Returns:
        List of detections with bounding boxes and class names
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Run detection
        pipeline = get_pipeline()
        results = pipeline.detect(tmp_path, conf_threshold=confidence)
        
        # Clean up
        os.unlink(tmp_path)
        
        return {
            "filename": file.filename,
            "detections": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def detect_in_batch(
    files: List[UploadFile] = File(...),
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
):
    """
    Detect logos in multiple images.
    
    Args:
        files: List of image files
        confidence: Detection confidence threshold (0.0-1.0)
    
    Returns:
        List of detection results for each image
    """
    results = []
    pipeline = get_pipeline()
    
    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "error": "Not an image file"
            })
            continue
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            detections = pipeline.detect(tmp_path, conf_threshold=confidence)
            os.unlink(tmp_path)
            
            results.append({
                "filename": file.filename,
                "detections": detections
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}


@router.get("/classes")
async def get_classes():
    """Get list of detectable logo classes."""
    pipeline = get_pipeline()
    return {"classes": pipeline.get_class_names()}
