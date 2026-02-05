"""
Visualization Router - API for generating annotated images
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import tempfile
import os
from pathlib import Path
import cv2
import io
import numpy as np

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


def _get_visualization_functions():
    """Lazy load visualization functions."""
    from back.services.visualization import draw_bounding_box, add_label
    return draw_bounding_box, add_label


@router.post("/annotate")
async def annotate_image_endpoint(
    file: UploadFile = File(...),
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
    show_labels: bool = Query(default=True),
    show_confidence: bool = Query(default=True)
):
    """
    Detect logos and return annotated image with bounding boxes.
    
    Args:
        file: Image file
        confidence: Detection confidence threshold
        show_labels: Show class labels on boxes
        show_confidence: Show confidence scores
    
    Returns:
        Annotated image as PNG
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        content = await file.read()
        nparr = cv2.imdecode(
            np.frombuffer(content, np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if nparr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Run detection
        pipeline = get_pipeline()
        
        # Save temp for detection
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, nparr)
            tmp_path = tmp.name
        
        results = pipeline.detect(tmp_path, conf_threshold=confidence)
        os.unlink(tmp_path)
        
        # Annotate image
        draw_bounding_box, add_label = _get_visualization_functions()
        annotated = nparr.copy()
        for det in results:
            bbox = (det['x1'], det['y1'], det['x2'], det['y2'])
            label = det['class_name']
            conf = det['confidence']
            
            annotated = draw_bounding_box(annotated, bbox)
            
            if show_labels:
                label_text = f"{label}"
                if show_confidence:
                    label_text += f" {conf:.2f}"
                annotated = add_label(annotated, label_text, (det['x1'], det['y1']))
        
        # Encode to PNG
        _, buffer = cv2.imencode('.png', annotated)
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crop-detections")
async def crop_detections(
    file: UploadFile = File(...),
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
    padding: int = Query(default=10, ge=0, le=50)
):
    """
    Detect logos and return cropped images of each detection.
    
    Args:
        file: Image file
        confidence: Detection confidence threshold
        padding: Padding around each crop in pixels
    
    Returns:
        List of base64 encoded cropped images
    """
    import base64
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        content = await file.read()
        nparr = cv2.imdecode(
            np.frombuffer(content, np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if nparr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        h, w = nparr.shape[:2]
        
        # Run detection
        pipeline = get_pipeline()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, nparr)
            tmp_path = tmp.name
        
        results = pipeline.detect(tmp_path, conf_threshold=confidence)
        os.unlink(tmp_path)
        
        # Crop each detection
        crops = []
        for i, det in enumerate(results):
            x1 = max(0, int(det['x1']) - padding)
            y1 = max(0, int(det['y1']) - padding)
            x2 = min(w, int(det['x2']) + padding)
            y2 = min(h, int(det['y2']) + padding)
            
            crop = nparr[y1:y2, x1:x2]
            _, buffer = cv2.imencode('.png', crop)
            b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            crops.append({
                "index": i,
                "class_name": det['class_name'],
                "confidence": det['confidence'],
                "bbox": [det['x1'], det['y1'], det['x2'], det['y2']],
                "image_base64": b64
            })
        
        return {"crops": crops}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
