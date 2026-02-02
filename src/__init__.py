# Logo Detection Pipeline
# Computer Vision Project - Team 4

__version__ = "1.0.0"

from .pipeline import DetectionPipeline, create_pipeline
from .video_analytics import VideoAnalyzer, VideoAnalyticsResult, analyze_video

__all__ = [
    "DetectionPipeline",
    "create_pipeline",
    "VideoAnalyzer",
    "VideoAnalyticsResult",
    "analyze_video",
]
