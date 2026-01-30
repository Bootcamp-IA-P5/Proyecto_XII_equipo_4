# 🔌 API Reference & Developer Guide

## Table of Contents
1. [Pipeline Module](#pipeline-module)
2. [Video Processor](#video-processor)
3. [Video Downloader](#video-downloader)
4. [Database Module](#database-module)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)

---

## Pipeline Module

### `DetectionPipeline` Class

Main class for running YOLO object detection on images.

#### Constructor
```python
from src.pipeline import DetectionPipeline

pipeline = DetectionPipeline(
    model_path="yolov8n.pt",           # Path to YOLO model
    confidence_threshold=0.5,           # Min confidence for detections
    iou_threshold=0.45                 # IOU threshold for NMS
)
```

#### Methods

##### `load_model()`
Load the YOLO model.
```python
pipeline.load_model()
# Returns: bool (True if successful)
```

##### `detect(image, verbose=False)`
Run detection on a single image.
```python
detections = pipeline.detect(image)
# Returns: List[Dict] with keys: box, label, confidence, class_id
```

##### `detect_objects(image, confidence_threshold=None)`
Detect objects with different return format (used by VideoProcessor).
```python
detections = pipeline.detect_objects(image, confidence_threshold=0.3)
# Returns: List[Dict] with keys: class, confidence, bbox
```

##### `process_image(image_path, save_output=True, show_result=False, output_dir=None)`
Process a single image through the pipeline.
```python
result = pipeline.process_image(
    "data/input/logo.jpg",
    save_output=True,
    show_result=False,
    output_dir="data/output"
)
# Returns: Dict with detection results
```

##### `process_directory(input_dir=None, output_dir=None, recursive=False, save_output=True)`
Process all images in a directory.
```python
results = pipeline.process_directory(
    input_dir="data/input",
    output_dir="data/output",
    recursive=False
)
# Returns: List[Dict] with results for each image
```

---

## Video Processor

### `VideoProcessor` Class

Processes videos and detects brands/logos in each frame.

#### Constructor
```python
from src.video_processor import VideoProcessor

processor = VideoProcessor(pipeline=None)  # Uses default pipeline if None
```

#### Methods

##### `process_video(video_path, confidence_threshold=None, frame_skip=5, progress_callback=None)`
Analyze a video file for brand detection.

```python
results = processor.process_video(
    video_path="video.mp4",
    confidence_threshold=0.5,
    frame_skip=5,  # Process every 5th frame
    progress_callback=lambda p, msg: print(f"{p:.0%}: {msg}")
)
```

**Returns:**
```python
{
    'video_path': str,
    'video_name': str,
    'duration_seconds': float,
    'total_frames': int,
    'fps': float,
    'width': int,
    'height': int,
    'detections': [
        {
            'frame_number': int,
            'timestamp': float,
            'class': str,
            'confidence': float,
            'bbox': [x1, y1, x2, y2]
        },
        ...
    ],
    'class_statistics': {
        'BrandName': {
            'detections_count': int,
            'frames_detected': int,
            'total_time': float,
            'percentage': float,
            'avg_confidence': float,
            'max_confidence': float
        },
        ...
    }
}
```

##### `extract_cropped_detections(video_path, results, output_dir=None)`
Extract cropped images of detected objects.

```python
results = processor.extract_cropped_detections(
    video_path="video.mp4",
    results=results,
    output_dir="data/output/crops"
)
# Adds 'cropped_images' list to results
```

##### `generate_report(results)`
Generate a text report from detection results.

```python
report = processor.generate_report(results)
print(report)
# Returns: str (formatted report)
```

---

## Video Downloader

### `VideoDownloader` Class

Download videos from various social media platforms.

#### Constructor
```python
from src.video_downloader import VideoDownloader

downloader = VideoDownloader(output_dir="downloads")
```

#### Static Methods

##### `detect_platform(url)`
Detect which platform a URL belongs to.

```python
platform = VideoDownloader.detect_platform("https://youtube.com/watch?v=...")
# Returns: str ('youtube', 'instagram', 'tiktok', 'facebook', 'twitter', or None)
```

#### Methods

##### `download_youtube(url, video_name=None)`
Download from YouTube.
```python
result = downloader.download_youtube(
    "https://www.youtube.com/watch?v=...",
    video_name="my_video"
)
```

##### `download_instagram(url, video_name=None, username=None, password=None)`
Download from Instagram (requires login).
```python
result = downloader.download_instagram(
    "https://instagram.com/p/...",
    username="your_username",
    password="your_password"
)
```

##### `download_tiktok(url, video_name=None)`
Download from TikTok.
```python
result = downloader.download_tiktok("https://vm.tiktok.com/...")
```

##### `download_facebook(url, video_name=None)`
Download from Facebook.
```python
result = downloader.download_facebook("https://facebook.com/video/...")
```

##### `download_twitter(url, video_name=None)`
Download from Twitter/X.
```python
result = downloader.download_twitter("https://twitter.com/.../status/...")
```

##### `download(url, video_name=None, credentials=None)`
Universal download method (auto-detects platform).

```python
result = downloader.download(
    url="https://youtube.com/watch?v=...",
    video_name="my_video",
    credentials={'username': 'user', 'password': 'pass'}  # For Instagram
)
```

**Returns:**
```python
{
    'success': bool,
    'path': str,  # Path to downloaded video
    'platform': str,  # Platform name
    'error': str,  # Error message if failed
    # Plus platform-specific fields (title, duration, uploader, etc.)
}
```

---

## Database Module

### `DetectionDatabase` Class

Manages SQLite database operations for detection results.

#### Constructor
```python
from src.database import DetectionDatabase

db = DetectionDatabase(db_path="detections.db")
```

#### Methods

##### `add_video(...)`
Add a video record to the database.

```python
video_id = db.add_video(
    filename="video.mp4",
    source_path="/path/to/video.mp4",
    source_url=None,
    platform="local",
    duration_seconds=120.5,
    fps=30.0,
    width=1920,
    height=1080,
    total_frames=3615
)
# Returns: int (video_id)
```

##### `add_detections(video_id, detections)`
Add multiple detections to database.

```python
count = db.add_detections(video_id, [
    {
        'frame_number': 10,
        'timestamp': 0.33,
        'class': 'Nike',
        'confidence': 0.95,
        'bbox': [100, 150, 200, 250]
    },
    ...
])
# Returns: int (count of detections added)
```

##### `add_brand_statistics(video_id, brand_name, statistics)`
Add or update brand statistics.

```python
db.add_brand_statistics(
    video_id=1,
    brand_name="Nike",
    statistics={
        'detections_count': 45,
        'frames_detected': 30,
        'total_time': 10.5,
        'percentage': 15.5,
        'avg_confidence': 0.92,
        'max_confidence': 0.98
    }
)
```

##### `get_video(video_id)`
Get video information by ID.
```python
video = db.get_video(1)
# Returns: Dict or None
```

##### `get_detections(video_id)`
Get all detections for a video.
```python
detections = db.get_detections(1)
# Returns: List[Dict]
```

##### `get_brand_statistics(video_id)`
Get brand statistics for a video.
```python
stats = db.get_brand_statistics(1)
# Returns: Dict[str, Dict]  - {brand_name: statistics}
```

##### `get_all_videos(status=None)`
Get all videos, optionally filtered by status.
```python
videos = db.get_all_videos(status='completed')
# Returns: List[Dict]
```

##### `update_video_status(video_id, status, processed_at=None)`
Update video processing status.
```python
db.update_video_status(1, 'completed')
```

##### `get_brand_summary()`
Get summary statistics for all brands.
```python
summary = db.get_brand_summary()
# Returns: Dict with aggregate brand statistics
```

##### `export_video_report(video_id)`
Export complete report for a video.
```python
report = db.export_video_report(1)
# Returns: Dict with video, detections, and statistics
```

##### `delete_video(video_id)`
Delete a video and all associated data.
```python
success = db.delete_video(1)
# Returns: bool
```

##### `close()`
Close database connection.
```python
db.close()
```

---

## Configuration

### `config.py` Module

Global configuration settings.

#### Key Variables
```python
# Paths
BASE_DIR              # Project base directory
DATA_DIR              # Data directory
INPUT_DIR             # Input directory
OUTPUT_DIR            # Output directory
MODELS_DIR            # Models directory

# Image Settings
SUPPORTED_FORMATS     # List of supported image formats
DEFAULT_IMAGE_SIZE    # (width, height) for resizing
NORMALIZE_MEAN        # ImageNet normalization mean
NORMALIZE_STD         # ImageNet normalization std

# Model Settings
DEFAULT_MODEL         # Default YOLO model name
CONFIDENCE_THRESHOLD  # Default confidence threshold
IOU_THRESHOLD         # IOU threshold for NMS

# Visualization
DEFAULT_BOX_COLOR     # Default bounding box color
DEFAULT_TEXT_COLOR    # Default text color
```

#### Key Functions
```python
ensure_directories()   # Create necessary directories
```

---

## Usage Examples

### Example 1: Process a Single Image
```python
from src.pipeline import DetectionPipeline

# Initialize pipeline
pipeline = DetectionPipeline(confidence_threshold=0.5)
pipeline.load_model()

# Process image
result = pipeline.process_image("data/input/logo.jpg", save_output=True)

# Print results
print(f"Detections: {result['detection_count']}")
for det in result['detections']:
    print(f"  {det['label']}: {det['confidence']:.2%} at {det['box']}")
```

### Example 2: Analyze a Video
```python
from src.video_processor import VideoProcessor

processor = VideoProcessor()

# Process video
results = processor.process_video(
    "video.mp4",
    confidence_threshold=0.5,
    frame_skip=5
)

# Extract crops
results = processor.extract_cropped_detections("video.mp4", results)

# Print report
report = processor.generate_report(results)
print(report)
```

### Example 3: Download and Analyze
```python
from src.video_downloader import VideoDownloader
from src.video_processor import VideoProcessor

# Download video
downloader = VideoDownloader()
result = downloader.download("https://youtube.com/watch?v=...")

if result['success']:
    # Analyze video
    processor = VideoProcessor()
    analysis = processor.process_video(result['path'])
    print(f"Found {len(analysis['detections'])} detections")
```

### Example 4: Database Operations
```python
from src.database import DetectionDatabase

db = DetectionDatabase("detections.db")

# Add video
video_id = db.add_video(
    filename="video.mp4",
    duration_seconds=120,
    fps=30,
    width=1920,
    height=1080,
    total_frames=3600
)

# Add detections
detections = [
    {'frame_number': 10, 'timestamp': 0.33, 'class': 'Nike', 
     'confidence': 0.95, 'bbox': [100, 150, 200, 250]},
]
db.add_detections(video_id, detections)

# Add statistics
db.add_brand_statistics(video_id, 'Nike', {
    'detections_count': 45,
    'frames_detected': 30,
    'total_time': 10.5,
    'percentage': 15.5,
    'avg_confidence': 0.92,
    'max_confidence': 0.98
})

# Query results
videos = db.get_all_videos()
for v in videos:
    print(f"{v['filename']}: {v['status']}")

# Close connection
db.close()
```

### Example 5: Complete Workflow
```python
from src.pipeline import DetectionPipeline
from src.video_processor import VideoProcessor
from src.video_downloader import VideoDownloader
from src.database import DetectionDatabase

# Initialize components
pipeline = DetectionPipeline()
processor = VideoProcessor(pipeline)
downloader = VideoDownloader()
db = DetectionDatabase()

# Download video
result = downloader.download("https://youtube.com/watch?v=...")
if not result['success']:
    print(f"Download failed: {result['error']}")
    exit()

video_path = result['path']

# Analyze video
analysis = processor.process_video(video_path)
analysis = processor.extract_cropped_detections(video_path, analysis)

# Store in database
video_id = db.add_video(
    filename=result.get('title', 'video'),
    source_url="https://youtube.com/watch?v=...",
    platform="youtube",
    duration_seconds=analysis['duration_seconds'],
    fps=analysis['fps'],
    width=analysis['width'],
    height=analysis['height'],
    total_frames=analysis['total_frames']
)

db.add_detections(video_id, analysis['detections'])

for brand, stats in analysis['class_statistics'].items():
    db.add_brand_statistics(video_id, brand, stats)

# Export report
report = processor.generate_report(analysis)
print(report)

db.close()
```

---

## Error Handling

### Common Exceptions

```python
# FileNotFoundError - when video/image not found
try:
    results = processor.process_video("nonexistent.mp4")
except FileNotFoundError as e:
    print(f"File not found: {e}")

# ValueError - when video cannot be opened
try:
    results = processor.process_video("corrupted.mp4")
except ValueError as e:
    print(f"Cannot open video: {e}")

# Request exceptions - for video downloads
try:
    result = downloader.download("invalid_url")
except Exception as e:
    print(f"Download failed: {e}")
```

---

## Performance Tips

1. **Batch Processing**
   - Use `process_directory()` for multiple images
   - Reuse model instance for multiple detections

2. **Video Processing**
   - Increase `frame_skip` for faster processing (trade-off with accuracy)
   - Use GPU if available (see config.py)
   - Process lower resolution videos when possible

3. **Database**
   - Commit regularly to avoid data loss
   - Use indexes for faster queries (already configured)
   - Archive old data periodically

---

## Contributing

To add new features:

1. Add new methods to appropriate module
2. Update docstrings with description and parameters
3. Include type hints
4. Add error handling
5. Write unit tests in `tests/`
6. Update this API reference

---

**API Version**: 1.0  
**Last Updated**: January 2026  
**Team**: Computer Vision Bootcamp - Team 4
