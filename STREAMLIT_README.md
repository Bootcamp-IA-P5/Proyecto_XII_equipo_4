# 🎯 Brand Logo Detection System - Streamlit Web Interface

## Overview

The Streamlit application provides an interactive web interface for analyzing videos to detect and track brand logos. It supports multiple input methods including local file uploads and video downloads from popular social media platforms.

## Features

### 📹 Video Upload & Analysis
- **Local Video Upload**: Upload MP4, AVI, MOV, MKV, FLV, WMV, WebM files
- **Real-time Video Info**: Displays duration, FPS, resolution, and frame count
- **Customizable Analysis**: Configure confidence thresholds and frame skip rates
- **Cropped Image Extraction**: Automatically extracts and saves detected logo regions
- **Database Integration**: Save all results for future reference

### 🔗 Social Media Integration
- **YouTube**: Download and analyze full YouTube videos
- **Instagram**: Download Instagram posts and reels (with authentication)
- **TikTok**: Download TikTok videos
- **Facebook**: Download Facebook videos
- **Twitter/X**: Download tweets with video content

### 📊 Comprehensive Reporting
- **Detection Statistics**: Total detections, detection rate, timing data
- **Brand Analytics**: Per-brand detection count, screen time, and confidence metrics
- **Visual Results**: Display of detected logo samples
- **Exportable Reports**: Download reports as TXT or JSON formats

### 💾 Database Management
- **SQLite Database**: Local persistent storage of all detection results
- **Video Tracking**: Keep history of all analyzed videos
- **Brand Summary**: Aggregate statistics across all videos
- **Data Export**: Query and export detection data

## Installation

### Requirements
- Python 3.8+
- 4GB+ RAM recommended
- GPU optional but recommended for faster processing

### Setup

1. **Clone the repository**
```bash
cd Proyecto_XII_equipo_4
```

2. **Create a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -m streamlit --version
```

## Running the Application

### Start the Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Access the Web Interface
- **Main URL**: `http://localhost:8501`
- **Restart on File Changes**: Enabled by default
- **Stop Server**: Press `Ctrl+C` in terminal

## Usage Guide

### Tab 1: Upload & Analyze

1. **Upload a Video**
   - Click "Choose a video file" button
   - Select MP4, AVI, MOV, or other supported format
   - Video information (duration, resolution, FPS) displays automatically

2. **Configure Analysis**
   - Adjust confidence threshold (sidebar)
   - Set frame skip rate for faster processing
   - Enable/disable cropped image extraction
   - Choose to save results to database

3. **Analyze**
   - Click "🔍 Analyze Video" button
   - Monitor progress in real-time
   - View results immediately upon completion

### Tab 2: Social Media Links

1. **Add Video Link**
   - Paste URL from YouTube, Instagram, TikTok, Facebook, or Twitter
   - App automatically detects platform
   - For Instagram, provide login credentials when prompted

2. **Download & Analyze**
   - Click "⬇️ Download Video" button
   - Video downloads and optionally analyzes automatically
   - Results appear in Results & Reports tab

### Tab 3: Results & Reports

1. **View Results**
   - See video statistics (duration, total detections, etc.)
   - Expand brand tabs for detailed per-brand analysis
   - View confidence metrics and screen time percentages

2. **Cropped Images**
   - Preview detected logo samples
   - See detection confidence for each image
   - First 9 samples displayed (more available)

3. **Generate Reports**
   - Comprehensive text report with all details
   - Download as TXT for sharing
   - Download as JSON for further processing

### Tab 4: Database

1. **View Statistics**
   - Total videos processed
   - Total detections across all videos
   - Number of unique brands detected

2. **Brand Summary**
   - Aggregate data for each brand
   - Detection count across all videos
   - Average confidence scores
   - Total screen time per brand

3. **Video Management**
   - Expand video entries to see details
   - View per-video brand breakdown
   - Delete individual videos if needed

## Configuration

### Sidebar Settings

**Model Settings:**
- **Confidence Threshold** (0.1-1.0): Minimum score for detections
- **Frame Skip** (1-30): Process every Nth frame
  - Lower = more thorough but slower
  - Higher = faster but may miss detections

### Advanced Settings

Edit `src/config.py` to modify:
- Default YOLO model
- Input/output directories
- Image preprocessing settings
- Normalization parameters

## Database Schema

### tables

**videos**
- id, filename, source_path, source_url, platform
- duration_seconds, fps, width, height, total_frames
- uploaded_at, processed_at, status

**detections**
- id, video_id, frame_number, timestamp_seconds
- class_name, confidence, bbox_x1/y1/x2/y2
- detection_time

**brand_statistics**
- id, video_id, brand_name
- total_detections, frames_detected, duration_seconds
- percentage, avg_confidence, max_confidence
- calculated_at

**cropped_images**
- id, detection_id, video_id, image_path
- class_name, confidence, saved_at

## File Structure

```
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── src/
│   ├── pipeline.py               # Detection pipeline
│   ├── video_processor.py        # Video analysis module
│   ├── video_downloader.py       # Social media download module
│   ├── database.py               # SQLite database module
│   ├── config.py                 # Configuration settings
│   ├── image_loader.py           # Image loading utilities
│   ├── preprocessing.py          # Image preprocessing
│   └── visualization.py          # Visualization utilities
├── data/
│   ├── input/                    # Input videos
│   └── output/                   # Output results and crops
└── models/                       # YOLO model files
```

## Troubleshooting

### Common Issues

**Issue: "No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**Issue: Streamlit not found**
```bash
pip install streamlit
```

**Issue: Video download fails**
- Check internet connection
- Update yt-dlp: `pip install --upgrade yt-dlp`
- Try different video URL
- Some videos may have regional restrictions

**Issue: Low detection accuracy**
- Increase confidence threshold gradually (start at 0.3)
- Ensure video quality is sufficient
- Check if model is trained on similar logos

**Issue: Slow processing**
- Increase frame skip value
- Lower video resolution
- Use GPU if available (configure in config.py)

## Performance Tips

1. **Faster Processing**
   - Increase frame skip to 10-15 for quick overview
   - Lower video resolution before uploading
   - Disable cropped image extraction if not needed

2. **Better Accuracy**
   - Use confidence threshold 0.4-0.6 for optimal balance
   - Enable all detections (lower confidence)
   - Process high-quality videos

3. **Database Performance**
   - Regularly export and archive old results
   - Delete completed videos if storage limited
   - Use database cleanup periodically

## Advanced Features

### Custom Model
To use a custom trained model:
1. Place `.pt` file in `models/` directory
2. Update `config.py` DEFAULT_MODEL path
3. Restart Streamlit app

### Instagram Download
Requires:
- Valid Instagram account
- Username and password
- 2FA may block automatic login

### GPU Acceleration
To enable GPU (CUDA):
1. Install CUDA Toolkit
2. Install GPU PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
3. Update `config.py` to enable GPU

## Export & Integration

### Export Formats
- **TXT Reports**: Human-readable detection reports
- **JSON Results**: Machine-readable complete results
- **Database Queries**: SQL export of detection data
- **Cropped Images**: High-quality logo samples

### API Integration
Database can be queried directly:
```python
from src.database import DetectionDatabase

db = DetectionDatabase('detections.db')
videos = db.get_all_videos()
brand_summary = db.get_brand_summary()
```

## Support & Contributing

For issues or improvements:
1. Check GitHub issues
2. Review troubleshooting section
3. Submit detailed bug reports
4. Include video samples if possible

## Team & Credits

**Project**: Brand Logo Detection System  
**Team**: Computer Vision Bootcamp - Team 4  
**Technologies**: YOLO, Streamlit, OpenCV, SQLite  

## License

See LICENSE file for details.

---

**Last Updated**: January 2026  
**Version**: 1.0.0
