"""
Streamlit Application for Brand Logo Detection
Interactive web interface for video upload, analysis, and reporting
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
from datetime import datetime
import json
import logging
import sys
import re
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
from back.services.pipeline import DetectionPipeline
from back.services.video_processor import VideoProcessor
from back.services.video_downloader import VideoDownloader
from back.services import config
from database.mysql_db import MySQLDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _build_crop_index(cropped_images: Optional[List[Dict]]) -> Dict[Tuple[int, str], List[Dict]]:
    index: Dict[Tuple[int, str], List[Dict]] = {}
    if not cropped_images:
        return index

    for item in cropped_images:
        frame_number = item.get("frame_number")
        class_name = item.get("class")
        if frame_number is None or class_name is None:
            continue
        key = (int(frame_number), str(class_name))
        index.setdefault(key, []).append(item)
    return index


def _attach_crops_to_detections(results: Dict) -> List[Dict]:
    detections = results.get("detections", [])
    crop_index = _build_crop_index(results.get("cropped_images"))
    review_items: List[Dict] = []

    for idx, detection in enumerate(detections):
        frame_number = detection.get("frame_number")
        class_name = detection.get("class")
        crop_path = None
        if frame_number is not None and class_name is not None:
            key = (int(frame_number), str(class_name))
            if key in crop_index and crop_index[key]:
                crop_path = crop_index[key].pop(0).get("path")

        review_items.append({
            **detection,
            "crop_path": crop_path,
            "review_id": idx
        })

    return review_items


def _build_frame_detections(detections: List[Dict]) -> Dict[int, List[Dict]]:
    frame_detections: Dict[int, List[Dict]] = {}
    for detection in detections:
        frame_number = detection.get("frame_number")
        if frame_number is None:
            continue
        frame_detections.setdefault(int(frame_number), []).append(detection)
    return frame_detections


def _compute_class_statistics(
    detections: List[Dict],
    frame_detections: Dict[int, List[Dict]],
    total_frames: int,
    fps: float,
    frame_skip: int
) -> Dict[str, Dict]:
    class_statistics: Dict[str, Dict] = {}

    for detection in detections:
        class_name = detection.get("class", "Unknown")
        confidence = float(detection.get("confidence", 0))

        if class_name not in class_statistics:
            class_statistics[class_name] = {
                "detections_count": 0,
                "frames_detected": 0,
                "total_time": 0,
                "avg_confidence": 0,
                "max_confidence": 0,
                "confidences": []
            }

        stats = class_statistics[class_name]
        stats["detections_count"] += 1
        stats["confidences"].append(confidence)
        stats["max_confidence"] = max(stats["max_confidence"], confidence)

    for class_name, stats in class_statistics.items():
        confidences = stats.get("confidences", [])
        if confidences:
            stats["avg_confidence"] = float(np.mean(confidences))
            stats["frames_detected"] = len([
                frame for frame, dets in frame_detections.items()
                if any(d.get("class") == class_name for d in dets)
            ])
            if fps > 0:
                stats["total_time"] = stats["frames_detected"] * (frame_skip / fps)
            if total_frames > 0 and frame_skip > 0:
                denominator = total_frames // frame_skip
                stats["percentage"] = (stats["frames_detected"] / denominator * 100) if denominator else 0
        if "confidences" in stats:
            del stats["confidences"]

    return class_statistics


def _build_verified_results(base_results: Dict, verified_detections: List[Dict]) -> Dict:
    cleaned_detections = [
        {k: v for k, v in det.items() if k not in {"crop_path", "review_id"}}
        for det in verified_detections
    ]

    frame_detections = _build_frame_detections(cleaned_detections)
    detected_frames = len(frame_detections)
    total_frames = int(base_results.get("total_frames", 0))
    fps = float(base_results.get("fps", 0))
    frame_skip = int(base_results.get("frame_skip", 1))

    verified_results = dict(base_results)
    verified_results["detections"] = cleaned_detections
    verified_results["frame_detections"] = frame_detections
    verified_results["detected_frames"] = detected_frames
    verified_results["class_statistics"] = _compute_class_statistics(
        cleaned_detections,
        frame_detections,
        total_frames,
        fps,
        frame_skip
    )

    if verified_results.get("cropped_images"):
        verified_results["cropped_images"] = [
            crop for crop in verified_results["cropped_images"]
            if any(
                crop.get("frame_number") == det.get("frame_number") and
                crop.get("class") == det.get("class")
                for det in cleaned_detections
            )
        ]

    return verified_results


def save_verified_detections() -> int:
    detections = st.session_state.get("pending_detections", [])
    verified = [
        det for det in detections
        if st.session_state.review_selection.get(det.get("review_id"), True)
    ]

    if not verified:
        return 0

    if st.session_state.pending_save_to_db and st.session_state.pending_video_id:
        inserted = st.session_state.database.add_detections(
            st.session_state.pending_video_id,
            [
                {k: v for k, v in det.items() if k not in {"crop_path", "review_id"}}
                for det in verified
            ]
        )
        return inserted

    return 0


def _render_review_ui(prefix: str = "review") -> None:
    """Render review gallery with checkboxes and confirmation button in a form.
    
    This function ONLY renders the UI - it does NOT save to database.
    Database save only happens when save_verified_detections() is explicitly called.
    """
    if not st.session_state.get("pending_results") or not st.session_state.get("pending_detections"):
        return

    st.subheader("🧪 Review Detections")
    st.write("Uncheck false positives, then confirm to save only verified detections.")

    detections = st.session_state.pending_detections
    
    # Use a form to prevent reruns when checkboxes are clicked
    with st.form(key=f"{prefix}_review_form"):
        cols = st.columns(3)

        for idx, det in enumerate(detections):
            review_id = det.get("review_id", idx)
            key = f"{prefix}_keep_{review_id}"

            # Initialize checkbox value from session state
            if key not in st.session_state:
                st.session_state[key] = st.session_state.review_selection.get(review_id, True)

            with cols[idx % 3]:
                crop_path = det.get("crop_path")
                if crop_path and Path(crop_path).exists():
                    img = cv2.imread(crop_path)
                    if img is not None:
                        st.image(
                            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            caption=f"{det.get('class', 'Unknown')} · {det.get('confidence', 0):.1%}",
                            width="stretch"
                        )
                else:
                    st.write("No crop available")

                # Checkbox inside form - value is preserved between reruns
                keep = st.checkbox(
                    label="Keep detection",
                    value=st.session_state[key],
                    key=key
                )
                st.session_state.review_selection[review_id] = keep

                st.caption(
                    f"Frame {det.get('frame_number', 'N/A')} · "
                    f"{det.get('class', 'Unknown')} · "
                    f"{det.get('confidence', 0):.1%}"
                )

        st.divider()

        # Submit button inside form - only executes when clicked
        submitted = st.form_submit_button("✅ Confirm & Save to Database", use_container_width=True)
        
        if submitted:
            verified = [
                det for det in detections
                if st.session_state.review_selection.get(det.get("review_id"), True)
            ]

            verified_results = _build_verified_results(st.session_state.pending_results, verified)
            st.session_state.verified_results = verified_results
            st.session_state.last_results = verified_results

            # CRITICAL: Only call save_verified_detections() when button is submitted
            inserted = save_verified_detections()
            if st.session_state.pending_save_to_db and st.session_state.pending_video_id:
                st.success(f"✅ Saved {inserted} verified detections to the database.")
            else:
                st.info("✅ Verified detections stored in session (database save disabled).")

            st.session_state.pending_results = None
            st.session_state.pending_detections = []
            st.session_state.pending_video_id = None
            st.session_state.pending_save_to_db = False
            st.session_state.analysis_complete = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Brand Logo Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = DetectionPipeline()

if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor(st.session_state.pipeline)

if 'video_downloader' not in st.session_state:
    st.session_state.video_downloader = VideoDownloader()

if 'database' not in st.session_state:
    st.session_state.database = MySQLDatabase()

if 'processing_video' not in st.session_state:
    st.session_state.processing_video = False

if 'pending_results' not in st.session_state:
    st.session_state.pending_results = None

if 'pending_detections' not in st.session_state:
    st.session_state.pending_detections = []

if 'pending_video_id' not in st.session_state:
    st.session_state.pending_video_id = None

if 'pending_save_to_db' not in st.session_state:
    st.session_state.pending_save_to_db = False

if 'verified_results' not in st.session_state:
    st.session_state.verified_results = None

if 'review_selection' not in st.session_state:
    st.session_state.review_selection = {}

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("Model Settings")
    confidence = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    frame_skip = st.slider(
        "Frame Skip",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Process every Nth frame for faster analysis"
    )
    
    st.divider()
    
    st.subheader("📊 Database")
    if st.session_state.database.test_connection():
        st.success("MySQL connected")
    else:
        st.error("MySQL connection failed")
    if st.button("View Database Stats", width="stretch"):
        st.session_state.show_db_stats = not st.session_state.get('show_db_stats', False)
    
    if st.button("Export Database", width="stretch"):
        st.info("Database export feature coming soon!")

# ============================================================================
# MAIN APP HEADER
# ============================================================================

st.title("🎯 Brand Logo Detection System")
st.markdown("""
Detect and track brand logos in videos from YouTube, Instagram, TikTok, Facebook, and local uploads.
Analyze screen time and generate detailed reports for marketing insights.
""")

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📹 Upload & Analyze",
    "🔗 Social Media Links",
    "📊 Results & Reports",
    "💾 Database"
])

# ============================================================================
# TAB 1: UPLOAD & ANALYZE
# ============================================================================

with tab1:
    st.header("Upload & Analyze Videos")

    if st.session_state.analysis_complete:
        with st.container():
            _render_review_ui(prefix="upload_review")
            st.divider()
    
    col1, col2 = st.columns(2)
    
    with col2:
        st.subheader("📹 Preview")
        preview_placeholder = st.empty()
    
    with col1:
        st.subheader("📤 Upload Local Video")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv", "flv", "wmv", "webm"]
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            st.success(f"Video uploaded: {uploaded_file.name}")
            
            # Display video info
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0
                cap.release()
                
                col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                with col_info1:
                    st.metric("Duration", f"{duration:.1f}s")
                with col_info2:
                    st.metric("FPS", f"{fps:.1f}")
                with col_info3:
                    st.metric("Resolution", f"{width}x{height}")
                with col_info4:
                    st.metric("Frames", total_frames)
            
            # Analysis options
            st.subheader("Analysis Options")
            
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                extract_crops = st.checkbox(
                    "Extract Cropped Images",
                    value=True,
                    help="Save cropped regions of detected logos"
                )
            
            with col_opt2:
                save_to_db = st.checkbox(
                    "Save to Database",
                    value=True,
                    help="Store detection results in database"
                )
            
            # Analyze button
            if st.button("🔍 Analyze Video", width="stretch", type="primary"):
                st.session_state.processing_video = True
                st.session_state.analysis_complete = False
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                preview_cap = cv2.VideoCapture(video_path)
                
                def progress_callback(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)

                    match = re.search(r"Processed (\d+)/(\d+) frames", message)
                    if match and preview_cap.isOpened():
                        frame_number = int(match.group(1))
                        preview_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = preview_cap.read()
                        if ret:
                            preview_placeholder.image(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                width="stretch"
                            )
                
                try:
                    status_text.text("Processing video...")
                    
                    # Process video
                    results = st.session_state.video_processor.process_video(
                        video_path,
                        confidence_threshold=confidence,
                        frame_skip=frame_skip,
                        progress_callback=progress_callback
                    )
                    
                    status_text.text("Extracting detections...")
                    
                    # Extract cropped images if requested
                    if extract_crops:
                        results = st.session_state.video_processor.extract_cropped_detections(
                            video_path,
                            results
                        )
                    
                    # CRITICAL: Only save video METADATA here, NOT detections
                    # All detections store in session state for manual verification
                    # Database INSERT of detections only happens when user clicks "Confirm & Save"
                    video_id = None
                    if save_to_db:
                        status_text.text("Saving video metadata...")
                        # This ONLY creates the video record - NO detections are saved yet
                        video_id = st.session_state.database.add_video(
                            nombre=uploaded_file.name,
                            duracion_seg=results['duration_seconds']
                        )
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis complete!")
                    
                    # Display results
                    st.success("Video analysis completed successfully!")

                    # Store pending results for manual verification
                    st.session_state.pending_results = results
                    st.session_state.pending_detections = _attach_crops_to_detections(results)
                    st.session_state.pending_video_id = video_id
                    st.session_state.pending_save_to_db = save_to_db
                    st.session_state.review_selection = {}
                    st.session_state.analysis_complete = True

                    st.session_state.last_results = results
                    st.session_state.last_video_id = video_id
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    logger.error(f"Error processing video: {e}")
                
                finally:
                    st.session_state.processing_video = False
                    if preview_cap.isOpened():
                        preview_cap.release()
                    # Clean up temporary file
                    if Path(video_path).exists():
                        os.remove(video_path)


# ============================================================================
# TAB 2: SOCIAL MEDIA LINKS
# ============================================================================

with tab2:
    st.header("Analyze Videos from Social Media")
    st.markdown("""
    Download videos from YouTube, Instagram, TikTok, Facebook, and Twitter for analysis.
    """)

    if st.session_state.analysis_complete:
        with st.container():
            _render_review_ui(prefix="social_review")
            st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Add Video Link")
        
        video_url = st.text_input(
            "Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the link to a video from YouTube, Instagram, TikTok, Facebook, or Twitter"
        )
        
        custom_name = st.text_input(
            "Custom Video Name (Optional)",
            placeholder="my_brand_video",
            help="Custom name for the downloaded video"
        )
        
        # Platform-specific options
        if video_url:
            platform = st.session_state.video_downloader.detect_platform(video_url)
            if platform:
                st.success(f"✅ Detected platform: **{platform.upper()}**")
                
                if platform == 'instagram':
                    st.warning("⚠️ Instagram requires login credentials")
                    with st.expander("Instagram Credentials"):
                        ig_username = st.text_input("Instagram Username", type="default")
                        ig_password = st.text_input("Instagram Password", type="password")
            else:
                st.error("❌ Platform not detected or not supported")
        
        # Download button
        if st.button("⬇️ Download Video", width="stretch", type="primary"):
            if not video_url:
                st.error("Please enter a video URL")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Detecting platform...")
                    progress_bar.progress(25)
                    
                    platform = st.session_state.video_downloader.detect_platform(video_url)
                    
                    if not platform:
                        st.error("Platform not supported")
                    else:
                        status_text.text(f"Downloading from {platform}...")
                        progress_bar.progress(50)
                        
                        credentials = None
                        if platform == 'instagram':
                            credentials = {
                                'username': st.session_state.get('ig_username'),
                                'password': st.session_state.get('ig_password')
                            }
                        
                        result = st.session_state.video_downloader.download(
                            video_url,
                            video_name=custom_name,
                            credentials=credentials
                        )
                        
                        progress_bar.progress(75)
                        
                        if result['success']:
                            status_text.text("Download complete!")
                            progress_bar.progress(100)
                            
                            st.success(f"✅ Video downloaded successfully!")
                            st.json(result)
                            
                            # Store for analysis
                            st.session_state.downloaded_video_path = result['path']
                            st.session_state.downloaded_video_platform = result.get('platform')
                        else:
                            st.error(f"❌ Download failed: {result.get('error')}")
                
                except Exception as e:
                    st.error(f"Error downloading video: {str(e)}")
                    logger.error(f"Error downloading video: {e}")
    
    with col2:
        st.subheader("📊 Supported Platforms")
        
        platforms_info = {
            "🔴 YouTube": "Full support - Public and private videos",
            "📷 Instagram": "Posts and Reels - Requires credentials",
            "🎵 TikTok": "Full support - TikTok videos",
            "📘 Facebook": "Full support - Facebook videos",
            "𝕏 Twitter/X": "Full support - Twitter videos"
        }
        
        for platform, info in platforms_info.items():
            st.write(f"**{platform}**: {info}")
        
        st.divider()
        st.subheader("📥 Analysis Settings")
        
        # Analysis options for downloaded video
        if st.session_state.get('downloaded_video_path'):
            st.success(f"✅ Video ready for analysis")
            
            extract_crops_sm = st.checkbox(
                "Extract Cropped Images",
                value=True,
                key="extract_crops_sm",
                help="Save cropped regions of detected logos"
            )
            
            save_to_db_sm = st.checkbox(
                "Save to Database",
                value=True,
                key="save_to_db_sm",
                help="Store detection results in database"
            )
            
            if st.button("🔍 Analyze Downloaded Video", width="stretch", type="primary"):
                st.session_state.processing_video = True
                st.session_state.analysis_complete = False
                
                video_path = st.session_state.downloaded_video_path
                
                # Display video info first
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                    
                    st.write("**Video Information:**")
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("Duration", f"{duration:.1f}s")
                    with col_info2:
                        st.metric("Resolution", f"{width}x{height}")
                    with col_info3:
                        st.metric("FPS", f"{fps:.1f}")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                try:
                    status_text.text("🔄 Processing video...")
                    
                    # Process video
                    results = st.session_state.video_processor.process_video(
                        video_path,
                        confidence_threshold=confidence,
                        frame_skip=frame_skip,
                        progress_callback=progress_callback
                    )
                    
                    status_text.text("🖼️ Extracting detections...")
                    
                    # Extract cropped images if requested
                    if extract_crops_sm:
                        results = st.session_state.video_processor.extract_cropped_detections(
                            video_path,
                            results
                        )
                    
                    # CRITICAL: Only save video METADATA here, NOT detections
                    # All detections store in session state for manual verification
                    # Database INSERT of detections only happens when user clicks "Confirm & Save"
                    video_id = None
                    if save_to_db_sm:
                        status_text.text("💾 Saving video metadata...")
                        # This ONLY creates the video record - NO detections are saved yet
                        video_name = Path(video_path).name
                        
                        video_id = st.session_state.database.add_video(
                            nombre=video_name,
                            duracion_seg=results['duration_seconds']
                        )
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis complete!")
                    
                    # Display results
                    st.success("Video analysis completed successfully!")
                    
                    # Store pending results for manual verification
                    st.session_state.pending_results = results
                    st.session_state.pending_detections = _attach_crops_to_detections(results)
                    st.session_state.pending_video_id = video_id
                    st.session_state.pending_save_to_db = save_to_db_sm
                    st.session_state.review_selection = {}
                    st.session_state.analysis_complete = True

                    st.session_state.last_results = results
                    st.session_state.last_video_id = video_id
                    
                    # Show quick summary
                    st.write("**Quick Summary:**")
                    st.write(f"- Total Detections: {len(results['detections'])}")
                    st.write(f"- Unique Brands: {len(results['class_statistics'])}")
                    st.write(f"- Processing Time: {results.get('processing_time', 'N/A')}")
                    st.info("📊 View full results in the 'Results & Reports' tab")
                    
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                    logger.error(f"Error processing downloaded video: {e}", exc_info=True)
                
                finally:
                    st.session_state.processing_video = False

# ============================================================================
# TAB 3: RESULTS & REPORTS
# ============================================================================

with tab3:
    st.header("📊 Detection Results & Reports")
    
    # Display verified results only
    if st.session_state.get('verified_results'):
        results = st.session_state.verified_results
        
        st.subheader("📈 Video Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{results['duration_seconds']:.1f}s")
        with col2:
            st.metric("Total Detections", len(results['detections']))
        with col3:
            st.metric("Frames with Detections", results['detected_frames'])
        with col4:
            st.metric("Detection Rate", f"{(results['detected_frames'] / (results['total_frames'] // results['frame_skip'])) * 100:.1f}%")
        
        st.divider()
        
        if results['class_statistics']:
            st.subheader("🏷️ Brand Detection Details")
            
            # Create tabs for each brand
            brand_tabs = st.tabs([f"📦 {brand}" for brand in results['class_statistics'].keys()])
            
            for tab, (brand_name, stats) in zip(brand_tabs, results['class_statistics'].items()):
                with tab:
                    # Brand-specific metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Detections", stats['detections_count'])
                    with col2:
                        st.metric("Duration", f"{stats['total_time']:.2f}s")
                    with col3:
                        st.metric("Percentage", f"{stats['percentage']:.2f}%")
                    with col4:
                        st.metric("Avg Confidence", f"{stats['avg_confidence']:.2%}")
                    
                    st.write(f"**Max Confidence**: {stats['max_confidence']:.2%}")
                    st.write(f"**Frames Detected**: {stats['frames_detected']}")
                    
                    st.divider()
                    
                    # Display cropped images for THIS BRAND ONLY
                    if results.get('cropped_images'):
                        st.subheader(f"🖼️ {brand_name} Logo Samples")
                        
                        # Filter cropped images by brand name
                        brand_cropped_images = [
                            img_info for img_info in results['cropped_images']
                            if img_info.get('class', '').lower() == brand_name.lower()
                        ]
                        
                        if brand_cropped_images:
                            # Create columns for image grid
                            cols = st.columns(3)
                            
                            # Show first 9 images for this brand
                            for idx, img_info in enumerate(brand_cropped_images[:9]):
                                with cols[idx % 3]:
                                    try:
                                        img = cv2.imread(img_info['path'])
                                        if img is not None:
                                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                            st.image(
                                                img_rgb,
                                                caption=f"Frame {img_info.get('frame_number', 'N/A')} - {img_info['confidence']:.2%}",
                                                width="stretch"
                                            )
                                    except Exception as e:
                                        st.error(f"Could not load image: {e}")
                            
                            if len(brand_cropped_images) > 9:
                                st.info(f"... and {len(brand_cropped_images) - 9} more {brand_name} images")
                        else:
                            st.info(f"No cropped images available for {brand_name}")
            
            st.divider()
        
        st.divider()
        
        # Generate and display report
        st.subheader("📄 Full Report")
        
        report = st.session_state.video_processor.generate_report(results)
        
        st.text(report)
        
        # Download report
        st.download_button(
            label="⬇️ Download Report (TXT)",
            data=report,
            file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            width="stretch"
        )
        
        # Download results as JSON
        st.download_button(
            label="⬇️ Download Results (JSON)",
            data=json.dumps(results, indent=2, default=str),
            file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            width="stretch"
        )
    
    elif st.session_state.get('pending_results'):
        st.info("Detections are pending manual verification. Please review and confirm to view results.")
    else:
        st.info("No results yet. Upload and analyze a video to see results here.")

# ============================================================================
# TAB 4: DATABASE
# ============================================================================

with tab4:
    st.header("💾 Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Database Statistics")
        
        all_videos = st.session_state.database.get_all_videos()
        
        st.metric("Total Videos", len(all_videos))
        
        # Get total detections
        total_detections = 0
        for video in all_videos:
            detections = st.session_state.database.get_detections(video['id'])
            total_detections += len(detections)
        
        st.metric("Total Detections", total_detections)
        
        # Brand summary
        brand_summary = st.session_state.database.get_brand_summary()
        st.metric("Unique Brands Detected", len(brand_summary))
    
    with col2:
        st.subheader("🏷️ Brand Summary")
        
        if brand_summary:
            summary_df = []
            for brand_name, stats in brand_summary.items():
                summary_df.append({
                    'Brand': brand_name,
                    'Detections': stats['total_detections'],
                    'Videos': stats['videos_with_brand'],
                    'Avg Confidence': f"{stats['avg_confidence']:.2%}",
                    'Total Duration': f"{stats['total_duration_seconds']:.1f}s"
                })
            
            st.dataframe(summary_df, width="stretch")
    
    st.divider()
    
    st.subheader("📹 Processed Videos")
    
    if all_videos:
        for video in all_videos:
            video_name = video.get('nombre', 'Unknown')
            processed_at = video.get('fecha_procesado')
            duration = video.get('duracion_seg')

            with st.expander(f"📹 {video_name} - COMPLETED"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**Processed**: {processed_at}")
                with col2:
                    if duration is not None:
                        st.write(f"**Duration**: {float(duration):.1f}s")
                    else:
                        st.write("**Duration**: N/A")
                with col3:
                    st.write("**Resolution**: N/A")
                with col4:
                    st.write("**Status**: COMPLETED")
                
                detections = st.session_state.database.get_detections(video['id'])
                st.write(f"**Total Detections**: {len(detections)}")
                
                # Delete button
                if st.button(f"🗑️ Delete Video #{video['id']}", key=f"delete_{video['id']}"):
                    st.session_state.database.delete_video(video['id'])
                    st.success("Video deleted")
                    st.rerun()
    
    else:
        st.info("No videos in database yet.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem 0;">
    <p>🎯 Brand Logo Detection System | Powered by YOLO Object Detection</p>
    <p style="font-size: 0.8rem;">Computer Vision Project - Team 4</p>
</div>
""", unsafe_allow_html=True)
