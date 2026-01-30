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

# Import project modules
from src.pipeline import DetectionPipeline
from src.video_processor import VideoProcessor
from src.video_downloader import VideoDownloader
from src.db_mysql import MySQLDatabase
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    if st.button("View Database Stats", use_container_width=True):
        st.session_state.show_db_stats = not st.session_state.get('show_db_stats', False)
    
    if st.button("Export Database", use_container_width=True):
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
    
    col1, col2 = st.columns(2)
    
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
            if st.button("🔍 Analyze Video", use_container_width=True, type="primary"):
                st.session_state.processing_video = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
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
                    
                    # Save to database if requested
                    video_id = None
                    if save_to_db:
                        status_text.text("Saving to database...")
                        
                        video_id = st.session_state.database.add_video(
                            nombre=uploaded_file.name,
                            duracion_seg=results['duration_seconds']
                        )
                        
                        # Add detections
                        st.session_state.database.add_detections(
                            video_id,
                            results['detections']
                        )
                        
                        # Brand statistics are derived from detections in MySQL
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis complete!")
                    
                    # Display results
                    st.success("Video analysis completed successfully!")
                    
                    # Store results in session
                    st.session_state.last_results = results
                    st.session_state.last_video_id = video_id
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    logger.error(f"Error processing video: {e}")
                
                finally:
                    st.session_state.processing_video = False
                    # Clean up temporary file
                    if Path(video_path).exists():
                        os.remove(video_path)

    with col2:
        st.subheader("📹 Preview")
        st.info("Video preview will appear here")

# ============================================================================
# TAB 2: SOCIAL MEDIA LINKS
# ============================================================================

with tab2:
    st.header("Analyze Videos from Social Media")
    st.markdown("""
    Download videos from YouTube, Instagram, TikTok, Facebook, and Twitter for analysis.
    """)
    
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
        if st.button("⬇️ Download Video", use_container_width=True, type="primary"):
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
        
        analyze_after_download = st.checkbox(
            "Analyze after download",
            value=True,
            help="Automatically analyze the video after download"
        )
        
        if analyze_after_download and st.session_state.get('downloaded_video_path'):
            if st.button("🔍 Analyze Downloaded Video", use_container_width=True, type="primary"):
                st.info("Analyzing downloaded video...")

# ============================================================================
# TAB 3: RESULTS & REPORTS
# ============================================================================

with tab3:
    st.header("📊 Detection Results & Reports")
    
    # Display last results if available
    if st.session_state.get('last_results'):
        results = st.session_state.last_results
        
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
            
            # Display cropped images if available
            if results.get('cropped_images'):
                st.subheader("🖼️ Detected Logo Samples")
                
                cropped_images = results['cropped_images']
                
                # Create columns for image grid
                cols = st.columns(3)
                
                for idx, img_info in enumerate(cropped_images[:9]):  # Show first 9
                    with cols[idx % 3]:
                        try:
                            img = cv2.imread(img_info['path'])
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                st.image(
                                    img_rgb,
                                    caption=f"{img_info['class']} ({img_info['confidence']:.2%})",
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.error(f"Could not load image: {e}")
                
                if len(cropped_images) > 9:
                    st.info(f"... and {len(cropped_images) - 9} more cropped images")
        
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
            use_container_width=True
        )
        
        # Download results as JSON
        st.download_button(
            label="⬇️ Download Results (JSON)",
            data=json.dumps(results, indent=2, default=str),
            file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
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
            
            st.dataframe(summary_df, use_container_width=True)
    
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
