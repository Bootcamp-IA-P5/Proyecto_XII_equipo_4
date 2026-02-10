"""
Streamlit Application for Brand Logo Detection
Interactive web interface for video upload, analysis, and reporting

3-PHASE PIPELINE:
  Phase 1 - Analysis (NO database writes for detections)
  Phase 2 - Review UI (gallery with checkboxes)
  Phase 3 - Save to Database (ONLY place detections are inserted)
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
import time
from datetime import datetime
import json
import logging
import sys
import re
import threading
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
from back.services.pipeline import DetectionPipeline
from back.services.video_processor import VideoProcessor
from back.services.video_downloader import VideoDownloader
from back.services.visualization import annotate_image
from back.services import config
from database.mysql_db import MySQLDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _camera_worker(camera_idx, pipeline, confidence_threshold, stop_event, shared):
    """Background thread: capture frames, run detection, update shared dict."""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        shared["error"] = "Cannot open webcam. Check the camera index."
        shared["finished"] = True
        return

    frame_count = 0
    start_time = time.time()
    all_detections: List[Dict] = []
    frame_detections: Dict[int, List[Dict]] = {}
    brands: Dict[str, Dict] = {}

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1
            frame_time = frame_count / 30.0
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            detections = pipeline.detect_objects(frame, confidence_threshold)

            if detections:
                vis_dets = [
                    {"box": d["bbox"], "label": d["class"], "confidence": d["confidence"]}
                    for d in detections
                ]
                display_frame = annotate_image(frame, vis_dets)

                frame_det_list = []
                for d in detections:
                    det_info = {
                        "frame_number": frame_count,
                        "timestamp": frame_time,
                        "class": d["class"],
                        "confidence": float(d["confidence"]),
                        "bbox": d["bbox"],
                    }
                    all_detections.append(det_info)
                    frame_det_list.append(det_info)
                frame_detections[frame_count] = frame_det_list

                for d in detections:
                    brand = d["class"]
                    if brand not in brands:
                        brands[brand] = {"count": 0, "max_conf": 0.0, "confs": []}
                    brands[brand]["count"] += 1
                    brands[brand]["max_conf"] = max(brands[brand]["max_conf"], d["confidence"])
                    brands[brand]["confs"].append(d["confidence"])
            else:
                display_frame = frame

            # Convert frame for Streamlit (BGR → RGB)
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Update shared state (main thread reads this)
            shared["frame"] = rgb_frame
            shared["frame_count"] = frame_count
            shared["fps"] = fps
            shared["elapsed"] = elapsed
            shared["total_detections"] = len(all_detections)
            shared["brands"] = brands
            shared["all_detections"] = all_detections
            shared["frame_detections"] = frame_detections

    except Exception as e:
        shared["error"] = str(e)
        logger.error(f"Camera worker error: {e}", exc_info=True)
    finally:
        cap.release()
        shared["finished"] = True




# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _safe_imread(path: str) -> "np.ndarray | None":
    """Read an image safely on Windows, even if the path contains non-ASCII chars.

    Uses np.fromfile + cv2.imdecode instead of cv2.imread to avoid
    ANSI code-page issues on Windows and prevent 'utf-8 codec can't decode' errors.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def _build_crop_index(cropped_images: Optional[List[Dict]]) -> Dict[Tuple[int, str], List[Dict]]:
    """Build lookup index: (frame_number, class_name) -> list of crop dicts."""
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
    """Convert raw results into a flat list of detection dicts with crop paths.

    Each returned dict has the schema:
        {
            "brand": str,
            "confidence": float,
            "crop_path": str | None,
            "frame_time": float,
            "verified": True,
            ... plus original keys for DB insert later
        }
    """
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
            "brand": class_name or "Unknown",
            "confidence": float(detection.get("confidence", 0)),
            "crop_path": crop_path,
            "frame_time": float(detection.get("timestamp", 0)),
            "verified": True,
            "review_id": idx,
        })

    return review_items


def _build_frame_detections(detections: List[Dict]) -> Dict[int, List[Dict]]:
    """Group detections by frame number."""
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
    frame_skip: int,
) -> Dict[str, Dict]:
    """Compute per-class summary statistics from a list of detections."""
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
                "percentage": 0,
                "confidences": [],
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
    """Rebuild a full results dict using only the verified detections."""
    STRIP_KEYS = {"crop_path", "review_id", "verified", "brand", "frame_time"}
    cleaned_detections = [
        {k: v for k, v in det.items() if k not in STRIP_KEYS}
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
        cleaned_detections, frame_detections, total_frames, fps, frame_skip,
    )

    if verified_results.get("cropped_images"):
        verified_results["cropped_images"] = [
            crop for crop in verified_results["cropped_images"]
            if any(
                crop.get("frame_number") == det.get("frame_number")
                and crop.get("class") == det.get("class")
                for det in cleaned_detections
            )
        ]

    return verified_results


# ============================================================================
# PHASE 3 -- save_verified_detections()
# THIS IS THE **ONLY** PLACE IN THE ENTIRE PROJECT WHERE
# db.add_detections() IS CALLED.
# ============================================================================

def save_verified_detections() -> int:
    """Insert ONLY verified detections into the database.

    Called exclusively when the user clicks 'Confirm & Save to Database'.
    Returns the number of rows inserted.
    """
    if not st.session_state.get("pending_save_to_db") or not st.session_state.get("pending_video_id"):
        st.warning("Video metadata was not registered. Check 'Save to Database' and re-run analysis.")
        return 0

    detections = st.session_state.get("pending_detections", [])
    verified = [det for det in detections if det.get("verified", True)]

    if not verified:
        st.info("No verified detections to save.")
        return 0

    STRIP_KEYS = {"crop_path", "review_id", "verified", "brand", "frame_time"}
    db_rows = [
        {k: v for k, v in det.items() if k not in STRIP_KEYS}
        for det in verified
    ]

    inserted = st.session_state.database.add_detections(
        st.session_state.pending_video_id,
        db_rows,
    )
    return inserted


# ============================================================================
# PHASE 2 -- Review UI
# ============================================================================

def _render_review_ui(prefix: str = "review") -> None:
    """Render review gallery grouped by brand with Select All and individual checkboxes.

    This function ONLY renders the UI.  The database is NOT touched here.
    Database save only happens inside save_verified_detections().
    """
    if not st.session_state.get("pending_results") or not st.session_state.get("pending_detections"):
        return

    st.subheader("Review Detections")
    st.write("Uncheck false positives, then click **Confirm & Save** to store only verified detections.")

    detections = st.session_state.pending_detections

    # -- Group detections by brand --
    brands: Dict[str, List[Dict]] = {}
    for det in detections:
        brand = det.get("brand", "Unknown")
        brands.setdefault(brand, []).append(det)

    # -- Initialise per-detection session-state keys (only on first render) --
    for det in detections:
        rid = det.get("review_id", 0)
        brand = det.get("brand", "Unknown")
        ind_key = f"{prefix}_sel_{brand}_{rid}"
        if ind_key not in st.session_state:
            st.session_state[ind_key] = det.get("verified", True)

    # -- Callback factory: "Select All" propagates to every detection in the brand --
    def _make_select_all_cb(brand_name: str, brand_dets: List[Dict]):
        def _cb():
            new_val = st.session_state[f"{prefix}_selall_{brand_name}"]
            for d in brand_dets:
                st.session_state[f"{prefix}_sel_{brand_name}_{d.get('review_id', 0)}"] = new_val
        return _cb

    COLS_PER_ROW = 3

    for brand_name in sorted(brands.keys()):
        brand_dets = brands[brand_name]
        count = len(brand_dets)

        st.markdown(f"### Brand: {brand_name} ({count} detection{'s' if count != 1 else ''})")

        # Compute whether every detection in this brand is currently selected
        all_selected = all(
            st.session_state.get(f"{prefix}_sel_{brand_name}_{d.get('review_id', 0)}", True)
            for d in brand_dets
        )
        # Sync the Select-All key so the checkbox reflects individual changes
        sa_key = f"{prefix}_selall_{brand_name}"
        st.session_state[sa_key] = all_selected

        st.checkbox(
            f"Select All ({count})",
            key=sa_key,
            on_change=_make_select_all_cb(brand_name, brand_dets),
        )

        # -- Detection grid --
        for row_start in range(0, count, COLS_PER_ROW):
            row_dets = brand_dets[row_start:row_start + COLS_PER_ROW]
            cols = st.columns(COLS_PER_ROW)
            for col_idx, det in enumerate(row_dets):
                rid = det.get("review_id", 0)
                ind_key = f"{prefix}_sel_{brand_name}_{rid}"

                with cols[col_idx]:
                    crop_path = det.get("crop_path")
                    if crop_path and Path(crop_path).exists():
                        img = _safe_imread(crop_path)
                        if img is not None:
                            st.image(
                                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                caption=f"{brand_name} – {det.get('confidence', 0):.1%}",
                            )
                    else:
                        st.write("No crop available")

                    st.checkbox("Keep detection", key=ind_key)

                    st.caption(
                        f"Frame {det.get('frame_number', 'N/A')} | "
                        f"{brand_name} | "
                        f"{det.get('confidence', 0):.1%}"
                    )

        st.divider()

    # -- Confirm & Save button (outside any form so callbacks work) --
    if st.button("Confirm & Save to Database", type="primary", key=f"{prefix}_confirm_btn"):
        # Sync session-state selections back to detection dicts
        for det in detections:
            rid = det.get("review_id", 0)
            brand = det.get("brand", "Unknown")
            det["verified"] = st.session_state.get(f"{prefix}_sel_{brand}_{rid}", True)

        verified = [d for d in detections if d.get("verified", True)]
        verified_results = _build_verified_results(st.session_state.pending_results, verified)
        st.session_state.verified_results = verified_results
        st.session_state.last_results = verified_results

        inserted = save_verified_detections()
        if st.session_state.get("pending_save_to_db") and st.session_state.get("pending_video_id"):
            st.success(f"Saved {inserted} verified detections to the database.")
        else:
            st.info("Verified detections stored in session (database save was disabled).")

        st.session_state.pending_results = None
        st.session_state.pending_detections = []
        st.session_state.pending_video_id = None
        st.session_state.pending_save_to_db = False
        st.session_state.analysis_complete = False
        st.session_state.video_processed = False
        st.rerun()


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Brand Logo Detection",
    page_icon="���",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
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

if "pipeline" not in st.session_state:
    st.session_state.pipeline = DetectionPipeline()

if "video_processor" not in st.session_state:
    st.session_state.video_processor = VideoProcessor(st.session_state.pipeline)

if "video_downloader" not in st.session_state:
    st.session_state.video_downloader = VideoDownloader()

if "database" not in st.session_state:
    st.session_state.database = MySQLDatabase()

if "processing_video" not in st.session_state:
    st.session_state.processing_video = False

if "pending_results" not in st.session_state:
    st.session_state.pending_results = None

if "pending_detections" not in st.session_state:
    st.session_state.pending_detections = []

if "pending_video_id" not in st.session_state:
    st.session_state.pending_video_id = None

if "pending_save_to_db" not in st.session_state:
    st.session_state.pending_save_to_db = False

if "verified_results" not in st.session_state:
    st.session_state.verified_results = None

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "live_running" not in st.session_state:
    st.session_state.live_running = False

if "live_detections" not in st.session_state:
    st.session_state.live_detections = []

if "live_stats" not in st.session_state:
    st.session_state.live_stats = {"frames": 0, "detections": 0, "brands": {}}

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("Configuration")

    st.subheader("Model Settings")
    confidence = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Minimum confidence score for detections",
    )

    frame_skip = st.slider(
        "Frame Skip",
        min_value=1, max_value=30, value=5, step=1,
        help="Process every Nth frame for faster analysis",
    )

    st.divider()

    st.subheader("Database")
    if st.session_state.database.test_connection():
        st.success("MySQL connected")
    else:
        st.error("MySQL connection failed")
    if st.button("View Database Stats"):
        st.session_state.show_db_stats = not st.session_state.get("show_db_stats", False)

    if st.button("Export Database"):
        st.info("Database export feature coming soon!")

# ============================================================================
# MAIN HEADER
# ============================================================================

st.title("Brand Logo Detection System")
st.markdown(
    "Detect and track brand logos in videos from YouTube, Instagram, TikTok, Facebook, and local uploads. "
    "Analyze screen time and generate detailed reports for marketing insights."
)

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs([
    "📁 Upload & Analyze",
    "📊 Results & Reports",
    "🗄️ Database",
])

# ============================================================================
# TAB 1 -- UPLOAD & ANALYZE
# ============================================================================

with tab1:
    st.header("Upload & Analyze Videos")

    # Reset stale analysis state if detections are empty
    if st.session_state.analysis_complete and not st.session_state.pending_detections:
        st.session_state.analysis_complete = False

    # Phase 2: Show review UI at the TOP when analysis is complete
    if st.session_state.analysis_complete and st.session_state.pending_detections:
        with st.container():
            _render_review_ui(prefix="upload_review")
            st.divider()

    # ----- Input source selector -----
    input_source = st.radio(
        "Video Source",
        ["Upload Local Video", "Social Media Link", "Webcam"],
        horizontal=True,
        help="Choose how to provide the video for analysis",
    )

    col1, col2 = st.columns(2)

    with col2:
        st.subheader("Preview")
        preview_placeholder = st.empty()
        preview_placeholder.info("Video preview will appear here during analysis")

        # Show supported platforms info when Social Media is selected
        if input_source == "Social Media Link":
            st.divider()
            st.subheader("Supported Platforms")
            platforms_info = {
                "YouTube": "Full support - Public and private videos",
                "Instagram": "Posts and Reels - Requires credentials",
                "TikTok": "Full support - TikTok videos",
                "Facebook": "Full support - Facebook videos",
                "Twitter/X": "Full support - Twitter videos",
            }
            for plat, info in platforms_info.items():
                st.write(f"**{plat}**: {info}")

        # Live stats panel for Webcam
        if input_source == "Webcam":
            st.divider()
            st.subheader("Live Statistics")
            live_stats_placeholder = st.empty()
            st.subheader("Detected Brands")
            live_brands_placeholder = st.empty()

    with col1:
        # =================================================================
        # OPTION A: Upload Local Video
        # =================================================================
        if input_source == "Upload Local Video":
            st.subheader("Upload Local Video")

            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=["mp4", "avi", "mov", "mkv", "flv", "wmv", "webm"],
            )

            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name

                st.success(f"Video uploaded: {uploaded_file.name}")
                st.session_state._current_video_path = video_path
                st.session_state._current_video_name = uploaded_file.name
                st.session_state._delete_after_analysis = True

                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Duration", f"{duration:.1f}s")
                    with c2:
                        st.metric("FPS", f"{fps:.1f}")
                    with c3:
                        st.metric("Resolution", f"{width}x{height}")
                    with c4:
                        st.metric("Frames", total_frames)

                st.subheader("Analysis Options")
                co1, co2 = st.columns(2)
                with co1:
                    extract_crops = st.checkbox("Extract Cropped Images", value=True,
                                                help="Save cropped regions of detected logos")
                with co2:
                    save_to_db = st.checkbox("Save to Database", value=True,
                                             help="Store detection results in database")

                st.session_state._extract_crops = extract_crops
                st.session_state._save_to_db = save_to_db
                st.session_state._ready_to_analyze = True

                if st.button("Analyze Video", type="primary"):
                    st.session_state._run_analysis = True

        # =================================================================
        # OPTION B: Social Media Link
        # =================================================================
        elif input_source == "Social Media Link":
            st.subheader("Download from Social Media")

            video_url = st.text_input(
                "Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Paste the link to a video from YouTube, Instagram, TikTok, Facebook, or Twitter",
            )

            custom_name = st.text_input(
                "Custom Video Name (Optional)",
                placeholder="my_brand_video",
                help="Custom name for the downloaded video",
            )

            if video_url:
                platform = st.session_state.video_downloader.detect_platform(video_url)
                if platform:
                    st.success(f"Detected platform: **{platform.upper()}**")
                    if platform == "instagram":
                        st.warning("Instagram requires login credentials")
                        with st.expander("Instagram Credentials"):
                            ig_username = st.text_input("Instagram Username", type="default")
                            ig_password = st.text_input("Instagram Password", type="password")
                else:
                    st.error("Platform not detected or not supported")

            if st.button("Download Video", type="primary"):
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
                            if platform == "instagram":
                                credentials = {
                                    "username": st.session_state.get("ig_username"),
                                    "password": st.session_state.get("ig_password"),
                                }
                            result = st.session_state.video_downloader.download(
                                video_url, video_name=custom_name, credentials=credentials,
                            )
                            progress_bar.progress(75)
                            if result["success"]:
                                status_text.text("Download complete!")
                                progress_bar.progress(100)
                                st.success("Video downloaded successfully!")
                                st.json(result)
                                st.session_state.downloaded_video_path = result["path"]
                                st.session_state.downloaded_video_platform = result.get("platform")
                            else:
                                st.error(f"Download failed: {result.get('error')}")
                    except Exception as e:
                        st.error(f"Error downloading video: {str(e)}")
                        logger.error(f"Error downloading video: {e}")

            # ----- Analyze downloaded video -----
            if st.session_state.get("downloaded_video_path"):
                st.divider()
                st.success("Video ready for analysis")

                video_path = st.session_state.downloaded_video_path
                st.session_state._current_video_path = video_path
                st.session_state._current_video_name = Path(video_path).name
                st.session_state._delete_after_analysis = False

                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()

                    st.write("**Video Information:**")
                    ci1, ci2, ci3 = st.columns(3)
                    with ci1:
                        st.metric("Duration", f"{duration:.1f}s")
                    with ci2:
                        st.metric("Resolution", f"{width}x{height}")
                    with ci3:
                        st.metric("FPS", f"{fps:.1f}")

                extract_crops_sm = st.checkbox(
                    "Extract Cropped Images", value=True, key="extract_crops_sm",
                    help="Save cropped regions of detected logos",
                )
                save_to_db_sm = st.checkbox(
                    "Save to Database", value=True, key="save_to_db_sm",
                    help="Store detection results in database",
                )

                st.session_state._extract_crops = extract_crops_sm
                st.session_state._save_to_db = save_to_db_sm
                st.session_state._ready_to_analyze = True

                if st.button("Analyze Downloaded Video", type="primary"):
                    st.session_state._run_analysis = True

        # =================================================================
        # OPTION C: Webcam
        # =================================================================
        elif input_source == "Webcam":
            st.subheader("📷 Webcam Detection")
            st.markdown("Detect brand logos in real-time using your webcam.")

            camera_index = st.number_input(
                "Camera Index",
                min_value=0, max_value=10, value=0, step=1,
                help="Select camera device index (0 = default webcam)",
            )

            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                start_clicked = st.button(
                    "▶️ Start Detection", type="primary", key="webcam_start",
                    disabled=st.session_state.live_running,
                )
            with btn_c2:
                stop_clicked = st.button(
                    "⏹️ Stop Detection", key="webcam_stop",
                    disabled=not st.session_state.live_running,
                )

            # --- Handle Start ---
            if start_clicked and not st.session_state.live_running:
                # Load model first
                pipeline = st.session_state.pipeline
                if pipeline.model is None:
                    preview_placeholder.info("Loading detection model...")
                    if not pipeline.load_model():
                        st.error("Failed to load detection model.")
                        st.stop()

                stop_event = threading.Event()
                shared = {
                    "frame": None, "frame_count": 0, "fps": 0.0,
                    "elapsed": 0.0, "total_detections": 0, "brands": {},
                    "all_detections": [], "frame_detections": {},
                    "finished": False, "error": None,
                }
                thread = threading.Thread(
                    target=_camera_worker,
                    args=(int(camera_index), pipeline, confidence, stop_event, shared),
                    daemon=True,
                )
                thread.start()

                st.session_state.live_running = True
                st.session_state._cam_stop_event = stop_event
                st.session_state._cam_shared = shared
                st.session_state._cam_thread = thread
                st.rerun()

            # --- Handle Stop ---
            if stop_clicked and st.session_state.live_running:
                stop_event = st.session_state.get("_cam_stop_event")
                if stop_event:
                    stop_event.set()
                thread = st.session_state.get("_cam_thread")
                if thread:
                    thread.join(timeout=3)
                st.session_state.live_running = False
                st.session_state._cam_finalizing = True
                st.rerun()

    # =====================================================================
    # WEBCAM: Finalize results after stopping
    # =====================================================================
    if st.session_state.get("_cam_finalizing"):
        st.session_state._cam_finalizing = False
        shared = st.session_state.get("_cam_shared", {})
        all_dets = shared.get("all_detections", [])
        frame_dets = shared.get("frame_detections", {})
        frame_count = shared.get("frame_count", 0)
        elapsed = shared.get("elapsed", 0)
        fps_display = shared.get("fps", 0)

        if all_dets:
            class_statistics: Dict[str, Dict] = {}
            for det in all_dets:
                cn = det["class"]
                if cn not in class_statistics:
                    class_statistics[cn] = {
                        "detections_count": 0, "frames_detected": 0,
                        "total_time": 0, "avg_confidence": 0,
                        "max_confidence": 0, "percentage": 0, "_confs": [],
                    }
                cs = class_statistics[cn]
                cs["detections_count"] += 1
                cs["_confs"].append(det["confidence"])
                cs["max_confidence"] = max(cs["max_confidence"], det["confidence"])

            for cn, cs in class_statistics.items():
                cs["avg_confidence"] = float(np.mean(cs["_confs"]))
                cs["frames_detected"] = len([
                    f for f, dlist in frame_dets.items()
                    if any(d["class"] == cn for d in dlist)
                ])
                cs["total_time"] = cs["frames_detected"] / 30.0
                cs["percentage"] = (
                    cs["frames_detected"] / frame_count * 100
                ) if frame_count > 0 else 0
                del cs["_confs"]

            live_results = {
                "video_path": "live_webcam",
                "video_name": "Live Webcam Session",
                "duration_seconds": elapsed,
                "total_frames": frame_count,
                "fps": fps_display,
                "width": 0, "height": 0,
                "detections": all_dets,
                "frame_detections": frame_dets,
                "class_statistics": class_statistics,
                "detected_frames": len(frame_dets),
                "frame_skip": 1,
                "processing_timestamp": datetime.now().isoformat(),
            }

            st.session_state.pending_results = live_results
            st.session_state.pending_detections = _attach_crops_to_detections(live_results)
            st.session_state.pending_video_id = None
            st.session_state.pending_save_to_db = False
            st.session_state.analysis_complete = True
            st.session_state.last_results = live_results

        # Clean up thread references
        for k in ("_cam_stop_event", "_cam_shared", "_cam_thread"):
            st.session_state.pop(k, None)

        st.rerun()

    # =====================================================================
    # WEBCAM: Display live feed while thread is running
    # =====================================================================
    if st.session_state.live_running:
        shared = st.session_state.get("_cam_shared", {})

        if shared.get("error"):
            st.error(shared["error"])
            st.session_state.live_running = False
            stop_event = st.session_state.get("_cam_stop_event")
            if stop_event:
                stop_event.set()
        elif shared.get("finished"):
            # Thread ended unexpectedly
            st.session_state.live_running = False
            st.session_state._cam_finalizing = True
            st.rerun()
        else:
            # Show latest frame
            frame = shared.get("frame")
            if frame is not None:
                preview_placeholder.image(
                    frame,
                    caption=f"Frame {shared.get('frame_count', 0)} | {shared.get('fps', 0):.1f} FPS",
                    width="stretch",
                )

            # Show stats
            fc = shared.get("frame_count", 0)
            td = shared.get("total_detections", 0)
            fps_d = shared.get("fps", 0)
            el = shared.get("elapsed", 0)
            live_stats_placeholder.markdown(
                f"**Frames:** {fc}  \n"
                f"**Detections:** {td}  \n"
                f"**FPS:** {fps_d:.1f}  \n"
                f"**Elapsed:** {el:.0f}s"
            )

            brands = shared.get("brands", {})
            if brands:
                brand_lines = []
                for bname, bstats in sorted(
                    brands.items(), key=lambda x: x[1]["count"], reverse=True,
                ):
                    avg_c = (
                        sum(bstats["confs"]) / len(bstats["confs"])
                        if bstats["confs"] else 0
                    )
                    brand_lines.append(
                        f"- **{bname}**: {bstats['count']} det. "
                        f"(avg {avg_c:.1%}, max {bstats['max_conf']:.1%})"
                    )
                live_brands_placeholder.markdown("\n".join(brand_lines))
            else:
                live_brands_placeholder.info("No brands detected yet...")

            # Auto-refresh every 0.3s to update display
            time.sleep(0.3)
            st.rerun()

    # =====================================================================
    # SHARED ANALYSIS PIPELINE (used by Upload & Social Media)
    # =====================================================================
    if st.session_state.get("_run_analysis") and st.session_state.get("_current_video_path"):
        video_path = st.session_state._current_video_path
        video_name = st.session_state._current_video_name
        extract_crops = st.session_state._extract_crops
        save_to_db = st.session_state._save_to_db
        delete_after = st.session_state.get("_delete_after_analysis", False)

        st.session_state._run_analysis = False
        st.session_state.processing_video = True
        st.session_state.analysis_complete = False
        st.session_state.pending_detections = []
        st.session_state.pending_results = None

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(progress, message, annotated_frame=None):
            progress_bar.progress(progress)
            status_text.text(message)
            # Live preview: show the annotated frame with bounding boxes
            match = re.search(r"Processed (\d+)/(\d+) frames", message)
            if match and annotated_frame is not None:
                frame_number = int(match.group(1))
                preview_placeholder.image(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                    caption=f"Processing frame {frame_number}",
                )

        try:
            status_text.text("Processing video...")

            # PHASE 1: ANALYSIS -- NO DATABASE WRITES FOR DETECTIONS
            results = st.session_state.video_processor.process_video(
                video_path,
                confidence_threshold=confidence,
                frame_skip=frame_skip,
                progress_callback=progress_callback,
            )

            if extract_crops:
                status_text.text("Extracting crops...")
                results = st.session_state.video_processor.extract_cropped_detections(
                    video_path, results,
                )

            # Save video METADATA only -- NO detections written here
            video_id = None
            if save_to_db:
                status_text.text("Saving video metadata...")
                video_id = st.session_state.database.add_video(
                    nombre=video_name,
                    duracion_seg=results["duration_seconds"],
                )

            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            st.success("Video analysis completed successfully!")

            # Stage ALL detections in session state (Phase 1 output)
            st.session_state.pending_results = results
            st.session_state.pending_detections = _attach_crops_to_detections(results)
            st.session_state.pending_video_id = video_id
            st.session_state.pending_save_to_db = save_to_db
            st.session_state.analysis_complete = True

            st.session_state.last_results = results
            st.session_state.last_video_id = video_id

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Error processing video: {e}", exc_info=True)

        finally:
            st.session_state.processing_video = False
            if delete_after and Path(video_path).exists():
                os.remove(video_path)

        # Clean up temporary state
        st.session_state._current_video_path = None
        st.session_state._current_video_name = None
        st.session_state._ready_to_analyze = False

        # Force rerun so the review gallery renders immediately
        if st.session_state.analysis_complete:
            st.rerun()


# ============================================================================
# TAB 2 -- RESULTS & REPORTS
# ============================================================================

with tab2:
    st.header("Detection Results & Reports")

    if st.session_state.get("verified_results"):
        results = st.session_state.verified_results

        st.subheader("Video Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Duration", f"{results['duration_seconds']:.1f}s")
        with c2:
            st.metric("Total Detections", len(results["detections"]))
        with c3:
            st.metric("Frames with Detections", results["detected_frames"])
        with c4:
            denom = results["total_frames"] // results["frame_skip"] if results.get("frame_skip") else 0
            rate = (results["detected_frames"] / denom * 100) if denom else 0
            st.metric("Detection Rate", f"{rate:.1f}%")

        st.divider()

        if results["class_statistics"]:
            st.subheader("Brand Detection Details")
            brand_tabs = st.tabs([f"{b}" for b in results["class_statistics"]])

            for btab, (brand_name, stats) in zip(brand_tabs, results["class_statistics"].items()):
                with btab:
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    with bc1:
                        st.metric("Detections", stats["detections_count"])
                    with bc2:
                        st.metric("Duration", f"{stats['total_time']:.2f}s")
                    with bc3:
                        st.metric("Percentage", f"{stats['percentage']:.2f}%")
                    with bc4:
                        st.metric("Avg Confidence", f"{stats['avg_confidence']:.2%}")

                    st.write(f"**Max Confidence**: {stats['max_confidence']:.2%}")
                    st.write(f"**Frames Detected**: {stats['frames_detected']}")
                    st.divider()

                    if results.get("cropped_images"):
                        st.subheader(f"{brand_name} Logo Samples")
                        brand_crops = [
                            ci for ci in results["cropped_images"]
                            if ci.get("class", "").strip().lower() == brand_name.strip().lower()
                        ]
                        if brand_crops:
                            cols = st.columns(3)
                            for idx, ci in enumerate(brand_crops[:9]):
                                with cols[idx % 3]:
                                    try:
                                        img = _safe_imread(ci["path"])
                                        if img is not None:
                                            st.image(
                                                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                                caption=f"Frame {ci.get('frame_number', 'N/A')} - {ci['confidence']:.2%}",
                                            )
                                    except Exception as e:
                                        st.error(f"Could not load image: {e}")
                            if len(brand_crops) > 9:
                                st.info(f"... and {len(brand_crops) - 9} more {brand_name} images")
                        else:
                            st.info(f"No cropped images available for {brand_name}")

            st.divider()

        st.divider()
        st.subheader("Full Report")
        report = st.session_state.video_processor.generate_report(results)
        st.text(report)

        st.download_button(
            label="Download Report (TXT)",
            data=report,
            file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )
        st.download_button(
            label="Download Results (JSON)",
            data=json.dumps(results, indent=2, default=str),
            file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    elif st.session_state.get("pending_results"):
        st.info("Detections are pending manual verification. Please review and confirm to view results.")
    else:
        st.info("No results yet. Upload and analyze a video to see results here.")


# ============================================================================
# TAB 3 -- DATABASE
# ============================================================================

with tab3:
    st.header("Database Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Database Statistics")
        all_videos = st.session_state.database.get_all_videos()
        st.metric("Total Videos", len(all_videos))

        total_detections = 0
        for video in all_videos:
            detections = st.session_state.database.get_detections(video["id"])
            total_detections += len(detections)
        st.metric("Total Detections", total_detections)

        brand_summary = st.session_state.database.get_brand_summary()
        st.metric("Unique Brands Detected", len(brand_summary))

    with col2:
        st.subheader("Brand Summary")
        if brand_summary:
            summary_df = []
            for brand_name, stats in brand_summary.items():
                summary_df.append({
                    "Brand": brand_name,
                    "Detections": stats["total_detections"],
                    "Videos": stats["videos_with_brand"],
                    "Avg Confidence": f"{stats['avg_confidence']:.2%}",
                    "Total Duration": f"{stats['total_duration_seconds']:.1f}s",
                })
            st.dataframe(summary_df)

    st.divider()
    st.subheader("Processed Videos")

    if all_videos:
        for video in all_videos:
            video_name = video.get("nombre", "Unknown")
            processed_at = video.get("fecha_procesado")
            duration = video.get("duracion_seg")

            with st.expander(f"{video_name} - COMPLETED"):
                vc1, vc2, vc3, vc4 = st.columns(4)
                with vc1:
                    st.write(f"**Processed**: {processed_at}")
                with vc2:
                    st.write(f"**Duration**: {float(duration):.1f}s" if duration else "**Duration**: N/A")
                with vc3:
                    st.write("**Resolution**: N/A")
                with vc4:
                    st.write("**Status**: COMPLETED")

                detections = st.session_state.database.get_detections(video["id"])
                st.write(f"**Total Detections**: {len(detections)}")

                if st.button(f"Delete Video #{video['id']}", key=f"delete_{video['id']}"):
                    st.session_state.database.delete_video(video["id"])
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
    <p>Brand Logo Detection System | Powered by YOLO Object Detection</p>
    <p style="font-size: 0.8rem;">Computer Vision Project - Team 4</p>
</div>
""", unsafe_allow_html=True)
