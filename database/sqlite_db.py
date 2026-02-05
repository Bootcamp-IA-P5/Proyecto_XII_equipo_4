"""
Database Module - Stores detection results and metadata
Supports SQLite for local storage or can be extended for remote databases
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DetectionDatabase:
    """Manages database operations for detection results."""
    
    def __init__(self, db_path: str = "detections.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.cursor = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        try:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()
            
            # Create tables
            self.cursor.executescript("""
                -- Videos table
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    source_path TEXT,
                    source_url TEXT,
                    platform TEXT,
                    duration_seconds REAL,
                    fps REAL,
                    width INTEGER,
                    height INTEGER,
                    total_frames INTEGER,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    status TEXT DEFAULT 'pending'
                );
                
                -- Detections table
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    frame_number INTEGER,
                    timestamp_seconds REAL,
                    class_name TEXT NOT NULL,
                    confidence REAL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
                );
                
                -- Cropped images table
                CREATE TABLE IF NOT EXISTS cropped_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL,
                    video_id INTEGER NOT NULL,
                    image_path TEXT,
                    class_name TEXT,
                    confidence REAL,
                    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE,
                    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
                );
                
                -- Brand statistics table
                CREATE TABLE IF NOT EXISTS brand_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    brand_name TEXT NOT NULL,
                    total_detections INTEGER,
                    frames_detected INTEGER,
                    duration_seconds REAL,
                    percentage REAL,
                    avg_confidence REAL,
                    max_confidence REAL,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
                    UNIQUE(video_id, brand_name)
                );
                
                -- Create indexes
                CREATE INDEX IF NOT EXISTS idx_video_id ON detections(video_id);
                CREATE INDEX IF NOT EXISTS idx_class_name ON detections(class_name);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(detection_time);
                CREATE INDEX IF NOT EXISTS idx_video_status ON videos(status);
            """)
            
            self.connection.commit()
            logger.info(f"Database initialized: {self.db_path}")
        
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
    
    def add_video(
        self,
        filename: str,
        source_path: Optional[str] = None,
        source_url: Optional[str] = None,
        platform: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        fps: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        total_frames: Optional[int] = None
    ) -> int:
        """
        Add a video record to database.
        
        Args:
            filename: Video filename
            source_path: Local path if uploaded
            source_url: URL if from social media
            platform: Source platform (youtube, instagram, etc.)
            duration_seconds: Video duration
            fps: Frames per second
            width: Video width
            height: Video height
            total_frames: Total number of frames
            
        Returns:
            Video ID
        """
        try:
            self.cursor.execute("""
                INSERT INTO videos 
                (filename, source_path, source_url, platform, duration_seconds, 
                 fps, width, height, total_frames)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (filename, source_path, source_url, platform, duration_seconds, 
                  fps, width, height, total_frames))
            
            self.connection.commit()
            return self.cursor.lastrowid
        
        except Exception as e:
            logger.error(f"Error adding video: {e}")
            raise
    
    def add_detections(self, video_id: int, detections: List[Dict]) -> int:
        """
        Add multiple detections to database.
        
        Args:
            video_id: ID of the video
            detections: List of detection dictionaries
            
        Returns:
            Number of detections added
        """
        try:
            count = 0
            for detection in detections:
                bbox = detection.get('bbox', [0, 0, 0, 0])
                
                self.cursor.execute("""
                    INSERT INTO detections
                    (video_id, frame_number, timestamp_seconds, class_name, confidence,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_id,
                    detection.get('frame_number'),
                    detection.get('timestamp'),
                    detection.get('class'),
                    detection.get('confidence'),
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3])
                ))
                count += 1
            
            self.connection.commit()
            return count
        
        except Exception as e:
            logger.error(f"Error adding detections: {e}")
            raise
    
    def add_cropped_image(
        self,
        detection_id: int,
        video_id: int,
        image_path: str,
        class_name: str,
        confidence: float
    ) -> int:
        """
        Add a cropped image record.
        
        Args:
            detection_id: Detection ID
            video_id: Video ID
            image_path: Path to cropped image
            class_name: Brand/class name
            confidence: Detection confidence
            
        Returns:
            Cropped image record ID
        """
        try:
            self.cursor.execute("""
                INSERT INTO cropped_images
                (detection_id, video_id, image_path, class_name, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (detection_id, video_id, image_path, class_name, confidence))
            
            self.connection.commit()
            return self.cursor.lastrowid
        
        except Exception as e:
            logger.error(f"Error adding cropped image: {e}")
            raise
    
    def add_brand_statistics(
        self,
        video_id: int,
        brand_name: str,
        statistics: Dict
    ) -> int:
        """
        Add or update brand statistics for a video.
        
        Args:
            video_id: Video ID
            brand_name: Brand/class name
            statistics: Dictionary with statistics
            
        Returns:
            Statistics record ID
        """
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO brand_statistics
                (video_id, brand_name, total_detections, frames_detected, 
                 duration_seconds, percentage, avg_confidence, max_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video_id,
                brand_name,
                statistics.get('detections_count', 0),
                statistics.get('frames_detected', 0),
                statistics.get('total_time', 0),
                statistics.get('percentage', 0),
                statistics.get('avg_confidence', 0),
                statistics.get('max_confidence', 0)
            ))
            
            self.connection.commit()
            return self.cursor.lastrowid
        
        except Exception as e:
            logger.error(f"Error adding brand statistics: {e}")
            raise
    
    def get_video(self, video_id: int) -> Dict:
        """Get video information by ID."""
        try:
            self.cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
            row = self.cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting video: {e}")
            return None
    
    def get_video_by_filename(self, filename: str) -> Dict:
        """Get video information by filename."""
        try:
            self.cursor.execute("SELECT * FROM videos WHERE filename = ?", (filename,))
            row = self.cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting video: {e}")
            return None
    
    def get_detections(self, video_id: int) -> List[Dict]:
        """Get all detections for a video."""
        try:
            self.cursor.execute("""
                SELECT * FROM detections WHERE video_id = ? ORDER BY frame_number
            """, (video_id,))
            return [dict(row) for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting detections: {e}")
            return []
    
    def get_brand_statistics(self, video_id: int) -> Dict[str, Dict]:
        """Get brand statistics for a video."""
        try:
            self.cursor.execute("""
                SELECT * FROM brand_statistics WHERE video_id = ?
            """, (video_id,))
            
            stats = {}
            for row in self.cursor.fetchall():
                brand = dict(row)
                brand_name = brand.pop('brand_name')
                stats[brand_name] = brand
            
            return stats
        except Exception as e:
            logger.error(f"Error getting brand statistics: {e}")
            return {}
    
    def get_all_videos(self, status: Optional[str] = None) -> List[Dict]:
        """Get all videos, optionally filtered by status."""
        try:
            if status:
                self.cursor.execute("SELECT * FROM videos WHERE status = ? ORDER BY uploaded_at DESC", (status,))
            else:
                self.cursor.execute("SELECT * FROM videos ORDER BY uploaded_at DESC")
            
            return [dict(row) for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting videos: {e}")
            return []
    
    def update_video_status(self, video_id: int, status: str, processed_at: Optional[datetime] = None):
        """Update video processing status."""
        try:
            processed_at = processed_at or datetime.now()
            self.cursor.execute("""
                UPDATE videos SET status = ?, processed_at = ? WHERE id = ?
            """, (status, processed_at, video_id))
            self.connection.commit()
        except Exception as e:
            logger.error(f"Error updating video status: {e}")
    
    def get_brand_summary(self) -> Dict:
        """Get summary statistics for all brands."""
        try:
            self.cursor.execute("""
                SELECT brand_name, 
                       COUNT(*) as videos_with_brand,
                       SUM(total_detections) as total_detections,
                       AVG(avg_confidence) as avg_confidence,
                       SUM(duration_seconds) as total_duration_seconds
                FROM brand_statistics
                GROUP BY brand_name
                ORDER BY total_detections DESC
            """)
            
            summary = {}
            for row in self.cursor.fetchall():
                row_dict = dict(row)
                brand_name = row_dict.pop('brand_name')
                summary[brand_name] = row_dict
            
            return summary
        except Exception as e:
            logger.error(f"Error getting brand summary: {e}")
            return {}
    
    def export_video_report(self, video_id: int) -> Dict:
        """Export complete report for a video."""
        try:
            video = self.get_video(video_id)
            if not video:
                return None
            
            detections = self.get_detections(video_id)
            statistics = self.get_brand_statistics(video_id)
            
            return {
                'video': dict(video),
                'detections': detections,
                'statistics': statistics,
                'export_time': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return None
    
    def delete_video(self, video_id: int) -> bool:
        """Delete a video and all associated data."""
        try:
            self.cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting video: {e}")
            return False
