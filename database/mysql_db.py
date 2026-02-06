"""
MySQL Database Module for Brand Logo Detection.
Connects to existing MySQL schema and inserts videos/detections.
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging
import os

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class MySQLDatabase:
    """Manages MySQL database operations for detection results."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        load_dotenv()

        self.host = host or os.getenv("MYSQL_HOST", "localhost")
        self.port = int(port or os.getenv("MYSQL_PORT", "3306"))
        self.database = database or os.getenv("MYSQL_DATABASE", "brand_vision")
        self.user = user or os.getenv("MYSQL_USER", "root")
        self.password = password or os.getenv("MYSQL_PASSWORD", "")

        self.connection = None
        self._brand_cache: Dict[str, int] = {}
        self._connect()

    def _connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                autocommit=False,
                charset="utf8mb4",
                collation="utf8mb4_general_ci",
            )
            logger.info("Connected to MySQL database")
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def test_connection(self) -> bool:
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Error as e:
            logger.error(f"MySQL connection test failed: {e}")
            return False

    def _get_brand_id(self, brand_name: str) -> Optional[int]:
        if not brand_name:
            return None

        if brand_name in self._brand_cache:
            return self._brand_cache[brand_name]

        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT id FROM brands WHERE nombre = %s",
                (brand_name,)
            )
            row = cursor.fetchone()
            cursor.close()
            if row:
                brand_id = int(row["id"])
                self._brand_cache[brand_name] = brand_id
                return brand_id
            return None
        except Error as e:
            logger.error(f"Error fetching brand id for {brand_name}: {e}")
            return None

    def add_video(self, nombre: str, duracion_seg: Optional[float] = None) -> int:
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO videos (nombre, duracion_seg, fecha_procesado)
                VALUES (%s, %s, %s)
                """,
                (nombre, duracion_seg, datetime.now()),
            )
            video_id = cursor.lastrowid
            self.connection.commit()
            cursor.close()
            return video_id
        except Error as e:
            self.connection.rollback()
            logger.error(f"Error adding video: {e}")
            raise

    def add_detections(self, video_id: int, detections: List[Dict]) -> int:
        try:
            cursor = self.connection.cursor()
            rows = []
            skipped = 0

            for detection in detections:
                brand_name = detection.get("class")
                brand_id = self._get_brand_id(brand_name)
                if brand_id is None:
                    skipped += 1
                    continue

                bbox = detection.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                bbox_w = float(x2 - x1)
                bbox_h = float(y2 - y1)

                rows.append(
                    (
                        video_id,
                        brand_id,
                        float(detection.get("timestamp", 0)),
                        float(detection.get("confidence", 0)),
                        float(x1),
                        float(y1),
                        bbox_w,
                        bbox_h,
                    )
                )

            if rows:
                cursor.executemany(
                    """
                    INSERT INTO detections
                    (video_id, brand_id, segundo, confianza, bbox_x, bbox_y, bbox_w, bbox_h)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
                self.connection.commit()

            cursor.close()
            return len(rows)
        except Error as e:
            self.connection.rollback()
            logger.error(f"Error adding detections: {e}")
            raise

    def get_all_videos(self) -> List[Dict]:
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, nombre, duracion_seg, fecha_procesado FROM videos ORDER BY fecha_procesado DESC"
            )
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except Error as e:
            logger.error(f"Error getting videos: {e}")
            return []

    def get_detections(self, video_id: int) -> List[Dict]:
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT d.*, b.nombre AS brand_name
                FROM detections d
                LEFT JOIN brands b ON d.brand_id = b.id
                WHERE d.video_id = %s
                ORDER BY d.segundo
                """,
                (video_id,),
            )
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except Error as e:
            logger.error(f"Error getting detections: {e}")
            return []

    def get_brand_summary(self) -> Dict[str, Dict]:
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT b.nombre AS brand_name,
                       COUNT(*) AS total_detections,
                       COUNT(DISTINCT d.video_id) AS videos_with_brand,
                       AVG(d.confianza) AS avg_confidence,
                       COUNT(DISTINCT CONCAT(d.video_id, '-', d.segundo)) AS total_duration_seconds
                FROM detections d
                LEFT JOIN brands b ON d.brand_id = b.id
                GROUP BY b.nombre
                ORDER BY total_detections DESC
                """
            )
            rows = cursor.fetchall()
            cursor.close()
            summary = {}
            for row in rows:
                brand_name = row.pop("brand_name")
                summary[brand_name] = row
            return summary
        except Error as e:
            logger.error(f"Error getting brand summary: {e}")
            return {}

    def delete_video(self, video_id: int) -> bool:
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM detections WHERE video_id = %s", (video_id,))
            cursor.execute("DELETE FROM videos WHERE id = %s", (video_id,))
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            self.connection.rollback()
            logger.error(f"Error deleting video: {e}")
            return False
