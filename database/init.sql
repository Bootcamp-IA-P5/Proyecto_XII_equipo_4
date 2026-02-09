-- Database initialization script for Logo Detection
-- This script runs automatically when MySQL container starts

-- Create the database if not exists
CREATE DATABASE IF NOT EXISTS logo_detection;
USE logo_detection;

-- Table for video analysis sessions
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) UNIQUE NOT NULL,
    video_name VARCHAR(255),
    video_source VARCHAR(50),
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,
    status ENUM('processing', 'completed', 'failed') DEFAULT 'processing',
    total_frames INT DEFAULT 0,
    processed_frames INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Table for detected logos
CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    frame_number INT NOT NULL,
    timestamp_ms FLOAT,
    class_name VARCHAR(100) NOT NULL,
    class_id INT,
    confidence FLOAT NOT NULL,
    bbox_x1 FLOAT,
    bbox_y1 FLOAT,
    bbox_x2 FLOAT,
    bbox_y2 FLOAT,
    bbox_width FLOAT,
    bbox_height FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_class_name (class_name),
    INDEX idx_frame_number (frame_number)
);

-- Table for analysis statistics
CREATE TABLE IF NOT EXISTS analysis_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    class_name VARCHAR(100) NOT NULL,
    total_detections INT DEFAULT 0,
    avg_confidence FLOAT,
    max_confidence FLOAT,
    min_confidence FLOAT,
    first_appearance_frame INT,
    last_appearance_frame INT,
    total_screen_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id) ON DELETE CASCADE,
    UNIQUE KEY unique_session_class (session_id, class_name)
);

-- Table for user preferences/settings
CREATE TABLE IF NOT EXISTS settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Insert default settings
INSERT INTO settings (setting_key, setting_value, description) VALUES
    ('default_confidence_threshold', '0.5', 'Default confidence threshold for detections'),
    ('default_model', 'yolov8n.pt', 'Default YOLO model to use'),
    ('max_video_size_mb', '500', 'Maximum video file size in MB'),
    ('supported_formats', 'mp4,avi,mov,mkv,webm', 'Supported video formats')
ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON analysis_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON analysis_sessions(created_at);
