-- Create database
DROP DATABASE IF EXISTS brand_vision;
CREATE DATABASE brand_vision;
USE brand_vision;

-- Create videos table
CREATE TABLE videos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    duracion_seg FLOAT NOT NULL,
    fecha_procesado DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create brands table (IDs from dataset className2ClassID.txt)
CREATE TABLE brands (
    id INT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL
);

-- Insert static brands (paste your list here)
INSERT INTO brands (id, nombre) VALUES
(0, 'HP'),
(1, 'adidas_symbol'),
(2, 'adidas_text'),
(3, 'aldi'),
(4, 'apple'),
(5, 'becks_symbol'),
(6, 'becks_text'),
(7, 'bmw'),
(8, 'carlsberg_symbol'),
(9, 'carlsberg_text'),
(10, 'chimay_symbol'),
(11, 'chimay_text'),
(12, 'cocacola'),
(13, 'corona_symbol'),
(14, 'corona_text'),
(15, 'dhl'),
(16, 'erdinger_symbol'),
(17, 'erdinger_text'),
(18, 'esso_symbol'),
(19, 'esso_text'),
(20, 'fedex'),
(21, 'ferrari'),
(22, 'ford'),
(23, 'fosters_symbol'),
(24, 'fosters_text'),
(25, 'google'),
(26, 'guinness_symbol'),
(27, 'guinness_text'),
(28, 'heineken'),
(29, 'milka'),
(30, 'nvidia_symbol'),
(31, 'nvidia_text'),
(32, 'paulaner_symbol'),
(33, 'paulaner_text'),
(34, 'pepsi_symbol'),
(35, 'pepsi_text'),
(36, 'rittersport'),
(37, 'shell'),
(38, 'singha_symbol'),
(39, 'singha_text'),
(40, 'starbucks'),
(41, 'stellaartois_symbol'),
(42, 'stellaartois_text'),
(43, 'texaco'),
(44, 'tsingtao_symbol'),
(45, 'tsingtao_text'),
(46, 'ups');

-- Create detections table
CREATE TABLE detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT NOT NULL,
    brand_id INT NOT NULL,
    segundo FLOAT NOT NULL,
    confianza FLOAT NOT NULL,
    bbox_x FLOAT,
    bbox_y FLOAT,
    bbox_w FLOAT,
    bbox_h FLOAT,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
    FOREIGN KEY (brand_id) REFERENCES brands(id) ON DELETE CASCADE
);