"""
Utilidades comunes para el reentrenamiento
"""

import os
import logging
from pathlib import Path


def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Configura logging para el reentrenamiento
    
    Args:
        log_file: Ruta del archivo de log (opcional)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger('Retrain')
    logger.setLevel(logging.DEBUG)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (opcional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directory_exists(path: str) -> Path:
    """
    Asegura que un directorio existe, lo crea si no existe
    
    Args:
        path: Ruta del directorio
    
    Returns:
        Path object del directorio
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_image_files(directory: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
    """
    Obtiene todos los archivos de imagen de un directorio
    
    Args:
        directory: Ruta del directorio
        extensions: Extensiones de archivo a buscar
    
    Returns:
        Lista de rutas de imágenes
    """
    image_path = Path(directory)
    images = []
    
    for ext in extensions:
        images.extend(image_path.glob(f'*{ext}'))
        images.extend(image_path.glob(f'*{ext.upper()}'))
    
    return sorted(images)
