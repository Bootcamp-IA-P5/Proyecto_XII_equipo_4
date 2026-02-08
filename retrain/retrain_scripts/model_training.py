"""
Entrenamiento del modelo YOLO
"""

from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO

from .utils import setup_logging, ensure_directory_exists


logger = setup_logging()


def train_model(
    model_path: str,
    dataset_yaml: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: int = 0,
    patience: int = 20,
    output_dir: str = None
) -> Dict:
    """
    Entrena el modelo YOLO
    
    Args:
        model_path: Ruta al modelo base (ej: models/best.pt)
        dataset_yaml: Ruta al archivo datasets.yaml
        epochs: Número de epochs (default 100)
        imgsz: Tamaño de imagen (default 640)
        batch_size: Tamaño del batch (default 16)
        device: GPU device ID (default 0) o 'cpu'
        patience: Early stopping patience (default 20)
        output_dir: Directorio para guardar resultados
    
    Returns:
        Diccionario con información del entrenamiento
    """
    logger.info("Iniciando entrenamiento del modelo...")
    
    # Usar directorio por defecto si no se especifica
    if output_dir is None:
        output_dir = "models/retrain/train_results"
    
    ensure_directory_exists(output_dir)
    
    try:
        # Cargar modelo
        logger.info(f"Cargando modelo desde {model_path}")
        model = YOLO(model_path)
        
        # Entrenar
        logger.info(f"Iniciando entrenamiento con {epochs} epochs")
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=patience,
            project=output_dir,
            name='retrain_run',
            exist_ok=True,
            verbose=True,
            save=True
        )
        
        training_info = {
            'success': True,
            'results': results,
            'model_path': str(model_path),
            'dataset_yaml': str(dataset_yaml),
            'epochs': epochs,
            'batch_size': batch_size,
            'output_dir': str(output_dir)
        }
        
        logger.info("Entrenamiento completado exitosamente")
        return training_info
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def save_trained_model(
    model_path: str,
    output_path: str = "models/retrain/best_retrained.pt"
) -> bool:
    """
    Guarda el modelo entrenado
    
    Args:
        model_path: Ruta al modelo entrenado
        output_path: Ruta donde guardar el modelo
    
    Returns:
        True si se guardó exitosamente
    """
    try:
        model = YOLO(model_path)
        output_file = Path(output_path)
        ensure_directory_exists(output_file.parent)
        
        model.save(str(output_file))
        logger.info(f"Modelo guardado en {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error al guardar el modelo: {e}")
        return False
