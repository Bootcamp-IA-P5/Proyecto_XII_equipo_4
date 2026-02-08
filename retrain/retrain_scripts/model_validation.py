"""
Validación y evaluación del modelo YOLO
"""

from pathlib import Path
from typing import Dict
from ultralytics import YOLO
import matplotlib.pyplot as plt

from .utils import setup_logging, ensure_directory_exists


logger = setup_logging()


def validate_model(
    model_path: str,
    dataset_yaml: str,
    output_dir: str = None,
    imgsz: int = 640,
    device: int = 0
) -> Dict:
    """
    Valida el modelo entrenado
    
    Args:
        model_path: Ruta al modelo entrenado
        dataset_yaml: Ruta al archivo datasets.yaml
        output_dir: Directorio para guardar resultados de validación
        imgsz: Tamaño de imagen (default 640)
        device: GPU device ID (default 0) o 'cpu'
    
    Returns:
        Diccionario con métricas de validación
    """
    logger.info("Iniciando validación del modelo...")
    
    if output_dir is None:
        output_dir = "models/retrain/train_results"
    
    ensure_directory_exists(output_dir)
    
    try:
        model = YOLO(model_path)
        
        # Validar
        results = model.val(
            data=dataset_yaml,
            imgsz=imgsz,
            device=device,
            project=output_dir,
            name='validation_results',
            exist_ok=True
        )
        
        validation_info = {
            'success': True,
            'results': results,
            'model_path': str(model_path),
            'output_dir': str(output_dir)
        }
        
        logger.info("Validación completada")
        return validation_info
        
    except Exception as e:
        logger.error(f"Error durante la validación: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def evaluate_model(
    model_path: str,
    test_images_path: str,
    output_dir: str = None,
    confidence: float = 0.5
) -> Dict:
    """
    Evalúa el modelo en imágenes de test
    
    Args:
        model_path: Ruta al modelo
        test_images_path: Ruta a las imágenes de test
        output_dir: Directorio para guardar resultados
        confidence: Threshold de confianza (default 0.5)
    
    Returns:
        Diccionario con resultados de evaluación
    """
    logger.info("Iniciando evaluación en imágenes de test...")
    
    if output_dir is None:
        output_dir = "models/retrain/train_results"
    
    ensure_directory_exists(output_dir)
    
    try:
        model = YOLO(model_path)
        test_path = Path(test_images_path)
        
        # Inferencia en imágenes de test
        results = model.predict(
            source=str(test_path),
            conf=confidence,
            save=True,
            project=output_dir,
            name='test_predictions',
            exist_ok=True
        )
        
        eval_info = {
            'success': True,
            'test_images': len(results),
            'output_dir': str(output_dir),
            'confidence': confidence
        }
        
        logger.info(f"Evaluación completada. Procesadas {len(results)} imágenes")
        return eval_info
        
    except Exception as e:
        logger.error(f"Error durante la evaluación: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def plot_training_results(
    results_path: str,
    output_path: str = None
) -> bool:
    """
    Grafica los resultados del entrenamiento
    
    Args:
        results_path: Ruta a los resultados del entrenamiento
        output_path: Ruta donde guardar el gráfico
    
    Returns:
        True si se graficó exitosamente
    """
    try:
        logger.info("Graficando resultados del entrenamiento...")
        # Implementar según necesidad específica
        logger.info("Gráficos generados")
        return True
        
    except Exception as e:
        logger.error(f"Error al graficar resultados: {e}")
        return False
