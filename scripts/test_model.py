from ultralytics import YOLO
# Ejemplo de ejecución:
# python scripts/test_model.py --model runs/logo_detection/train/weights/best.pt

import argparse
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import random
import sys
from pathlib import Path

# Add project root to path to import src
sys.path.append(str(Path(__file__).parent.parent))
from src.config import DEFAULT_MODEL, OUTPUT_DIR


def test_model(
    model_path: str,
    source: str,
    output_dir: str = 'inference_results',
    conf_threshold: float = 0.25,
    show: bool = False
):
    """
    Prueba el modelo entrenado en imágenes nuevas.
    
    Args:
        model_path: Ruta al modelo .pt
        source: Ruta a imagen o directorio de imágenes
        output_dir: Directorio para guardar resultados
        conf_threshold: Umbral de confianza
        show: Mostrar imágenes en ventana
    """
    print(f"Cargando modelo: {model_path}")
    model = YOLO(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Realizar inferencia
    print(f"Ejecutando inferencia en: {source}")
    results = model.predict(
        source=source,
        conf=conf_threshold,
        save=True,
        project=output_dir,
        name='predict',
        exist_ok=True
    )
    
    print(f"Resultados guardados en {output_dir}/predict")
    
    # Mostrar algunas detecciones si es un directorio
    if Path(source).is_dir():
        result_files = list(Path(f"{output_dir}/predict").glob('*.jpg')) + \
                       list(Path(f"{output_dir}/predict").glob('*.png'))
        
        if result_files:
            # Mostrar 3 imágenes aleatorias
            samples = random.sample(result_files, min(3, len(result_files)))
            
            # Solo si estamos en un entorno que soporte visualización (opcional)
            # Aquí solo imprimimos las rutas
            print("Imágenes generadas (muestras):")
            for sample in samples:
                print(f" - {sample}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Probar modelo YOLOv8')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help=f'Ruta al modelo entrenado (default: {DEFAULT_MODEL})')
    parser.add_argument('--source', type=str, default=None, help='Fuente (imagen o carpeta). Si no se indica, usa una imagen de validación.')
    parser.add_argument('--output', type=str, default='inference_results', help='Carpeta de salida')
    parser.add_argument('--conf', type=float, default=0.25, help='Umbral de confianza')
    
    args = parser.parse_args()
    
    # Determine source if not provided
    source = args.source
    if source is None:
        # Try to find a validation image
        val_img_dir = OUTPUT_DIR / "yolo_dataset" / "images" / "val"
        if val_img_dir.exists():
            images = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
            if images:
                source = str(random.choice(images))
                print(f"No se especificó fuente. Usando imagen aleatoria de validación: {source}")
            else:
                print("No hay imágenes en la carpeta de validación.")
        
    if not source:
        print("Error: Debes especificar una fuente (--source) o generar el dataset primero.")
        exit(1)
    
    test_model(
        model_path=args.model,
        source=source,
        output_dir=args.output,
        conf_threshold=args.conf
    )
