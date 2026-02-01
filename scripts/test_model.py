from ultralytics import YOLO
# Ejemplo de ejecución:
# python scripts/test_model.py --model runs/logo_detection/train/weights/best.pt
# Para definir los argumentos:
# python scripts/train_yolo.py --epochs 100 --batch 8 --imgsz 640 --name full_training_run

#Para ver los resultados del entrenamiento:
# python scripts/test_model.py --model runs/logo_detection/full_training_run/weights/best.pt --source path/to/test/images
# \runs\detect\runs\logo_detection\full_training_run\results.csv
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
from src.config import DEFAULT_MODEL, OUTPUT_DIR, MODELS_DIR
import shutil


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
    
    # Guardar una copia en la carpeta de modelos central
    model_dest = MODELS_DIR / Path(model_path).name
    if not model_dest.exists():
        print(f"Guardando copia del modelo en: {model_dest}")
        shutil.copy(model_path, model_dest)
    else:
        print(f"El modelo ya existe en la carpeta central: {model_dest}")
    
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

    # Calcular métricas si hay ground truth disponible
    # Se asume que si el source es un directorio de validación, existen labels en formato YOLO
    # y que el modelo fue entrenado con ese mismo dataset
    if Path(source).is_dir():
        # Intentar encontrar el directorio de labels de validación
        # Usamos las rutas configuradas en src.config para mayor seguridad
        val_labels_dir = OUTPUT_DIR / "yolo_dataset" / "labels" / "val"
        val_images_dir = OUTPUT_DIR / "yolo_dataset" / "images" / "val"
        data_yaml_path = OUTPUT_DIR / "yolo_dataset" / "data.yaml"
        
        if val_labels_dir.exists() and val_images_dir.exists() and data_yaml_path.exists():
            print("Calculando métricas de evaluación sobre el conjunto de validación...")
            try:
                metrics = model.val(
                    data=str(data_yaml_path),
                    split="val"
                )
                print("\n--- MÉTRICAS DE EVALUACIÓN ---")
                print(f"Precisión (precision): {metrics.box.precision:.3f}")
                print(f"Recall: {metrics.box.recall:.3f}")
                print(f"F1-score: {metrics.box.f1:.3f}")
                print(f"mAP50: {metrics.box.map50:.3f}")
                print(f"mAP50-95: {metrics.box.map:.3f}")
                print(f"Número de imágenes evaluadas: {metrics.box.seen}")
                
                # Overfitting: comparar mAP de validación vs entrenamiento si está disponible
                print("\nPara evaluar overfitting, compara el mAP de validación mostrado aquí con el mAP de entrenamiento reportado al final del entrenamiento en train_yolo.py.")
            except Exception as e:
                print(f"No se pudieron calcular métricas de validación: {e}")
        else:
            print("No se encontraron etiquetas de validación para calcular métricas.")

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
