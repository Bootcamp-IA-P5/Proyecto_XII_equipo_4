from ultralytics import YOLO
# Ejemplo de ejecución:
# python scripts/train_yolo.py

# Para definir los argumentos:
# python scripts/train_yolo.py --epochs 100 --batch 8 --imgsz 640 --name full_training_run

#Para ver los resultados del entrenamiento:
# python scripts/test_model.py --model runs/logo_detection/full_training_run/weights/best.pt --source path/to/test/images
# \runs\detect\runs\logo_detection\full_training_run\results.csv
import argparse
import os
import sys
from pathlib import Path

# Para lanzar el entrenamiento:
# python scripts/train_yolo.py --epochs 100 --batch 8 --imgsz 640 --name full_training_run

# Add project root to path to import src
sys.path.append(str(Path(__file__).parent.parent))
from src.config import OUTPUT_DIR, DEFAULT_MODEL


def train_yolo(
    data_yaml: str,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    project_name: str = 'logo_detection',
    experiment_name: str = 'exp',
    single_class: str = None
):
    """
    Entrena un modelo YOLOv8.
    
    Args:
        data_yaml: Ruta al archivo data.yaml
        model_name: Modelo base a utilizar (yolov8n.pt, yolov8s.pt, etc.)
        epochs: Número de épocas
        img_size: Tamaño de imagen
        batch_size: Tamaño del batch
        project_name: Nombre del proyecto (carpeta en runs/)
        experiment_name: Nombre del experimento
        single_class: Si se entrena una sola clase (opcional)
    """
    print(f"Iniciando entrenamiento con {model_name}...")
    print(f"Datos: {data_yaml}")
    print(f"Épocas: {epochs}")
    
    # Cargar modelo
    model = YOLO(model_name)
    
    # Argumentos adicionales
    args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'project': f"runs/{project_name}",
        'name': experiment_name,
        'patience': 10,  # Early stopping
        'save': True,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'workers': 2,  # Reducido para evitar consumo excesivo de RAM en Windows
        'device': 'cpu' # Forzar CPU si no hay GPU nvidia detectada (aunque auto lo hace, es mas seguro explicitar si sabemos el entorno)
    }
    
    # Entrenar
    results = model.train(**args)
    print(f"Entrenamiento finalizado. Resultados guardados en runs/{project_name}/{experiment_name}")

    # Mostrar métricas de entrenamiento
    # Mostrar métricas de entrenamiento (Calculando sobre el set de entrenamiento)
    print("\n--- MÉTRICAS DE ENTRENAMIENTO (Evaluar Overfitting) ---")
    print("Calculando métricas sobre el conjunto de TRAIN (esto puede tardar)...")
    try:
        # Forzamos validación en split='train' para ver rendimiento real en entrenamiento
        train_metrics = model.val(split='train')
        
        print(f"Precisión (precision): {train_metrics.box.precision:.3f}")
        print(f"Recall: {train_metrics.box.recall:.3f}")
        print(f"F1-score: {train_metrics.box.f1:.3f}")
        print(f"mAP50: {train_metrics.box.map50:.3f}")
        print(f"mAP50-95: {train_metrics.box.map:.3f}")
    except Exception as e:
        print(f"No se pudieron calcular métricas de entrenamiento: {e}")


    # Validar en el conjunto de validación
    print("\nEjecutando validación...")
    metrics = model.val()
    print("\n--- MÉTRICAS DE VALIDACIÓN ---")
    print(f"Precisión (precision): {metrics.box.precision:.3f}")
    print(f"Recall: {metrics.box.recall:.3f}")
    print(f"F1-score: {metrics.box.f1:.3f}")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Número de imágenes evaluadas: {metrics.box.seen}")

    # Overfitting: sugerencia
    print("\nPara evaluar overfitting, compara las métricas de entrenamiento y validación. Si el mAP de entrenamiento es mucho mayor que el de validación, puede haber overfitting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar YOLOv8 para detección de logos')
    
    # Default paths
    default_data = OUTPUT_DIR / "yolo_dataset" / "data.yaml"
    
    parser.add_argument('--data', type=str, default=str(default_data), 
                        help=f'Ruta al archivo data.yaml (default: {default_data})')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, 
                        help=f'Modelo base (default: {DEFAULT_MODEL})')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Tamaño de imagen')
    parser.add_argument('--project', type=str, default='logo_detection', help='Nombre del proyecto')
    parser.add_argument('--name', type=str, default='train', help='Nombre del experimento')
    parser.add_argument('--single-class', type=str, default=None, help='Nombre de clase única (para log)')
    
    args = parser.parse_args()
    
    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch,
        project_name=args.project,
        experiment_name=args.name,
        single_class=args.single_class
    )
