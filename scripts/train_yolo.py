from ultralytics import YOLO
# Ejemplo de ejecución:
# python scripts/train_yolo.py

# Para definir los argumentos:
# python scripts/train_yolo.py --epochs 100 --batch 8 --imgsz 640 --name full_training_run

# Modelos guardados en: models/best.pt y models/last.pt
# Métricas guardadas en: results/<nombre>_metrics.json

import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path to import back
sys.path.append(str(Path(__file__).parent.parent))
from back.services.config import OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, ensure_directories


def train_yolo(
    data_yaml: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    experiment_name: str = 'train'
):
    """
    Entrena un modelo YOLOv8.
    Guarda modelos en models/ y métricas en results/.
    
    Args:
        data_yaml: Ruta al archivo data.yaml
        model_name: Modelo base a utilizar (yolov8n.pt, yolov8s.pt, etc.)
        epochs: Número de épocas
        img_size: Tamaño de imagen
        batch_size: Tamaño del batch
        experiment_name: Nombre del experimento
    """
    ensure_directories()
    
    print("=" * 60)
    print("🎯 ENTRENAMIENTO YOLO - DETECCIÓN DE LOGOS")
    print("=" * 60)
    print(f"Iniciando entrenamiento con {model_name}...")
    print(f"Datos: {data_yaml}")
    print(f"Épocas: {epochs}")
    
    # Cargar modelo
    model = YOLO(model_name)
    
    # Directorio temporal para entrenamiento (ruta absoluta)
    project_root = Path(__file__).parent.parent
    temp_project = project_root / "temp_training"
    
    # Argumentos adicionales
    args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'project': str(temp_project),
        'name': experiment_name,
        'patience': 10,  # Early stopping
        'save': True,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'workers': 2,  # Reducido para evitar consumo excesivo de RAM en Windows
        'device': 'cpu',  # Forzar CPU si no hay GPU nvidia detectada
        'plots': False,  # No generar gráficos
    }
    
    # Entrenar
    results = model.train(**args)
    print("\n✅ Entrenamiento completado!")
    
    # Copiar modelos a la carpeta models/
    experiment_dir = temp_project / experiment_name
    weights_dir = experiment_dir / "weights"
    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"
    results_csv_src = experiment_dir / "results.csv"
    
    if best_src.exists():
        shutil.copy(best_src, MODELS_DIR / "best.pt")
        print(f"✅ Modelo best.pt copiado a: {MODELS_DIR / 'best.pt'}")
    else:
        print(f"⚠️ No se encontró best.pt en: {best_src}")
    
    if last_src.exists():
        shutil.copy(last_src, MODELS_DIR / "last.pt")
        print(f"✅ Modelo last.pt copiado a: {MODELS_DIR / 'last.pt'}")
    else:
        print(f"⚠️ No se encontró last.pt en: {last_src}")
    
    # Copiar results.csv a results/
    if results_csv_src.exists():
        shutil.copy(results_csv_src, RESULTS_DIR / f"{experiment_name}_results.csv")
        print(f"✅ results.csv copiado a: {RESULTS_DIR / f'{experiment_name}_results.csv'}")

    # Validar y guardar métricas
    print("\n--- MÉTRICAS DE VALIDACIÓN ---")
    try:
        metrics = model.val(data=data_yaml, split='val', plots=False, project=str(temp_project), name='val', exist_ok=True)
        print(f"Precisión (precision): {metrics.box.mp:.3f}")
        print(f"Recall: {metrics.box.mr:.3f}")
        print(f"mAP50: {metrics.box.map50:.3f}")
        print(f"mAP50-95: {metrics.box.map:.3f}")
        print(f"Número de imágenes evaluadas: {metrics.box.seen}")
        
        # Guardar métricas en archivo JSON
        metrics_data = {
            "model_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "training_info": {
                "base_model": model_name,
                "epochs": epochs,
                "img_size": img_size,
                "batch_size": batch_size,
                "data_yaml": data_yaml
            },
            "metrics": {
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "images_evaluated": int(metrics.box.seen)
            }
        }
        
        results_file = RESULTS_DIR / f"{experiment_name}_metrics.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Métricas guardadas en: {results_file}")
        
    except Exception as e:
        print(f"Nota: No se pudo ejecutar validación adicional: {e}")

    # Limpiar directorios temporales
    print("\n🧹 Limpiando archivos temporales...")
    try:
        if temp_project.exists():
            shutil.rmtree(temp_project)
            print("✅ temp_training/ eliminado")
    except Exception as e:
        print(f"⚠️ No se pudo eliminar temp_training/: {e}")
    
    # Eliminar runs/ si se creó (YOLO a veces lo crea por defecto)
    runs_dir = project_root / "runs"
    try:
        if runs_dir.exists():
            shutil.rmtree(runs_dir)
            print("✅ runs/ eliminado")
    except Exception as e:
        print(f"⚠️ No se pudo eliminar runs/: {e}")

    print("\n" + "=" * 60)
    print("🎉 ENTRENAMIENTO FINALIZADO")
    print("=" * 60)
    print(f"📁 Modelos guardados en: {MODELS_DIR}")
    print(f"📊 Métricas guardadas en: {RESULTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar YOLOv8 para detección de logos')
    
    # Default paths
    default_data = OUTPUT_DIR / "yolo_dataset" / "data.yaml"
    
    parser.add_argument('--data', type=str, default=str(default_data), 
                        help=f'Ruta al archivo data.yaml (default: {default_data})')
    parser.add_argument('--model', type=str, default="yolov8n.pt", 
                        help='Modelo base (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Tamaño de imagen')
    parser.add_argument('--name', type=str, default='train', help='Nombre del experimento')
    
    args = parser.parse_args()
    
    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch,
        experiment_name=args.name
    )
