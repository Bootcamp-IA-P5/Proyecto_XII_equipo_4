"""
Script de entrenamiento para Google Colab
Entrena YOLOv8s en LogoDet-3K dataset

PASOS:
1. Sube tu dataset a Google Drive (ver opciones abajo)
2. Abre Google Colab (colab.research.google.com)
3. Cambia el runtime a GPU: Runtime → Change runtime type → T4 GPU
4. Copia cada sección en una celda y ejecuta en orden

OPCIONES DE DATASET:
  A) Dataset YOLO ya preparado en Drive (con images/, labels/, data.yaml)
  B) Solo labels en Drive + LogoDet-3K crudo en Drive
  C) LogoDet-3K crudo completo en Drive (el script convierte)
"""

# ============================================================================
# CELDA 1: MONTAR GOOGLE DRIVE
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

print("✓ Google Drive montado en /content/drive/MyDrive/")

# ============================================================================
# CELDA 2: CONFIGURACIÓN
# ============================================================================

import os
import shutil
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

# Instalar ultralytics
os.system("pip install -q ultralytics opencv-python")
from ultralytics import YOLO

# --- CONFIGURA ESTAS RUTAS ---
# Pon aquí la ruta dentro de tu Google Drive donde subiste los datos.
# Ejemplos:
#   "BrandVision/retrain_yolo"         → carpeta preparada (descomprimida)
#   "BrandVision/retrain_yolo.zip"     → ZIP del dataset preparado
#   "BrandVision/LogoDet-3K"           → dataset crudo

DRIVE_DATASET_PATH = "BrandVision/retrain_yolo.zip"  # ← CAMBIA ESTO

# --- NO TOCAR ---
DRIVE_BASE = Path('/content/drive/MyDrive')
DATASET_PATH = DRIVE_BASE / DRIVE_DATASET_PATH
WORK_DIR = Path('/content/working')
OUTPUT_DIR = WORK_DIR / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Carpeta en Drive donde se guardará el modelo entrenado
DRIVE_OUTPUT = DRIVE_BASE / 'BrandVision' / 'trained_models'
DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"Dataset: {DATASET_PATH}")
print(f"Existe: {DATASET_PATH.exists()}")
print(f"Working dir: {WORK_DIR}")
print(f"Output en Drive: {DRIVE_OUTPUT}")

# Verificar GPU
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    gpu_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    print(f"\n✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("\n⚠️  No hay GPU. Ve a Runtime → Change runtime type → T4 GPU")

# ============================================================================
# CELDA 3: DETECTAR TIPO DE DATASET
# ============================================================================

print("\n" + "="*60)
print("DETECTANDO DATASET")
print("="*60)

SKIP_CONVERSION = False
yolo_dataset = None
data_yaml = None


def is_prepared_yolo_dataset(path: Path) -> bool:
    """Verificar si es un dataset YOLO ya preparado."""
    has_yaml = (path / 'data.yaml').exists()
    has_train = (path / 'images' / 'train').is_dir()
    has_labels = (path / 'labels' / 'train').is_dir()
    return has_yaml and has_train and has_labels


def find_raw_logodet(path: Path) -> Optional[Path]:
    """Buscar LogoDet-3K crudo (con XMLs)."""
    xml_files = list(path.rglob('*.xml'))
    if len(xml_files) > 100:
        return path
    return None


if not DATASET_PATH.exists():
    print(f"❌ No se encontró: {DATASET_PATH}")
    print(f"   Verifica que la ruta en DRIVE_DATASET_PATH sea correcta.")
    print(f"\n   Contenido de {DRIVE_BASE}:")
    for item in sorted(DRIVE_BASE.iterdir())[:20]:
        print(f"     {'📁' if item.is_dir() else '📄'} {item.name}")
    raise SystemExit(1)

# Si es un ZIP, descomprimir primero
if DATASET_PATH.suffix == '.zip':
    import zipfile
    extract_dir = WORK_DIR / 'retrain_yolo'
    if not extract_dir.exists():
        print(f"📦 Descomprimiendo {DATASET_PATH.name}...")
        t0 = time.time()
        with zipfile.ZipFile(DATASET_PATH, 'r') as zf:
            zf.extractall(extract_dir)
        print(f"✓ Descomprimido en {time.time() - t0:.0f}s → {extract_dir}")
    else:
        print(f"✓ Ya descomprimido en {extract_dir}")
    DATASET_PATH = extract_dir

# Diagnóstico: mostrar qué hay dentro
print(f"\nContenido de {DATASET_PATH}:")
for item in sorted(DATASET_PATH.iterdir())[:15]:
    prefix = "📁" if item.is_dir() else "📄"
    print(f"  {prefix} {item.name}")
    if item.is_dir():
        for sub in sorted(item.iterdir())[:5]:
            sub_prefix = "📁" if sub.is_dir() else "📄"
            print(f"    {sub_prefix} {sub.name}")

# Buscar data.yaml recursivamente (por si está en subcarpeta)
def find_yolo_root(base: Path) -> Optional[Path]:
    """Busca recursivamente la raíz del dataset YOLO."""
    # Primero comprobar en base directamente
    if is_prepared_yolo_dataset(base):
        return base
    # Buscar data.yaml en subcarpetas
    for yaml_file in base.rglob('data.yaml'):
        candidate = yaml_file.parent
        if is_prepared_yolo_dataset(candidate):
            return candidate
    return None

yolo_root = find_yolo_root(DATASET_PATH)
if yolo_root and yolo_root != DATASET_PATH:
    print(f"\n✓ Dataset YOLO encontrado en subcarpeta: {yolo_root.relative_to(DATASET_PATH)}")
    DATASET_PATH = yolo_root

# Caso A: Dataset YOLO completo
if is_prepared_yolo_dataset(DATASET_PATH):
    print(f"✅ Dataset YOLO ya preparado encontrado")
    yolo_dataset = DATASET_PATH
    SKIP_CONVERSION = True

    for split in ['train', 'val', 'test']:
        img_dir = yolo_dataset / 'images' / split
        if img_dir.exists():
            count = len(list(img_dir.glob('*')))
            print(f"  {split}: {count} imágenes")

    # Crear data.yaml ajustado
    data_yaml = OUTPUT_DIR / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write(f"""path: {yolo_dataset.absolute()}
train: images/train
val: images/val
test: images/test

nc: 1

names:
  0: logo
""")
    print(f"✓ data.yaml creado: {data_yaml}")

# Caso B: LogoDet-3K crudo
else:
    logodet_path = find_raw_logodet(DATASET_PATH)
    if logodet_path is None:
        print(f"❌ No se reconoce el dataset en: {DATASET_PATH}")
        print("   Debe ser:")
        print("   - Dataset YOLO (con images/, labels/, data.yaml)")
        print("   - LogoDet-3K crudo (con archivos .xml)")
        raise SystemExit(1)

    print(f"✓ LogoDet-3K crudo encontrado en: {logodet_path}")
    SKIP_CONVERSION = False

    # Crear estructura YOLO en disco local (rápido)
    yolo_dataset = WORK_DIR / 'retrain_yolo'
    for split in ['train', 'val', 'test']:
        (yolo_dataset / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_dataset / 'labels' / split).mkdir(parents=True, exist_ok=True)

# ============================================================================
# CELDA 4: CONVERSIÓN (solo si el dataset es crudo)
# ============================================================================


def get_all_images_and_labels(source_dir) -> List[Tuple[Path, Path]]:
    """Buscar pares imagen-XML (búsqueda recursiva)."""
    pairs = []
    source_dir = Path(source_dir)
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_file in source_dir.rglob(ext):
            xml_file = img_file.with_suffix('.xml')
            if xml_file.exists():
                pairs.append((img_file, xml_file))
    return pairs


def parse_xml_annotation(xml_file: Path) -> List[Dict]:
    """Parsear anotación PASCAL VOC."""
    annotations = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size_elem = root.find('size')
        if size_elem is None:
            return annotations

        img_width = int(size_elem.find('width').text)
        img_height = int(size_elem.find('height').text)

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))

            xmin = max(0, min(xmin, img_width))
            ymin = max(0, min(ymin, img_height))
            xmax = max(0, min(xmax, img_width))
            ymax = max(0, min(ymax, img_height))

            if xmax > xmin and ymax > ymin:
                annotations.append({
                    'xmin': xmin, 'ymin': ymin,
                    'xmax': xmax, 'ymax': ymax,
                    'img_width': img_width, 'img_height': img_height
                })
    except Exception as e:
        print(f"  Error con {xml_file}: {e}")
    return annotations


def convert_to_yolo_format(annotations: List[Dict], class_id: int = 0) -> List[str]:
    """Convertir coordenadas a YOLO (normalizado)."""
    yolo_lines = []
    for ann in annotations:
        img_w, img_h = ann['img_width'], ann['img_height']
        x_center = max(0.0, min(1.0, ((ann['xmin'] + ann['xmax']) / 2) / img_w))
        y_center = max(0.0, min(1.0, ((ann['ymin'] + ann['ymax']) / 2) / img_h))
        width = max(0.0, min(1.0, (ann['xmax'] - ann['xmin']) / img_w))
        height = max(0.0, min(1.0, (ann['ymax'] - ann['ymin']) / img_h))
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_lines


if not SKIP_CONVERSION:
    print("\n" + "="*60)
    print("CONVIRTIENDO A FORMATO YOLO")
    print("="*60)

    pairs = get_all_images_and_labels(logodet_path)
    print(f"Encontradas {len(pairs)} imágenes con anotaciones XML")

    if not pairs:
        print("❌ No se encontraron pares imagen/XML")
        raise SystemExit(1)

    random.seed(42)
    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)

    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }

    stats = {'train': 0, 'val': 0, 'test': 0, 'total_annotations': 0}
    t0 = time.time()

    for split, split_pairs in splits.items():
        print(f"\nProcesando {split} ({len(split_pairs)} pares)...")
        for i, (img_file, xml_file) in enumerate(split_pairs):
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(split_pairs)} ({time.time() - t0:.0f}s)")
            try:
                annotations = parse_xml_annotation(xml_file)
                if not annotations:
                    continue
                yolo_lines = convert_to_yolo_format(annotations)
                unique_name = f"{img_file.parent.name}_{img_file.stem}"

                label_file = yolo_dataset / 'labels' / split / f"{unique_name}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                dest_img = yolo_dataset / 'images' / split / f"{unique_name}.jpg"
                try:
                    os.symlink(str(img_file), str(dest_img))
                except OSError:
                    shutil.copy2(img_file, dest_img)

                stats[split] += 1
                stats['total_annotations'] += len(yolo_lines)
            except Exception as e:
                print(f"  Error: {e}")

    print(f"\n✓ Conversión en {time.time() - t0:.0f}s:")
    print(f"  Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
    print(f"  Total anotaciones: {stats['total_annotations']}")

    # Crear data.yaml
    data_yaml = yolo_dataset / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write(f"""path: {yolo_dataset.absolute()}
train: images/train
val: images/val
test: images/test

nc: 1

names:
  0: logo
""")

    if stats['train'] == 0:
        print("❌ No se copiaron imágenes al split de train.")
        raise SystemExit(1)

else:
    print("\n⏭️  Dataset ya preparado — saltando conversión.")

# ============================================================================
# CELDA 5: ENTRENAR MODELO
# ============================================================================

print("\n" + "="*60)
print("ENTRENANDO YOLOV8S")
print("="*60)

model = YOLO('yolov8s.pt')

results = model.train(
    data=str(data_yaml),
    epochs=100,
    imgsz=640,
    batch=16,       # T4 tiene 16GB VRAM → batch 16 es seguro
    device=0,
    patience=20,
    project=str(OUTPUT_DIR),
    name='yolov8s_retrain',
    exist_ok=True,
    verbose=True,
    save=True
)

print("\n✓ Entrenamiento completado")

# ============================================================================
# CELDA 6: VALIDAR Y GUARDAR
# ============================================================================

print("\n" + "="*60)
print("VALIDANDO MODELO")
print("="*60)

best_model = OUTPUT_DIR / 'yolov8s_retrain' / 'weights' / 'best.pt'

if best_model.exists():
    print(f"✓ Modelo: {best_model} ({best_model.stat().st_size / 1e6:.1f} MB)")

    model_trained = YOLO(str(best_model))
    val_results = model_trained.val()

    print(f"\n✓ Validación completada")
    print(f"  mAP50:    {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")

    # Copiar modelo a Google Drive
    drive_model = DRIVE_OUTPUT / 'best.pt'
    shutil.copy2(best_model, drive_model)
    print(f"\n✓ Modelo copiado a Drive: {drive_model}")

    # También copiar last.pt
    last_model = OUTPUT_DIR / 'yolov8s_retrain' / 'weights' / 'last.pt'
    if last_model.exists():
        shutil.copy2(last_model, DRIVE_OUTPUT / 'last.pt')

    # Copiar gráficas de entrenamiento
    for plot_file in (OUTPUT_DIR / 'yolov8s_retrain').glob('*.png'):
        shutil.copy2(plot_file, DRIVE_OUTPUT / plot_file.name)

    print(f"\n{'='*60}")
    print(f"✓ PROCESO COMPLETADO")
    print(f"{'='*60}")
    print(f"\nModelo guardado en Google Drive:")
    print(f"  📁 {DRIVE_OUTPUT}")
    print(f"  📄 best.pt ({drive_model.stat().st_size / 1e6:.1f} MB)")
    print(f"\nDescárgalo desde Drive o accede en tu proyecto local.")
else:
    print(f"❌ No se encontró {best_model}")
