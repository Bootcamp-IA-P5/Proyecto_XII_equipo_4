"""
Script de entrenamiento para Kaggle Notebooks
Entrena YOLOv8s en LogoDet-3K dataset

MODOS DE USO:
  A) Dataset YA preparado (formato YOLO): Sube un dataset con estructura
     images/{train,val,test}/ y labels/{train,val,test}/ + data.yaml.
     El script detecta que ya está listo y salta directo al entrenamiento.

  B) Dataset crudo (LogoDet-3K original con XMLs): Sube el LogoDet-3K
     original. El script convierte XML→YOLO automáticamente.

Pasos en Kaggle:
1. Crea un nuevo Notebook (Python) con GPU activada
2. En "Add data" sube tu dataset (preparado o crudo)
3. Copia este código en una celda y ejecuta
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

# ============================================================================
# SETUP
# ============================================================================

# Directorio de trabajo en Kaggle
WORK_DIR = Path('/kaggle/working')
DATA_DIR = Path('/kaggle/input')
OUTPUT_DIR = WORK_DIR / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Working dir: {WORK_DIR}")
print(f"Data dir: {DATA_DIR}")

# Instalar ultralytics si no está
os.system("pip install -q ultralytics opencv-python")

from ultralytics import YOLO

# ============================================================================
# 1. DETECTAR DATASET (¿ya preparado o crudo?)
# ============================================================================

print("\n" + "="*60)
print("DETECTANDO DATASET")
print("="*60)

def find_prepared_yolo_dataset() -> Optional[Path]:
    """
    Busca un dataset YOLO ya preparado en /kaggle/input.
    Estructura esperada: images/{train,val}/ + labels/{train,val}/ + data.yaml
    """
    for d in DATA_DIR.iterdir():
        if not d.is_dir():
            continue
        # Buscar recursivamente data.yaml
        for yaml_file in d.rglob('data.yaml'):
            parent = yaml_file.parent
            has_train_imgs = (parent / 'images' / 'train').is_dir()
            has_train_lbls = (parent / 'labels' / 'train').is_dir()
            has_val_imgs = (parent / 'images' / 'val').is_dir()
            has_val_lbls = (parent / 'labels' / 'val').is_dir()
            if has_train_imgs and has_train_lbls and has_val_imgs and has_val_lbls:
                # Verificar que tenga imágenes
                train_imgs = list((parent / 'images' / 'train').glob('*'))
                if train_imgs:
                    return parent
    return None


def find_raw_logodet_dataset() -> Optional[Path]:
    """Busca el LogoDet-3K crudo (con XMLs) en /kaggle/input."""
    # Nombres típicos
    for path in [
        Path('/kaggle/input/logodet3k'),
        Path('/kaggle/input/logodet-3k'),
        Path('/kaggle/input/LogoDet-3K'),
    ]:
        if path.exists():
            subdirs = list(path.iterdir())
            if subdirs and subdirs[0].is_dir():
                return path

    # Búsqueda genérica
    for d in DATA_DIR.iterdir():
        if not d.is_dir():
            continue
        # Buscar .xml recursivamente (indicador de LogoDet-3K)
        xml_files = list(d.rglob('*.xml'))
        if len(xml_files) > 100:  # mínimo razonable
            return d
    return None


# --- Intentar primero dataset preparado ---
prepared_path = find_prepared_yolo_dataset()

if prepared_path is not None:
    print(f"✅ Dataset YOLO ya preparado encontrado en: {prepared_path}")
    yolo_dataset = prepared_path
    SKIP_CONVERSION = True

    # Contar archivos
    for split in ['train', 'val', 'test']:
        img_dir = yolo_dataset / 'images' / split
        lbl_dir = yolo_dataset / 'labels' / split
        if img_dir.exists():
            img_count = len(list(img_dir.glob('*')))
            lbl_count = len(list(lbl_dir.glob('*'))) if lbl_dir.exists() else 0
            print(f"  {split}: {img_count} imágenes, {lbl_count} labels")

    # Copiar data.yaml a working (ultralytics necesita paths escribibles)
    data_yaml_src = yolo_dataset / 'data.yaml'
    data_yaml = OUTPUT_DIR / 'data.yaml'
    # Leer y ajustar path al directorio real
    yaml_text = data_yaml_src.read_text()
    # Reemplazar path si apunta a otro lugar
    new_yaml = f"path: {yolo_dataset.absolute()}\n"
    lines = yaml_text.splitlines()
    with open(data_yaml, 'w') as f:
        for line in lines:
            if line.strip().startswith('path:'):
                f.write(new_yaml)
            else:
                f.write(line + '\n')
    print(f"✓ data.yaml copiado y ajustado en: {data_yaml}")

else:
    SKIP_CONVERSION = False
    print("No se encontró dataset preparado. Buscando LogoDet-3K crudo...")

    logodet_path = find_raw_logodet_dataset()

    if logodet_path is None:
        print("❌ No encontré ningún dataset válido.")
        print("   Opciones:")
        print("   a) Sube un dataset YOLO preparado (con images/, labels/, data.yaml)")
        print("   b) Sube el LogoDet-3K original (con archivos .xml)")
        print("\n   Directorios en /kaggle/input:")
        for d in DATA_DIR.iterdir():
            print(f"     - {d.name}")
        exit(1)

    print(f"✓ LogoDet-3K crudo encontrado en: {logodet_path}")

    # Mostrar estructura para diagnóstico
    print(f"\nEstructura del dataset:")
    for item in sorted(logodet_path.iterdir())[:10]:
        prefix = "📁" if item.is_dir() else "📄"
        print(f"  {prefix} {item.name}")
        if item.is_dir():
            for sub in sorted(item.iterdir())[:5]:
                sub_prefix = "📁" if sub.is_dir() else "📄"
                print(f"    {sub_prefix} {sub.name}")

    # Crear estructura YOLO
    yolo_dataset = OUTPUT_DIR / 'retrain_yolo'
    for split in ['train', 'val', 'test']:
        (yolo_dataset / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_dataset / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print(f"✓ Estructura YOLO creada en: {yolo_dataset}")

# ============================================================================
# 2. CONVERTIR DATASET (solo si es crudo)
# ============================================================================

def get_all_images_and_labels(source_dir) -> List[Tuple[Path, Path]]:
    """Buscar pares imagen-XML del LogoDet-3K (búsqueda recursiva)"""
    pairs = []
    source_dir = Path(source_dir)
    
    # Búsqueda recursiva - funciona sin importar niveles de anidamiento
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_file in source_dir.rglob(ext):
            xml_file = img_file.with_suffix('.xml')
            if xml_file.exists():
                pairs.append((img_file, xml_file))
    
    return pairs

def parse_xml_annotation(xml_file: Path) -> List[Dict]:
    """Parsear anotación PASCAL VOC"""
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
            name_elem = obj.find('name')
            class_name = name_elem.text if name_elem is not None else 'logo'
            
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            # Clamp
            xmin = max(0, min(xmin, img_width))
            ymin = max(0, min(ymin, img_height))
            xmax = max(0, min(xmax, img_width))
            ymax = max(0, min(ymax, img_height))
            
            if xmax > xmin and ymax > ymin:
                annotations.append({
                    'xmin': xmin, 'ymin': ymin,
                    'xmax': xmax, 'ymax': ymax,
                    'img_width': img_width,
                    'img_height': img_height
                })
    except Exception as e:
        print(f"  Error con {xml_file}: {e}")
    
    return annotations

def convert_to_yolo_format(annotations: List[Dict], class_id: int = 0) -> List[str]:
    """Convertir coordenadas a YOLO (normalizado)"""
    yolo_lines = []
    for ann in annotations:
        img_w = ann['img_width']
        img_h = ann['img_height']
        
        x_center = ((ann['xmin'] + ann['xmax']) / 2) / img_w
        y_center = ((ann['ymin'] + ann['ymax']) / 2) / img_h
        width = (ann['xmax'] - ann['xmin']) / img_w
        height = (ann['ymax'] - ann['ymin']) / img_h
        
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    
    return yolo_lines

# Buscar pares imagen/XML
if not SKIP_CONVERSION:
    print("\n" + "="*60)
    print("CONVIRTIENDO A FORMATO YOLO")
    print("="*60)

    pairs = get_all_images_and_labels(logodet_path)
    print(f"Encontradas {len(pairs)} imágenes con anotaciones XML")

    if not pairs:
        print("❌ No se encontraron pares imagen/XML")
        exit(1)

    # Dividir en train/val/test
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

    import time
    t0 = time.time()

    for split, split_pairs in splits.items():
        print(f"\nProcesando {split} ({len(split_pairs)} pares)...")
        for i, (img_file, xml_file) in enumerate(split_pairs):
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{len(split_pairs)} ({elapsed:.0f}s)")

            try:
                annotations = parse_xml_annotation(xml_file)
                if not annotations:
                    continue

                yolo_lines = convert_to_yolo_format(annotations)

                unique_name = f"{img_file.parent.name}_{img_file.stem}"

                # Guardar label
                label_file = yolo_dataset / 'labels' / split / f"{unique_name}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                # Enlace simbólico en vez de copiar (mucho más rápido)
                dest_img = yolo_dataset / 'images' / split / f"{unique_name}.jpg"
                try:
                    os.symlink(str(img_file), str(dest_img))
                except OSError:
                    # Fallback a copia si symlinks no funcionan
                    shutil.copy2(img_file, dest_img)

                stats[split] += 1
                stats['total_annotations'] += len(yolo_lines)
            except Exception as e:
                print(f"  Error con {img_file}: {e}")

    elapsed_total = time.time() - t0
    print(f"\n✓ Conversión completada en {elapsed_total:.0f}s:")
    print(f"  Train: {stats['train']}")
    print(f"  Val:   {stats['val']}")
    print(f"  Test:  {stats['test']}")
    print(f"  Total anotaciones: {stats['total_annotations']}")

    # Verificar que los archivos realmente existen en disco
    print(f"\n🔍 Verificación de archivos en disco:")
    for split in ['train', 'val', 'test']:
        img_count = len(list((yolo_dataset / 'images' / split).glob('*')))
        lbl_count = len(list((yolo_dataset / 'labels' / split).glob('*')))
        print(f"  {split}: {img_count} imágenes, {lbl_count} labels")

    if stats['train'] == 0:
        print("\n❌ ERROR: No se copiaron imágenes al split de train.")
        print("   Revisa la estructura del dataset y los logs de errores.")
        exit(1)

    # Crear data.yaml
    yaml_content = f"""path: {yolo_dataset.absolute()}
train: images/train
val: images/val
test: images/test

nc: 1

names:
  0: logo
"""
    data_yaml = yolo_dataset / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ data.yaml creado")

else:
    print("\n⏭️  Dataset ya preparado — saltando conversión.")

# ============================================================================
# 4. ENTRENAR MODELO
# ============================================================================

print("\n" + "="*60)
print("ENTRENANDO YOLOV8S")
print("="*60)

model = YOLO('yolov8s.pt')

results = model.train(
    data=str(data_yaml),
    epochs=100,
    imgsz=640,
    batch=32,  # Más batch porque Kaggle tiene 30GB VRAM
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
# 5. GUARDAR MODELO
# ============================================================================

# El modelo se guarda automáticamente en:
# OUTPUT_DIR / 'yolov8s_retrain' / 'weights' / 'best.pt'

best_model = OUTPUT_DIR / 'yolov8s_retrain' / 'weights' / 'best.pt'
if best_model.exists():
    print(f"\n✓ Modelo guardado en: {best_model}")
    print(f"  Tamaño: {best_model.stat().st_size / 1e6:.1f} MB")

# ============================================================================
# 6. VALIDAR MODELO
# ============================================================================

print("\n" + "="*60)
print("VALIDANDO MODELO")
print("="*60)

model_trained = YOLO(str(best_model))
val_results = model_trained.val()

print(f"\n✓ Validación completada")
print(f"  mAP50: {val_results.box.map50:.4f}")
print(f"  mAP50-95: {val_results.box.map:.4f}")

print("\n" + "="*60)
print("✓ PROCESO COMPLETADO")
print("="*60)
print(f"\nModelo entrenado: {best_model}")

# ============================================================================
# 7. EMPAQUETAR PARA DESCARGA
# ============================================================================

import zipfile

zip_path = WORK_DIR / 'best_model.zip'
if best_model.exists():
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(best_model, 'best.pt')
    print(f"\n📦 Modelo empaquetado en: {zip_path}")
    print(f"   Tamaño ZIP: {zip_path.stat().st_size / 1e6:.1f} MB")
    print(f"\n👉 Para descargar: pestaña 'Output' → 'best_model.zip'")
else:
    print(f"\n⚠️  No se encontró {best_model}, no se pudo empaquetar.")
