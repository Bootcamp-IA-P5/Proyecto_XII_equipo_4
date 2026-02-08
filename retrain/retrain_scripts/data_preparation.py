"""
Preparación de datos para el reentrenamiento del modelo YOLO.

Este módulo soporta dos flujos:

  A) Dataset pre-etiquetado (LogoDet-3K, PASCAL VOC XML):
     convert_logodet_dataset() convierte XML → formato YOLO.

  B) Imágenes crudas sin etiquetar:
     prepare_raw_dataset() crea la estructura YOLO con labels vacíos
     para etiquetar manualmente (ej. con LabelImg / Roboflow).

Ambos flujos generan un data.yaml listo para YOLOv8.

Autor: Equipo 4 - Bootcamp IA
"""

import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .utils import setup_logging, ensure_directory_exists, get_image_files


logger = setup_logging()


# =============================================================================
# 1. FUNCIONES DE LECTURA Y PARSEO
# =============================================================================

def get_all_images_and_labels(source_dir: str) -> List[Tuple[str, str]]:
    """
    Busca todos los pares imagen-XML del dataset LogoDet-3K.

    Recorre la estructura: Categoría / Marca / {id.jpg, id.xml}

    Args:
        source_dir: Directorio raíz de LogoDet-3K

    Returns:
        Lista de tuplas (ruta_imagen, ruta_xml)
    """
    pairs = []
    source_path = Path(source_dir)

    for category_dir in source_path.iterdir():
        if not category_dir.is_dir():
            continue
        for brand_dir in category_dir.iterdir():
            if not brand_dir.is_dir():
                continue
            for img_file in brand_dir.glob('*.jpg'):
                xml_file = img_file.with_suffix('.xml')
                if xml_file.exists():
                    pairs.append((str(img_file), str(xml_file)))

    return pairs


def parse_xml_annotation(xml_file: str) -> List[Dict]:
    """
    Parsea un archivo XML en formato PASCAL VOC.

    Extrae las dimensiones de la imagen y las coordenadas de cada
    bounding box anotado.

    Args:
        xml_file: Ruta al archivo XML

    Returns:
        Lista de diccionarios con las anotaciones:
        [{class_name, xmin, ymin, xmax, ymax, img_width, img_height}, ...]
    """
    annotations = []

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Dimensiones de la imagen
        size_elem = root.find('size')
        if size_elem is None:
            logger.warning(f"Dimensiones no encontradas en {xml_file}")
            return annotations

        img_width = int(size_elem.find('width').text)
        img_height = int(size_elem.find('height').text)

        # Extraer cada objeto anotado
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

            # Clamp a los límites de la imagen
            xmin = max(0, min(xmin, img_width))
            ymin = max(0, min(ymin, img_height))
            xmax = max(0, min(xmax, img_width))
            ymax = max(0, min(ymax, img_height))

            if xmax > xmin and ymax > ymin:
                annotations.append({
                    'class_name': class_name,
                    'xmin': xmin, 'ymin': ymin,
                    'xmax': xmax, 'ymax': ymax,
                    'img_width': img_width,
                    'img_height': img_height
                })

    except Exception as e:
        logger.error(f"Error parseando {xml_file}: {e}")

    return annotations


# =============================================================================
# 2. CONVERSIÓN A FORMATO YOLO
# =============================================================================

def convert_to_yolo_format(annotations: List[Dict], class_id: int = 0) -> List[str]:
    """
    Convierte anotaciones PASCAL VOC a formato YOLO.

    Formato YOLO: class_id x_center y_center width height
    (todos normalizados entre 0 y 1)

    Args:
        annotations: Lista de anotaciones parseadas
        class_id: ID de clase a asignar (default 0, una sola clase: logo)

    Returns:
        Lista de strings en formato YOLO
    """
    yolo_lines = []

    for ann in annotations:
        img_w = ann['img_width']
        img_h = ann['img_height']

        x_center = ((ann['xmin'] + ann['xmax']) / 2) / img_w
        y_center = ((ann['ymin'] + ann['ymax']) / 2) / img_h
        width = (ann['xmax'] - ann['xmin']) / img_w
        height = (ann['ymax'] - ann['ymin']) / img_h

        # Asegurar valores en [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return yolo_lines


# =============================================================================
# 3. PIPELINE PRINCIPAL: CONVERSIÓN DEL DATASET
# =============================================================================

def convert_logodet_dataset(
    source_dir: str,
    dest_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    copy_images: bool = True,
    seed: int = 42
) -> Dict:
    """
    Pipeline completo: convierte LogoDet-3K a formato YOLO listo
    para entrenar.

    Pasos:
        1. Busca pares imagen/XML
        2. Crea estructura de directorios YOLO
        3. Divide en train/val/test
        4. Parsea XML → YOLO para cada imagen
        5. Copia imágenes y guarda etiquetas .txt
        6. Genera data.yaml

    Args:
        source_dir: Directorio raíz de LogoDet-3K
        dest_dir: Directorio destino para el dataset YOLO
        train_ratio: Proporción de entrenamiento (default 0.7)
        val_ratio: Proporción de validación (default 0.15)
        copy_images: Si True, copia las imágenes al destino
        seed: Semilla para reproducibilidad del split

    Returns:
        Diccionario con estadísticas del dataset convertido
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    logger.info("Iniciando conversión de LogoDet-3K a YOLO...")

    dest_path = ensure_directory_exists(dest_dir)

    # Crear estructura de directorios
    for split in ['train', 'val', 'test']:
        ensure_directory_exists(dest_path / 'images' / split)
        ensure_directory_exists(dest_path / 'labels' / split)
    logger.info("Estructura de directorios YOLO creada")

    # Buscar pares imagen/XML
    logger.info("Buscando pares imagen/XML...")
    pairs = get_all_images_and_labels(source_dir)
    logger.info(f"Encontradas {len(pairs)} imágenes con anotaciones")

    if not pairs:
        logger.error("No se encontraron imágenes en el directorio origen")
        return {'success': False, 'error': 'No images found'}

    # Dividir en train/val/test
    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }

    logger.info(f"Split: Train={len(splits['train'])}, "
                f"Val={len(splits['val'])}, Test={len(splits['test'])}")

    # Procesar cada split
    stats = {'train': 0, 'val': 0, 'test': 0, 'total_annotations': 0}

    for split, split_pairs in splits.items():
        logger.info(f"Procesando {split}...")
        for img_file, xml_file in split_pairs:
            try:
                annotations = parse_xml_annotation(xml_file)
                if not annotations:
                    continue

                yolo_lines = convert_to_yolo_format(annotations)

                # Nombre único: marca_id
                img_path = Path(img_file)
                unique_name = f"{img_path.parent.name}_{img_path.stem}"

                # Guardar etiquetas
                label_file = dest_path / 'labels' / split / f"{unique_name}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                # Copiar imagen
                if copy_images:
                    dest_img = dest_path / 'images' / split / f"{unique_name}.jpg"
                    shutil.copy2(img_file, dest_img)

                stats[split] += 1
                stats['total_annotations'] += len(yolo_lines)

            except Exception as e:
                logger.warning(f"Error procesando {img_file}: {e}")
                continue

        logger.info(f"  ✓ {split}: {stats[split]} imágenes")

    # Generar data.yaml
    create_data_yaml(dest_path)

    # Resumen
    stats['success'] = True
    stats['total_images'] = stats['train'] + stats['val'] + stats['test']
    stats['dest_dir'] = str(dest_path)
    stats['data_yaml'] = str(dest_path / 'data.yaml')
    stats['train_ratio'] = train_ratio
    stats['val_ratio'] = val_ratio
    stats['test_ratio'] = test_ratio

    logger.info("=" * 50)
    logger.info("CONVERSIÓN COMPLETADA")
    logger.info(f"Total imágenes: {stats['total_images']}")
    logger.info(f"Total anotaciones: {stats['total_annotations']}")
    logger.info(f"Dataset en: {dest_path}")
    logger.info("=" * 50)

    return stats


# =============================================================================
# 4. GENERACIÓN DE data.yaml
# =============================================================================

def create_data_yaml(
    dest_path,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None
) -> bool:
    """
    Crea el archivo data.yaml requerido por YOLOv8.

    Args:
        dest_path: Ruta al directorio del dataset
        class_names: Lista de nombres de clases (default ['logo'])
        num_classes: Número de clases (default len(class_names))

    Returns:
        True si se creó exitosamente
    """
    dest_path = Path(dest_path)

    if class_names is None:
        class_names = ['logo']
    if num_classes is None:
        num_classes = len(class_names)

    # Construir sección names
    names_section = '\n'.join(
        f"  {i}: {name}" for i, name in enumerate(class_names)
    )

    yaml_content = f"""# YOLOv8 Dataset Configuration
# Generado por data_preparation.py

path: {dest_path.absolute()}
train: images/train
val: images/val
test: images/test

nc: {num_classes}

names:
{names_section}
"""

    try:
        yaml_file = dest_path / 'data.yaml'
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        logger.info(f"✓ data.yaml creado en {yaml_file}")
        return True
    except Exception as e:
        logger.error(f"Error al crear data.yaml: {e}")
        return False


# =============================================================================
# 5. FLUJO ALTERNATIVO: IMÁGENES CRUDAS (SIN ETIQUETAR)
# =============================================================================

def prepare_raw_dataset(
    raw_images_path: str,
    dest_dir: str,
    class_names: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict:
    """
    Prepara un dataset YOLO a partir de imágenes crudas sin etiquetar.

    Crea la estructura de directorios, divide las imágenes en
    train/val/test, genera archivos .txt vacíos como placeholder
    y un data.yaml. Las etiquetas deberán completarse manualmente
    con herramientas como LabelImg, CVAT o Roboflow.

    Args:
        raw_images_path: Directorio con las imágenes originales
        dest_dir: Directorio destino para el dataset YOLO
        class_names: Lista de nombres de clases (default ['logo'])
        train_ratio: Proporción de entrenamiento (default 0.7)
        val_ratio: Proporción de validación (default 0.15)
        seed: Semilla para reproducibilidad

    Returns:
        Diccionario con estadísticas del dataset preparado
    """
    logger.info("Preparando dataset desde imágenes crudas...")

    dest_path = ensure_directory_exists(dest_dir)

    # Crear estructura YOLO
    for split in ['train', 'val', 'test']:
        ensure_directory_exists(dest_path / 'images' / split)
        ensure_directory_exists(dest_path / 'labels' / split)

    # Buscar imágenes
    images = get_image_files(raw_images_path)
    total = len(images)

    if total == 0:
        logger.error(f"No se encontraron imágenes en {raw_images_path}")
        return {'success': False, 'error': 'No images found'}

    logger.info(f"Encontradas {total} imágenes")

    # Dividir
    random.seed(seed)
    shuffled = list(images)
    random.shuffle(shuffled)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': shuffled[:train_end],
        'val': shuffled[train_end:val_end],
        'test': shuffled[val_end:]
    }

    stats = {'train': 0, 'val': 0, 'test': 0}

    for split, split_images in splits.items():
        for img_path in split_images:
            img_path = Path(img_path)
            dest_img = dest_path / 'images' / split / img_path.name
            shutil.copy2(img_path, dest_img)

            # Crear archivo de etiqueta vacío (placeholder)
            label_file = dest_path / 'labels' / split / f"{img_path.stem}.txt"
            label_file.touch()

            stats[split] += 1

        logger.info(f"  ✓ {split}: {stats[split]} imágenes")

    # Generar data.yaml
    create_data_yaml(dest_path, class_names=class_names)

    stats['success'] = True
    stats['total_images'] = total
    stats['dest_dir'] = str(dest_path)
    stats['labels_empty'] = True

    logger.info("=" * 50)
    logger.info("DATASET PREPARADO (etiquetas vacías)")
    logger.info(f"Total imágenes: {total}")
    logger.info(f"Dataset en: {dest_path}")
    logger.info("⚠  Las etiquetas están vacías.")
    logger.info("   Usa LabelImg, CVAT o Roboflow para anotarlas.")
    logger.info("=" * 50)

    return stats


# =============================================================================
# 6. VALIDACIÓN DEL DATASET
# =============================================================================

def validate_dataset(dataset_path: str) -> bool:
    """
    Valida que el dataset tenga la estructura YOLO correcta.

    Comprueba que existan images/ y labels/ con subdirectorios
    train/, val/ y al menos algunos archivos.

    Args:
        dataset_path: Ruta al dataset procesado

    Returns:
        True si es válido
    """
    dataset_path = Path(dataset_path)
    is_valid = True

    for folder in ['images', 'labels']:
        folder_path = dataset_path / folder
        if not folder_path.exists():
            logger.warning(f"✗ Falta directorio: {folder}")
            is_valid = False
            continue

        for split in ['train', 'val']:
            split_path = folder_path / split
            if not split_path.exists():
                logger.warning(f"✗ Falta subdirectorio: {folder}/{split}")
                is_valid = False
            else:
                count = len(list(split_path.iterdir()))
                logger.info(f"  ✓ {folder}/{split}: {count} archivos")

    yaml_file = dataset_path / 'data.yaml'
    if not yaml_file.exists():
        logger.warning("✗ Falta data.yaml")
        is_valid = False
    else:
        logger.info("  ✓ data.yaml encontrado")

    return is_valid


def validate_conversion(dest_dir: str, num_samples: int = 3):
    """
    Muestra muestras del dataset convertido para verificación visual.

    Args:
        dest_dir: Directorio del dataset YOLO
        num_samples: Número de muestras a mostrar
    """
    dest_path = Path(dest_dir)

    logger.info("=" * 50)
    logger.info("MUESTRAS DE VERIFICACIÓN")
    logger.info("=" * 50)

    label_dir = dest_path / 'labels' / 'train'
    if not label_dir.exists():
        logger.warning("No se encontró labels/train/")
        return

    label_files = list(label_dir.glob('*.txt'))[:num_samples]

    for label_file in label_files:
        logger.info(f"\n{label_file.name}:")
        with open(label_file, 'r') as f:
            lines = [l for l in f.read().split('\n') if l]
            logger.info(f"  Anotaciones: {len(lines)}")
            for line in lines[:3]:
                logger.info(f"    {line}")

        img_file = dest_path / 'images' / 'train' / f"{label_file.stem}.jpg"
        status = "✓" if img_file.exists() else "✗"
        logger.info(f"  Imagen existe: {status}")
