"""
Script para convertir el dataset FlickrLogos-47 al formato YOLO.
# Ejemplo de ejecución:
# python scripts/convert_flickrlogos_to_yolo.py


Este script:
1. Lee las anotaciones del formato FlickrLogos (.gt_data.txt)
2. Las convierte al formato YOLO (class_id x_center y_center width height)
3. Organiza las imágenes y etiquetas en la estructura requerida por YOLO

Autor: Equipo 4 - Bootcamp IA
"""

import os
import shutil
import sys
from pathlib import Path

# Add project root to path to import src
sys.path.append(str(Path(__file__).parent.parent))
from src.config import INPUT_DIR, OUTPUT_DIR

from PIL import Image
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse


def load_class_mapping(class_file: str) -> Dict[str, int]:
    """
    Carga el mapeo de nombre de clase a class_id.
    
    Args:
        class_file: Ruta al archivo className2ClassID.txt
        
    Returns:
        Diccionario {nombre_clase: class_id}
    """
    class_mapping = {}
    with open(class_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    class_name, class_id = parts
                    class_mapping[class_name] = int(class_id)
    return class_mapping


def get_class_names(class_file: str) -> List[str]:
    """
    Obtiene la lista de nombres de clases ordenados por class_id.
    
    Args:
        class_file: Ruta al archivo className2ClassID.txt
        
    Returns:
        Lista de nombres de clases ordenados por índice
    """
    class_mapping = load_class_mapping(class_file)
    # Ordenar por class_id
    sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
    return [name for name, _ in sorted_classes]


def parse_gt_data_file(gt_file: str) -> List[Dict]:
    """
    Parsea un archivo .gt_data.txt de FlickrLogos.
    
    Formato: <x1> <y1> <x2> <y2> <class_id> <dummy> <mask> <difficult> <truncated>
    
    Args:
        gt_file: Ruta al archivo de groundtruth
        
    Returns:
        Lista de diccionarios con las anotaciones
    """
    annotations = []
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 9:
                annotation = {
                    'x1': int(parts[0]),
                    'y1': int(parts[1]),
                    'x2': int(parts[2]),
                    'y2': int(parts[3]),
                    'class_id': int(parts[4]),
                    'difficult': int(parts[7]),
                    'truncated': int(parts[8])
                }
                annotations.append(annotation)
    return annotations


def convert_to_yolo_format(
    annotation: Dict,
    img_width: int,
    img_height: int,
    class_filter: Optional[int] = None,
    new_class_id: Optional[int] = None
) -> Optional[str]:
    """
    Convierte una anotación al formato YOLO.
    
    Formato YOLO: <class_id> <x_center> <y_center> <width> <height>
    Todos los valores normalizados entre 0 y 1.
    
    Args:
        annotation: Diccionario con la anotación
        img_width: Ancho de la imagen en píxeles
        img_height: Alto de la imagen en píxeles
        class_filter: Si se especifica, solo procesa esta clase
        new_class_id: Si se especifica, usa este ID en lugar del original
        
    Returns:
        String en formato YOLO o None si se filtra
    """
    # Filtrar por clase si es necesario
    if class_filter is not None and annotation['class_id'] != class_filter:
        return None
    
    # Ignorar anotaciones marcadas como "difficult"
    if annotation['difficult'] == 1:
        return None
    
    x1, y1, x2, y2 = annotation['x1'], annotation['y1'], annotation['x2'], annotation['y2']
    
    # Calcular centro y dimensiones normalizadas
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Asegurar que los valores estén en el rango [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    # Usar nuevo class_id si se especifica (para entrenamiento de una sola clase)
    class_id = new_class_id if new_class_id is not None else annotation['class_id']
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def get_image_size(image_path: str) -> Tuple[int, int]:
    """
    Obtiene el tamaño de una imagen.
    
    Args:
        image_path: Ruta a la imagen
        
    Returns:
        Tupla (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


def convert_dataset(
    source_dir: str,
    dest_dir: str,
    class_file: str,
    single_class: Optional[str] = None,
    train_split: float = 0.8,
    copy_images: bool = True
):
    """
    Convierte el dataset FlickrLogos completo al formato YOLO.
    
    Args:
        source_dir: Directorio raíz de FlickrLogos (contiene train/ y test/)
        dest_dir: Directorio de destino para el dataset YOLO
        class_file: Ruta al archivo className2ClassID.txt
        single_class: Si se especifica, solo incluye esta clase
        train_split: Proporción de imágenes para entrenamiento
        copy_images: Si True, copia las imágenes al directorio destino
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Cargar mapeo de clases
    class_mapping = load_class_mapping(class_file)
    class_names = get_class_names(class_file)
    
    # Determinar filtro de clase
    class_filter = None
    if single_class:
        if single_class not in class_mapping:
            print(f"Error: Clase '{single_class}' no encontrada.")
            print(f"Clases disponibles: {list(class_mapping.keys())}")
            return
        class_filter = class_mapping[single_class]
        print(f"Filtrando solo clase: {single_class} (ID: {class_filter})")
        # Para entrenamiento de una sola clase, usamos class_id=0
        class_names = [single_class]
    
    # Crear estructura de directorios
    for split in ['train', 'val']:
        (dest_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dest_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Procesar conjuntos train y test de FlickrLogos
    all_images = []
    
    for flickr_split in ['train', 'test']:
        split_dir = source_path / flickr_split
        if not split_dir.exists():
            continue
            
        # Buscar todas las carpetas de imágenes (000000, 000001, etc.)
        for subdir in split_dir.iterdir():
            if subdir.is_dir() and subdir.name != 'no-logo':
                # Buscar archivos .gt_data.txt
                for gt_file in subdir.glob('*.gt_data.txt'):
                    # Obtener nombre base de la imagen
                    img_name = gt_file.stem.replace('.gt_data', '')
                    img_file = gt_file.parent / f"{img_name}.png"
                    
                    if img_file.exists():
                        all_images.append({
                            'gt_file': gt_file,
                            'img_file': img_file,
                            'flickr_split': flickr_split
                        })
    
    print(f"Total de imágenes encontradas: {len(all_images)}")
    
    # Filtrar imágenes que contienen la clase deseada
    filtered_images = []
    for img_info in tqdm(all_images, desc="Filtrando imágenes"):
        annotations = parse_gt_data_file(str(img_info['gt_file']))
        
        # Si hay filtro de clase, solo incluir imágenes con esa clase
        if class_filter is not None:
            has_class = any(a['class_id'] == class_filter and a['difficult'] != 1 
                          for a in annotations)
            if has_class:
                filtered_images.append(img_info)
        else:
            if annotations:  # Incluir si tiene alguna anotación
                filtered_images.append(img_info)
    
    print(f"Imágenes después de filtrar: {len(filtered_images)}")
    
    if not filtered_images:
        print("No se encontraron imágenes con las clases especificadas.")
        return
    
    # Dividir en train/val
    import random
    random.seed(42)
    random.shuffle(filtered_images)
    
    train_count = int(len(filtered_images) * train_split)
    train_images = filtered_images[:train_count]
    val_images = filtered_images[train_count:]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Procesar imágenes
    stats = {'train': 0, 'val': 0, 'annotations': 0}
    
    for split, images in [('train', train_images), ('val', val_images)]:
        for img_info in tqdm(images, desc=f"Procesando {split}"):
            gt_file = img_info['gt_file']
            img_file = img_info['img_file']
            
            try:
                img_width, img_height = get_image_size(str(img_file))
            except Exception as e:
                print(f"Error leyendo imagen {img_file}: {e}")
                continue
            
            annotations = parse_gt_data_file(str(gt_file))
            
            # Convertir anotaciones al formato YOLO
            yolo_annotations = []
            for ann in annotations:
                # Si filtramos por clase, usar class_id=0 para la única clase
                new_class_id = 0 if class_filter is not None else None
                yolo_line = convert_to_yolo_format(ann, img_width, img_height, 
                                                   class_filter, new_class_id)
                if yolo_line:
                    yolo_annotations.append(yolo_line)
            
            if not yolo_annotations:
                continue
            
            # Nombre único para la imagen
            unique_name = f"{img_file.parent.name}_{img_file.stem}"
            
            # Guardar archivo de etiquetas
            label_file = dest_path / 'labels' / split / f"{unique_name}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            # Copiar imagen si es necesario
            if copy_images:
                dest_img = dest_path / 'images' / split / f"{unique_name}.png"
                shutil.copy2(img_file, dest_img)
            
            stats[split] += 1
            stats['annotations'] += len(yolo_annotations)
    
    print(f"\n--- Resumen ---")
    print(f"Imágenes de entrenamiento: {stats['train']}")
    print(f"Imágenes de validación: {stats['val']}")
    print(f"Total de anotaciones: {stats['annotations']}")
    
    # Crear archivo data.yaml
    create_data_yaml(dest_path, class_names)
    
    print(f"\nDataset creado en: {dest_path}")
    print(f"Archivo de configuración: {dest_path / 'data.yaml'}")


def create_data_yaml(dest_path: Path, class_names: List[str]):
    """
    Crea el archivo data.yaml para YOLOv8.
    
    Args:
        dest_path: Directorio raíz del dataset YOLO
        class_names: Lista de nombres de clases
    """
    yaml_content = f"""# YOLOv8 Dataset Configuration
# Generated from FlickrLogos-47 dataset

path: {dest_path.absolute()}
train: images/train
val: images/val

# Number of classes
nc: {len(class_names)}

# Class names
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    yaml_file = dest_path / 'data.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"Archivo data.yaml creado: {yaml_file}")


def validate_conversion(dest_dir: str, num_samples: int = 5):
    """
    Valida la conversión mostrando algunas muestras.
    
    Args:
        dest_dir: Directorio del dataset YOLO
        num_samples: Número de muestras a mostrar
    """
    dest_path = Path(dest_dir)
    
    print("\n--- Validación de conversión ---")
    
    label_dir = dest_path / 'labels' / 'train'
    if not label_dir.exists():
        print("No se encontró el directorio de etiquetas")
        return
    
    label_files = list(label_dir.glob('*.txt'))[:num_samples]
    
    for label_file in label_files:
        print(f"\n{label_file.name}:")
        with open(label_file, 'r') as f:
            content = f.read()
            print(content if content else "  (vacío)")
        
        # Verificar que existe la imagen correspondiente
        img_name = label_file.stem + '.png'
        img_file = dest_path / 'images' / 'train' / img_name
        print(f"  Imagen existe: {img_file.exists()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convertir FlickrLogos a formato YOLO')
    
    # Default paths from config
    default_source = INPUT_DIR / "FlickrLogos_47"
    default_dest = OUTPUT_DIR / "yolo_dataset"
    
    parser.add_argument('--source', type=str, default=str(default_source),
                        help=f'Directorio raíz de FlickrLogos (default: {default_source})')
    parser.add_argument('--dest', type=str, default=str(default_dest),
                        help=f'Directorio destino para dataset YOLO (default: {default_dest})')
    parser.add_argument('--single-class', type=str, default=None,
                        help='Filtrar por una sola clase (ej: starbucks)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Proporción de datos para entrenamiento (default: 0.8)')
    parser.add_argument('--validate', action='store_true',
                        help='Validar conversión mostrando muestras')
    parser.add_argument('--no-copy-images', action='store_true',
                        help='No copiar imágenes (solo crear etiquetas)')
    
    args = parser.parse_args()
    
    # Ruta al archivo de clases
    class_file = os.path.join(args.source, 'className2ClassID.txt')
    
    if not os.path.exists(class_file):
        print(f"Error: No se encontró {class_file}")
        exit(1)
    
    convert_dataset(
        source_dir=args.source,
        dest_dir=args.dest,
        class_file=class_file,
        single_class=args.single_class,
        train_split=args.train_split,
        copy_images=not args.no_copy_images
    )
    
    if args.validate:
        validate_conversion(args.dest)
