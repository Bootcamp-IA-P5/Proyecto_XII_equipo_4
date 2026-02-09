#!/usr/bin/env python3
"""
Script de prueba para verificar visualización con colores dinámicos
"""
import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path (tests/ -> project_root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import annotate_image, save_annotated_image

# Crear imagen de prueba
img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # Blanco

# Detecciones de prueba con diferentes niveles de confianza
detections = [
    {
        'box': (50, 50, 150, 150),
        'label': 'Alta Confianza',
        'confidence': 0.95,  # Verde
        'class_id': 0
    },
    {
        'box': (200, 50, 300, 150),
        'label': 'Media Confianza',
        'confidence': 0.65,  # Amarillo
        'class_id': 1
    },
    {
        'box': (350, 50, 450, 150),
        'label': 'Baja Confianza',
        'confidence': 0.35,  # Rojo
        'class_id': 2
    },
    {
        'box': (150, 200, 250, 300),
        'label': 'Límite Alto',
        'confidence': 0.80,  # Amarillo (justo en límite)
        'class_id': 3
    },
    {
        'box': (350, 200, 450, 300),
        'label': 'Límite Bajo',
        'confidence': 0.50,  # Rojo (justo en límite)
        'class_id': 4
    }
]

# Anotar imagen
annotated = annotate_image(img, detections)

# Agregar título
cv2.putText(annotated, "Prueba de Colores por Confianza", (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
cv2.putText(annotated, "Verde: >80% | Amarillo: 50-80% | Rojo: <50%", (20, 380), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Guardar
output_path = Path("data/output/test_confidence_colors.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)
save_annotated_image(annotated, output_path)

print(f"✅ Imagen de prueba guardada en: {output_path}")
print("\nDetecciones:")
for i, det in enumerate(detections, 1):
    conf = det['confidence']
    color_name = "VERDE" if conf > 0.8 else "AMARILLO" if conf > 0.5 else "ROJO"
    print(f"  {i}. {det['label']}: {conf:.1%} → {color_name}")

print("\n🎨 Abre la imagen para verificar los colores!")
