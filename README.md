# 🎯 Logo Detection Pipeline

Sistema de detección de logos en imágenes usando Computer Vision y YOLO.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)](https://ultralytics.com/)

## 📋 Descripción

Este proyecto implementa un pipeline de detección de logos en imágenes utilizando:
- **OpenCV** para procesamiento de imágenes
- **YOLOv8** para detección de objetos
- **Python** como lenguaje principal

El sistema permite cargar imágenes, detectar objetos/logos, y visualizar los resultados con bounding boxes anotados.

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/Proyecto_XII_equipo_4.git
cd Proyecto_XII_equipo_4
```

### 2. Crear entorno virtual
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
Proyecto_XII_equipo_4/
├── src/
│   ├── __init__.py          # Package init
│   ├── config.py             # Configuración global
│   ├── preprocessing.py      # Preprocesamiento de imágenes
│   ├── image_loader.py       # Carga de imágenes
│   ├── visualization.py      # Visualización de detecciones
│   └── pipeline.py           # Pipeline principal
├── data/
│   ├── input/                # Imágenes de entrada
│   └── output/               # Imágenes con detecciones
├── models/                   # Modelos entrenados
├── main.py                   # Script principal (CLI)
├── requirements.txt          # Dependencias
└── README.md                 # Este archivo
```

## 💻 Uso

### Línea de Comandos

**Procesar una imagen:**
```bash
python main.py --image data/input/ejemplo.jpg
```

**Procesar un directorio:**
```bash
python main.py --directory data/input/
```

**Mostrar resultados en pantalla:**
```bash
python main.py --image data/input/ejemplo.jpg --show
```

**Ajustar umbral de confianza:**
```bash
python main.py --image data/input/ejemplo.jpg --confidence 0.7
```

### Uso Programático

```python
from src.pipeline import DetectionPipeline

# Crear pipeline
pipeline = DetectionPipeline(confidence_threshold=0.5)

# Cargar modelo
pipeline.load_model()

# Procesar imagen
result = pipeline.process_image("data/input/logo.jpg")

print(f"Detecciones encontradas: {result['detection_count']}")
for det in result['detections']:
    print(f"  - {det['label']}: {det['confidence']:.1%}")
```

### Módulos Individuales

```python
from src.image_loader import ImageLoader
from src.preprocessing import resize_image
from src.visualization import annotate_image, save_annotated_image

# Cargar imagen
loader = ImageLoader()
image = loader.load_image("path/to/image.jpg")

# Preprocesar
processed = resize_image(image, target_size=(640, 640))

# Visualizar detecciones (ejemplo)
detections = [
    {'box': (100, 100, 200, 200), 'label': 'Logo', 'confidence': 0.95, 'class_id': 0}
]
annotated = annotate_image(image, detections)
save_annotated_image(annotated, "output.jpg")
```

## ⚙️ Configuración

Edita `src/config.py` para ajustar:

- **Rutas de directorios**
- **Tamaño de imagen por defecto**
- **Umbral de confianza**
- **Colores de visualización**
- **Formatos de imagen soportados**

## 📊 Características

- ✅ Carga de imágenes individuales y por lotes
- ✅ Preprocesamiento con redimensionado y normalización
- ✅ Detección de objetos con YOLOv8
- ✅ Visualización de bounding boxes con etiquetas
- ✅ Soporte para múltiples formatos (JPG, PNG, WebP, BMP)
- ✅ CLI completa con múltiples opciones
- ✅ Código modular y documentado

## 🔮 Próximas Funcionalidades

- [ ] Procesamiento de video
- [ ] Entrenamiento de modelo custom para logos
- [ ] Base de datos para almacenar detecciones
- [ ] Frontend web con Streamlit
- [ ] API REST para integración

## 👥 Equipo

**Equipo 4** - Bootcamp IA Computer Vision

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles.