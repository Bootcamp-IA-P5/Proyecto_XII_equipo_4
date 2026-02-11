<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Computer%20Vision-blue?style=for-the-badge&logo=yolo" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/MySQL-Database-4479A1?style=for-the-badge&logo=mysql&logoColor=white" alt="MySQL"/>
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

# 🔍 Brand Vision — Detección de Logos en Vídeos

> Sistema de detección de logos de marcas en vídeos mediante **YOLOv8** y **Computer Vision**, con interfaz web interactiva, API REST y base de datos para almacenar resultados.

---

## 📋 Índice

- [Descripción](#-descripción)
- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Tech Stack](#-tech-stack)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Uso](#-uso)
- [Entrenamiento del Modelo](#-entrenamiento-del-modelo)
- [Docker](#-docker)
- [API REST](#-api-rest)
- [Base de Datos](#-base-de-datos)
- [Equipo](#-equipo)
- [Licencia](#-licencia)

---

## 📖 Descripción

**Brand Vision** es una aplicación de Computer Vision que detecta y clasifica logos de **47 marcas comerciales** en vídeos. El sistema permite analizar vídeos subidos desde el equipo local o descargados directamente desde redes sociales (YouTube, Instagram, TikTok, Facebook, Twitter/X).

El proyecto fue desarrollado como parte del **Bootcamp de Inteligencia Artificial (P5) — Proyecto XII** en Factoría F5.

### Pipeline de 3 fases

| Fase | Descripción |
|------|-------------|
| **1. Análisis** | El modelo YOLOv8 procesa cada frame del vídeo y genera detecciones de logos |
| **2. Revisión** | El usuario revisa las detecciones en una galería interactiva con checkboxes |
| **3. Guardado** | Solo las detecciones validadas por el usuario se almacenan en base de datos |

---

## ✨ Características

- 🎥 **Análisis de vídeo** — Procesamiento frame a frame con barra de progreso en tiempo real
- 🌐 **Descarga de redes sociales** — Soporte para YouTube, Instagram, TikTok, Facebook y Twitter/X
- 🏷️ **47 marcas detectables** — Entrenado con el dataset FlickrLogos-47
- 👁️ **Revisión visual** — Galería de detecciones agrupadas por marca con selección masiva
- 📊 **Reportes y analíticas** — Timeline de apariciones, distribución de confianza, métricas por marca
- 🎨 **Bounding boxes dinámicos** — Colores por marca y nivel de confianza
- 💾 **Persistencia** — Resultados almacenados en MySQL para consulta posterior
- 🐳 **Dockerizado** — Despliegue completo con Docker Compose (backend + frontend + base de datos)

---

## 🏗️ Arquitectura

La aplicación sigue un enfoque monolítico donde el frontend Streamlit importa directamente los servicios del backend:

```
┌─────────────────────────────────────────┐     ┌──────────────────┐
│              STREAMLIT APP              │     │                  │
│                                         │     │   DATABASE       │
│   Frontend (UI)                         │     │   MySQL 8.0      │
│   • Upload vídeo                        │────▶│   Puerto 3306    │
│   • Descarga RRSS                       │     │                  │
│   • Review UI                           │     │  • videos        │
│   • Reportes                            │     │  • brands (47)   │
│                                         │     │  • detections    │
│   Backend Services (import directo)     │     │                  │
│   • DetectionPipeline (YOLOv8)          │     └──────────────────┘
│   • VideoProcessor (OpenCV, FFmpeg)     │
│   • VideoDownloader (yt-dlp)            │
│   • Visualization (bounding boxes)      │
│                                         │
└─────────────────────────────────────────┘
```

> **Nota:** El proyecto incluye también una **API REST con FastAPI** (`back/main.py`) preparada para uso futuro, que permitiría conectar otros clientes (apps móviles, otros frontends, etc.).

---

## 🛠️ Tech Stack

| Componente | Tecnología |
|------------|-----------|
| **Modelo ML** | YOLOv8 (Ultralytics) |
| **Framework ML** | PyTorch |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Base de datos** | MySQL 8.0 |
| **Visión por computador** | OpenCV |
| **Descarga de vídeos** | yt-dlp, instagrapi |
| **Procesamiento de vídeo** | FFmpeg, MoviePy, imageio |
| **Contenedores** | Docker + Docker Compose |
| **Lenguaje** | Python 3.11 |

---

## 📁 Estructura del Proyecto

```
brand-vision/
├── back/                          # Backend
│   ├── main.py                    # FastAPI entry point
│   ├── routers/                   # Endpoints API
│   │   ├── detection.py           # Endpoints de detección
│   │   ├── videos.py              # Endpoints de vídeos
│   │   └── visualization.py       # Endpoints de visualización
│   ├── services/                  # Lógica de negocio
│   │   ├── pipeline.py            # Pipeline de detección (carga YOLO, inferencia)
│   │   ├── video_processor.py     # Procesamiento frame a frame
│   │   ├── video_downloader.py    # Descarga desde RRSS
│   │   ├── video_analytics.py     # Analíticas y reportes
│   │   ├── visualization.py       # Bounding boxes y anotaciones
│   │   ├── image_loader.py        # Carga y validación de imágenes
│   │   ├── preprocessing.py       # Preprocesamiento de frames
│   │   └── config.py              # Configuración centralizada
│   ├── Dockerfile
│   └── requirements.txt
│
├── front/                         # Frontend
│   ├── streamlit_app.py           # App principal Streamlit
│   ├── Dockerfile
│   └── requirements.txt
│
├── database/                      # Base de datos
│   ├── mysql_db.py                # Conector MySQL
│   ├── sql/
│   │   └── setup_brand_vision.sql # Script de inicialización
│   └── Dockerfile
│
├── scripts/                       # Scripts de utilidades
│   ├── train_yolo.py              # Entrenamiento del modelo
│   ├── convert_flickrlogos_to_yolo.py  # Conversión del dataset
│   ├── analyze_video.py           # Análisis de vídeo por CLI
│   └── test_model.py             # Test del modelo
│
├── models/                        # Modelos entrenados
│   ├── best.pt                    # Mejor modelo (producción)
│   └── last.pt                    # Último checkpoint
│
├── data/                          # Datos de entrada/salida
├── results/                       # Métricas y reportes
├── tests/                         # Tests unitarios
│
├── docker-compose.yml             # Orquestación de contenedores
├── requirements.txt               # Dependencias globales
├── .env.example                   # Variables de entorno de ejemplo
├── streamlit_app.py               # App Streamlit (ejecución local directa)
└── LICENSE                        # MIT License
```

---

## 🚀 Instalación y Configuración

### Requisitos previos

- **Python** 3.11+
- **pip** (gestor de paquetes)
- **FFmpeg** instalado en el sistema
- **Docker** y **Docker Compose** (opcional, para despliegue containerizado)

### Opción 1: Ejecución local (recomendado para desarrollo)

```bash
# 1. Clonar el repositorio
git clone https://github.com/Bootcamp-IA-P5/Proyecto_XII_equipo_4.git
cd Proyecto_XII_equipo_4

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales de MySQL

# 5. Ejecutar la aplicación
streamlit run streamlit_app.py
```

La app estará disponible en **http://localhost:8501**

### Opción 2: Docker Compose

```bash
# 1. Clonar el repositorio
git clone https://github.com/Bootcamp-IA-P5/Proyecto_XII_equipo_4.git
cd Proyecto_XII_equipo_4

# 2. Asegurarse de que Docker Desktop esté corriendo

# 3. Construir y levantar los servicios
docker compose up --build

# 4. Acceder a la aplicación
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

Para detener los servicios:
```bash
docker compose down
```

---

## 🎮 Uso

### 1. Subir un vídeo local

1. Abre la app en **http://localhost:8501**
2. Selecciona la pestaña **"📤 Upload Video"**
3. Sube un archivo de vídeo (.mp4, .avi, .mov, .mkv)
4. Configura el umbral de confianza si lo deseas
5. Haz clic en **"🚀 Analizar"**
6. Revisa las detecciones en la galería interactiva
7. Selecciona las detecciones válidas y guarda en base de datos

### 2. Analizar vídeo de redes sociales

1. Selecciona la pestaña **"🌐 Social Media"**
2. Pega la URL del vídeo (YouTube, Instagram, TikTok, Facebook, Twitter/X)
3. La app descargará y analizará el vídeo automáticamente
4. Revisa y guarda las detecciones

### 3. Consultar historial

- Accede a la pestaña de reportes para ver análisis anteriores
- Filtra por marca, fecha o nivel de confianza

---

## 🧠 Entrenamiento del Modelo

El modelo fue entrenado con el dataset **FlickrLogos-47** que contiene imágenes de 47 marcas comerciales.

### Marcas detectables

`HP` · `Adidas` · `Aldi` · `Apple` · `Beck's` · `BMW` · `Carlsberg` · `Chimay` · `Coca-Cola` · `Corona` · `DHL` · `Erdinger` · `Esso` · `FedEx` · `Ferrari` · `Ford` · `Foster's` · `Google` · `Guinness` · `Heineken` · `Milka` · `NVIDIA` · `Paulaner` · `Pepsi` · `Ritter Sport` · `Shell` · `Singha` · `Starbucks` · `Stella Artois` · `Texaco` · `Tsingtao` · `UPS`

### Reentrenar el modelo

```bash
# Convertir el dataset FlickrLogos al formato YOLO
python scripts/convert_flickrlogos_to_yolo.py

# Entrenar el modelo
python scripts/train_yolo.py --epochs 50 --batch 16 --imgsz 640 --name mi_entrenamiento

# Los modelos se guardan en models/best.pt y models/last.pt
# Las métricas se guardan en results/
```

### Parámetros de entrenamiento

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `--model` | `yolov8n.pt` | Modelo base (nano, small, medium, etc.) |
| `--epochs` | `50` | Número de épocas |
| `--batch` | `16` | Tamaño del batch |
| `--imgsz` | `640` | Tamaño de imagen |
| `--name` | `train` | Nombre del experimento |

---

## 🐳 Docker

El proyecto incluye 3 contenedores orquestados con Docker Compose:

| Contenedor | Imagen | Puerto |
|------------|--------|--------|
| `logo-detection-backend` | Python 3.11 + FastAPI + YOLOv8 | 8000 |
| `logo-detection-frontend` | Python 3.11 + Streamlit | 8501 |
| `logo-detection-mysql` | MySQL 8.0 | 3307 → 3306 |

### Comandos útiles

```bash
# Construir imágenes
docker compose build

# Levantar servicios
docker compose up -d

# Ver logs
docker compose logs -f

# Detener servicios
docker compose down

# Limpiar todo (imágenes, contenedores, volúmenes)
docker system prune -a
```

---

## 🔌 API REST (preparada para uso futuro)

El proyecto incluye una API REST con FastAPI preparada para futuros clientes. Actualmente, el frontend Streamlit importa los servicios directamente, pero la API permite integrar otros frontends o aplicaciones móviles.

**Documentación interactiva (Swagger):** http://localhost:8000/docs

### Endpoints disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/detection/image` | Detectar logos en una imagen |
| `POST` | `/api/detection/batch` | Detectar logos en múltiples imágenes |
| `GET` | `/api/detection/classes` | Listar clases de logos detectables |
| `POST` | `/api/videos/upload` | Subir y analizar un vídeo |
| `POST` | `/api/videos/download` | Descargar vídeo por URL y analizarlo |
| `POST` | `/api/videos/extract-frames` | Extraer frames con detecciones |
| `POST` | `/api/visualization/annotate` | Imagen anotada con bounding boxes |
| `POST` | `/api/visualization/crop-detections` | Recortar cada detección individualmente |

---

## 🗄️ Base de Datos

MySQL 8.0 con 3 tablas principales:

```sql
videos       -- Información de los vídeos procesados
brands       -- Catálogo de 47 marcas (pre-cargado)
detections   -- Detecciones validadas por el usuario
```

### Diagrama ER

```
┌───────────┐     ┌───────────────┐     ┌──────────┐
│  videos   │     │  detections   │     │  brands  │
├───────────┤     ├───────────────┤     ├──────────┤
│ id (PK)   │◄───┤ video_id (FK) │     │ id (PK)  │
│ nombre    │     │ brand_id (FK) ├────▶│ nombre   │
│ duracion  │     │ segundo       │     └──────────┘
│ fecha     │     │ confianza     │
└───────────┘     │ bbox_x/y/w/h  │
                  └───────────────┘
```

---

## 👥 Equipo

Proyecto desarrollado por el **Equipo 4** del Bootcamp de Inteligencia Artificial (P5) en **Factoría F5**:

| Miembro | GitHub |
|---------|--------|
| 👩‍💻 **Maria** | — |
| 👨‍💻 **Bunty** | — |
| 👨‍💻 **Ciprian** | — |
| 👨‍💻 **Anthony** | — |

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">
  <b>Factoría F5 · Bootcamp IA P5 · Proyecto XII · Equipo 4</b><br>
  <i>Hecho con ❤️ y mucho café ☕</i>
</p>