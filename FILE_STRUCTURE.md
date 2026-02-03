# 📁 PROJECT FILE STRUCTURE

## Complete File Listing

### 🌐 Application Files (1,900+ lines of code)

```
streamlit_app.py (600+ lines)
├─ 4 main tabs
├─ Upload & analysis
├─ Social media integration
├─ Results & reporting
└─ Database management

src/
├── pipeline.py (300+ lines)
│   └─ YOLO detection engine
│
├── video_processor.py (400+ lines)
│   ├─ Video analysis
│   ├─ Frame extraction
│   ├─ Detection processing
│   ├─ Statistics calculation
│   └─ Report generation
│
├── video_downloader.py (500+ lines)
│   ├─ YouTube download
│   ├─ Instagram download
│   ├─ TikTok download
│   ├─ Facebook download
│   ├─ Twitter/X download
│   └─ Platform detection
│
├── database.py (400+ lines)
│   ├─ SQLite management
│   ├─ Video storage
│   ├─ Detection storage
│   ├─ Statistics tracking
│   └─ Query operations
│
├── config.py (99 lines)
│   ├─ Global settings
│   ├─ Path configuration
│   ├─ Model settings
│   └─ Visualization settings
│
├── image_loader.py
│   └─ Image loading utilities
│
├── preprocessing.py
│   └─ Image preprocessing
│
├── visualization.py
│   └─ Visualization utilities
│
└── __init__.py
    └─ Package initialization
```

### ⚙️ Configuration & Deployment Files

```
.streamlit/
└── config.toml (Streamlit configuration)

Dockerfile (Docker image)

docker-compose.yml (Docker Compose)

run_app.py (Smart launcher)

setup.bat (Windows setup)

setup.sh (Linux/macOS setup)

.gitignore (Git ignore rules)

requirements.txt (Python dependencies)

requirements-dev.txt (Development dependencies)
```

### 📚 Documentation Files (8 Comprehensive Guides)

```
README.md (Main project overview - updated)

INSTALLATION_SUMMARY.md (5-minute quick start)

STREAMLIT_README.md (Complete app documentation)
├─ Features guide
├─ Installation instructions
├─ Usage guide
├─ Database schema
├─ Troubleshooting
└─ Advanced features

DEPLOYMENT_GUIDE.md (Cloud & production deployment)
├─ Local development
├─ Docker deployment
├─ Cloud providers (AWS, Azure, GCP, Heroku)
├─ Production configuration
├─ SSL/TLS setup
├─ Monitoring & logging
└─ Troubleshooting

ARCHITECTURE.md (System design & architecture)
├─ System overview diagram
├─ Data flow diagram
├─ Module dependencies
├─ Technology stack
└─ File organization

API_REFERENCE.md (Developer API guide)
├─ Pipeline module API
├─ VideoProcessor API
├─ VideoDownloader API
├─ Database API
├─ Configuration guide
├─ Usage examples
└─ Error handling

IMPLEMENTATION_SUMMARY.md (This implementation checklist)
├─ Files created
├─ Features implemented
├─ Checklists
├─ Quick start options
├─ Testing guide
└─ Project metadata

main.py (Command-line interface)
```

### 📊 Data Directories

```
data/
├── input/
│   └── (Upload videos here)
│
└── output/
    ├── crops/
    │   └── (Extracted cropped images)
    │
    └── reports/
        └── (Exported reports)

models/
└── (YOLO model files)

detections.db
└── (SQLite database)
```

### 🧪 Testing

```
tests/
└── test_pipeline.py
```

### 📦 Dependencies

```
requirements.txt
├─ Core:
│  ├─ opencv-python>=4.8.0
│  ├─ numpy>=1.24.0
│  ├─ pillow>=10.0.0
│  ├─ ultralytics>=8.0.0
│  ├─ torch>=2.0.0
│  └─ torchvision>=0.15.0
│
├─ Web Interface:
│  ├─ streamlit>=1.28.0
│  └─ streamlit-option-menu>=0.3.0
│
├─ Video Download:
│  ├─ yt-dlp>=2023.11.0
│  ├─ requests>=2.31.0
│  └─ instagrapi>=2.0.0
│
├─ Database:
│  └─ sqlalchemy>=2.0.0
│
└─ Utilities:
   ├─ tqdm>=4.65.0
   └─ python-dotenv>=1.0.0

requirements-dev.txt
├─ Testing:
│  ├─ pytest>=7.0.0
│  ├─ pytest-cov>=4.0.0
│  └─ pytest-asyncio>=0.21.0
│
├─ Code Quality:
│  ├─ black>=23.0.0
│  ├─ flake8>=6.0.0
│  ├─ pylint>=2.17.0
│  └─ isort>=5.12.0
│
├─ Type Checking:
│  ├─ mypy>=1.0.0
│  ├─ types-requests
│  └─ types-pillow
│
├─ Documentation:
│  ├─ sphinx>=6.0.0
│  └─ sphinx-rtd-theme>=1.2.0
│
└─ Development:
   ├─ ipython>=8.0.0
   ├─ jupyter>=1.0.0
   ├─ notebook>=6.5.0
   └─ memory-profiler>=0.61.0
```

---

## 📊 File Statistics

### Code Files
| File | Lines | Type |
|------|-------|------|
| streamlit_app.py | 600+ | Python |
| video_processor.py | 400+ | Python |
| video_downloader.py | 500+ | Python |
| database.py | 400+ | Python |
| pipeline.py | 300+ | Python |
| Dockerfile | 30 | Docker |
| docker-compose.yml | 20 | YAML |
| run_app.py | 100+ | Python |
| setup.bat | 50+ | Batch |
| setup.sh | 50+ | Shell |
| **Total Code** | **2,450+** | - |

### Documentation Files
| File | Size | Purpose |
|------|------|---------|
| INSTALLATION_SUMMARY.md | 5 KB | Quick start |
| STREAMLIT_README.md | 15 KB | App guide |
| DEPLOYMENT_GUIDE.md | 20 KB | Deployment |
| ARCHITECTURE.md | 10 KB | Design |
| API_REFERENCE.md | 15 KB | API docs |
| IMPLEMENTATION_SUMMARY.md | 10 KB | Summary |
| README.md | 8 KB | Overview |
| **Total Docs** | **93 KB** | - |

---

## 🔄 File Dependencies

```
streamlit_app.py
    ├─→ src/pipeline.py
    ├─→ src/video_processor.py
    ├─→ src/video_downloader.py
    ├─→ src/database.py
    └─→ src/config.py

src/video_processor.py
    ├─→ src/pipeline.py
    ├─→ src/visualization.py
    └─→ src/database.py

src/pipeline.py
    ├─→ src/image_loader.py
    ├─→ src/preprocessing.py
    ├─→ src/visualization.py
    └─→ src/config.py

src/database.py
    └─→ sqlite3 (built-in)

src/video_downloader.py
    ├─→ yt-dlp
    ├─→ requests
    └─→ instagrapi (optional)

main.py
    └─→ src/pipeline.py
```

---

## 📍 Key Directories

```
Proyecto_XII_equipo_4/
├── src/                    (Core modules)
├── data/                   (Data storage)
│   ├── input/             (Input videos)
│   └── output/            (Output results)
├── models/                (YOLO models)
├── tests/                 (Unit tests)
├── .streamlit/            (Streamlit config)
└── (Root level)           (Setup & config files)
```

---

## 🚀 Getting Started Files

### For First-Time Users
1. **Read**: `INSTALLATION_SUMMARY.md`
2. **Run**: `python run_app.py`
3. **Access**: `http://localhost:8501`

### For Developers
1. **Read**: `API_REFERENCE.md`
2. **Review**: `ARCHITECTURE.md`
3. **Code**: `src/` modules
4. **Deploy**: `DEPLOYMENT_GUIDE.md`

### For DevOps/System Admins
1. **Read**: `DEPLOYMENT_GUIDE.md`
2. **Build**: `Dockerfile`
3. **Orchestrate**: `docker-compose.yml`
4. **Configure**: `.streamlit/config.toml`

---

## 🔐 Security Files

```
.gitignore              (Git ignore rules)
.env                    (Environment variables - create yourself)
.streamlit/secrets.toml (Streamlit secrets - create yourself)
```

---

## 📦 Distribution Files

```
requirements.txt        (Production dependencies)
requirements-dev.txt    (Development dependencies)
setup.bat              (Windows installer)
setup.sh               (Unix installer)
Dockerfile             (Container image)
docker-compose.yml     (Container orchestration)
```

---

## 📄 Documentation Organization

```
START HERE
    │
    ├─→ INSTALLATION_SUMMARY.md   (5 min setup)
    │
    ├─→ README.md                 (Project overview)
    │
    ├─→ STREAMLIT_README.md       (User guide)
    │
    ├─→ API_REFERENCE.md          (Developer guide)
    │
    ├─→ ARCHITECTURE.md           (System design)
    │
    ├─→ DEPLOYMENT_GUIDE.md       (Production)
    │
    └─→ IMPLEMENTATION_SUMMARY.md (Project info)
```

---

## 🎯 File Checklist

### Essential Files ✅
- [x] streamlit_app.py
- [x] src/video_processor.py
- [x] src/video_downloader.py
- [x] src/database.py
- [x] requirements.txt
- [x] run_app.py

### Configuration Files ✅
- [x] .streamlit/config.toml
- [x] Dockerfile
- [x] docker-compose.yml
- [x] .gitignore

### Setup Files ✅
- [x] setup.bat (Windows)
- [x] setup.sh (Linux/macOS)
- [x] requirements-dev.txt

### Documentation Files ✅
- [x] README.md (updated)
- [x] INSTALLATION_SUMMARY.md
- [x] STREAMLIT_README.md
- [x] DEPLOYMENT_GUIDE.md
- [x] ARCHITECTURE.md
- [x] API_REFERENCE.md
- [x] IMPLEMENTATION_SUMMARY.md

---

## 💾 File Sizes Estimate

| Category | Count | Total Size |
|----------|-------|-----------|
| Python Source | 10+ | ~2.5 MB |
| Config Files | 5+ | ~50 KB |
| Documentation | 7+ | ~100 KB |
| **Total** | **22+** | **~2.65 MB** |

(Plus dependencies when installed: ~3-5 GB)

---

## 🔄 File Update Frequency

### Frequently Updated
- `data/detections.db` (After each analysis)
- `.streamlit/secrets.toml` (When adding credentials)

### Occasionally Updated
- `requirements.txt` (When upgrading dependencies)
- `src/config.py` (When changing settings)
- `src/pipeline.py` (When improving detection)

### Rarely Updated
- `Dockerfile` (Docker config)
- `docker-compose.yml` (Deployment setup)
- Documentation files

---

## 🎯 Quick File Reference

| Need | File(s) |
|------|---------|
| Start app | `run_app.py` |
| Understand setup | `INSTALLATION_SUMMARY.md` |
| Use web app | `STREAMLIT_README.md` |
| Deploy to cloud | `DEPLOYMENT_GUIDE.md` |
| Understand code | `API_REFERENCE.md` |
| See architecture | `ARCHITECTURE.md` |
| Modify settings | `src/config.py` |
| Add dependencies | `requirements.txt` |
| Docker setup | `Dockerfile` + `docker-compose.yml` |
| Database schema | `src/database.py` |

---

**Total Files**: 22+  
**Total Lines of Code**: 2,450+  
**Total Documentation**: 93+ KB  
**Status**: ✅ Production Ready  

---

Last Updated: January 2026  
Team: Computer Vision Bootcamp - Team 4
