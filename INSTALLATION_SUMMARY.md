# 🎯 STREAMLIT APPLICATION - QUICK START GUIDE

## ✨ What's New

Your project now has a **complete Streamlit web application** for analyzing videos and detecting brand logos!

## 🚀 Getting Started (5 minutes)

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Run the App
```bash
python run_app.py
```

Or directly:
```bash
streamlit run streamlit_app.py
```

### Step 3: Open in Browser
- App will automatically open at `http://localhost:8501`
- Or manually visit the URL shown in terminal

## 📁 New Files Added

### Application Files
- **`streamlit_app.py`** - Main web application (600+ lines)
- **`src/video_processor.py`** - Video analysis & detection (400+ lines)
- **`src/video_downloader.py`** - Social media video download (500+ lines)
- **`src/database.py`** - SQLite database management (400+ lines)

### Configuration & Setup
- **`.streamlit/config.toml`** - Streamlit configuration
- **`run_app.py`** - Smart launcher script
- **`Dockerfile`** - Docker container setup
- **`docker-compose.yml`** - Docker Compose configuration
- **`setup.bat`** - Windows setup script
- **`setup.sh`** - Linux/macOS setup script

### Documentation
- **`STREAMLIT_README.md`** - Complete Streamlit app documentation
- **`DEPLOYMENT_GUIDE.md`** - Deployment & cloud hosting guide
- **`INSTALLATION_SUMMARY.md`** - This file

## 💡 Main Features

### 📹 Tab 1: Upload & Analyze
- Upload local videos (MP4, AVI, MOV, MKV, etc.)
- Real-time video information display
- Configurable detection settings
- Progress tracking
- Automatic cropped image extraction
- Database storage

### 🔗 Tab 2: Social Media Links
- Download from YouTube
- Download from Instagram (with login)
- Download from TikTok
- Download from Facebook
- Download from Twitter/X
- Automatic platform detection

### 📊 Tab 3: Results & Reports
- Detailed detection statistics
- Per-brand analysis
- Cropped logo samples
- Confidence metrics
- Screen time percentages
- Export as TXT or JSON

### 💾 Tab 4: Database
- View all processed videos
- Brand summary statistics
- Unique detections tracking
- Video deletion
- Export capabilities

## 🎯 How to Use

### Basic Workflow
1. **Upload video** → Tab 1
2. **Configure settings** → Sidebar
3. **Click "Analyze"** → Wait for processing
4. **View results** → Tab 3
5. **Check database** → Tab 4

### Social Media Workflow
1. **Paste URL** → Tab 2
2. **Platform auto-detected** → See platform name
3. **Click "Download"** → Video downloads
4. **Analyze** → Automatically analyzes if enabled
5. **Results** → Tab 3

## ⚙️ Configuration

### Sidebar Settings
- **Confidence Threshold**: 0.1-1.0 (lower = more detections, more false positives)
- **Frame Skip**: 1-30 (lower = more thorough, slower)

### Advanced (Edit `src/config.py`)
- Default YOLO model version
- Input/output directories
- Image normalization parameters
- Supported video formats

## 📊 Database

### What Gets Stored
- ✅ Video metadata (name, duration, resolution, FPS)
- ✅ All detections (frame, class, confidence, bounding box)
- ✅ Brand statistics (duration, percentage, confidence metrics)
- ✅ Cropped image paths
- ✅ Processing timestamps

### Database File
- Location: `data/detections.db`
- Type: SQLite3
- Automatic creation: Yes

## 🐳 Docker Deployment

### Quick Start with Docker
```bash
# Build and run
docker-compose up

# Access at http://localhost:8501

# Stop
docker-compose down
```

### Docker Commands
```bash
# Build image
docker build -t brand-detection:latest .

# Run container
docker run -p 8501:8501 brand-detection:latest

# View logs
docker-compose logs -f
```

## 📦 Dependencies Added

### Web & UI
- `streamlit>=1.28.0` - Web framework
- `streamlit-option-menu>=0.3.0` - Menu component

### Video Download
- `yt-dlp>=2023.11.0` - YouTube/Social media download
- `requests>=2.31.0` - HTTP requests
- `instagrapi>=2.0.0` - Instagram API (optional)

### Database
- `sqlalchemy>=2.0.0` - ORM (future use)

### Existing (Already in requirements)
- OpenCV, NumPy, PyTorch, YOLO, etc.

## 🔧 Troubleshooting

### "Module not found" Error
```bash
pip install -r requirements.txt
```

### Port 8501 Already in Use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Video Processing is Slow
- Increase "Frame Skip" in sidebar (5 → 10)
- Videos may take 1-5 minutes depending on length

### Instagram Download Fails
- Ensure valid Instagram credentials
- 2FA may block automatic login
- Try without 2FA first

### Database Error
- Delete `data/detections.db` to reset
- Will be recreated automatically on next run

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `STREAMLIT_README.md` | Complete Streamlit app documentation |
| `DEPLOYMENT_GUIDE.md` | Cloud deployment & production setup |
| `README.md` | Main project overview |
| `INSTALLATION_SUMMARY.md` | This quick start guide |

## 🌐 Deployment Options

### Local (Development)
```bash
python run_app.py
```

### Docker (Recommended)
```bash
docker-compose up
```

### Streamlit Cloud (Free)
1. Push to GitHub
2. Go to https://streamlit.io/cloud
3. Select repository and deploy

### Cloud Providers
- **AWS EC2**: See DEPLOYMENT_GUIDE.md
- **Heroku**: See DEPLOYMENT_GUIDE.md
- **Azure**: See DEPLOYMENT_GUIDE.md
- **Google Cloud**: See DEPLOYMENT_GUIDE.md

## ✅ Checklist

- [x] Streamlit web app created
- [x] Video upload functionality
- [x] Social media download support
- [x] Brand detection & analysis
- [x] SQLite database integration
- [x] Cropped image extraction
- [x] Reporting & export
- [x] Docker containerization
- [x] Complete documentation
- [x] Setup scripts (Windows, Linux, macOS)

## 🎓 Learning Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [YOLO Object Detection](https://docs.ultralytics.com/)
- [OpenCV Guide](https://docs.opencv.org/)
- [SQLite Tutorial](https://www.sqlite.org/docs.html)

### Project Docs
- See `STREAMLIT_README.md` for detailed feature documentation
- See `DEPLOYMENT_GUIDE.md` for production deployment
- See source code comments for implementation details

## 🎯 Next Steps

1. **Test the app locally**
   ```bash
   python run_app.py
   ```

2. **Upload a test video**
   - Or paste a YouTube link

3. **Check database results**
   - See Tab 4 for stored detections

4. **Export reports**
   - Download as TXT or JSON

5. **Deploy to cloud** (optional)
   - Follow DEPLOYMENT_GUIDE.md

## 💬 Support

For issues or questions:
1. Check documentation files
2. Review error messages
3. See Troubleshooting section
4. Check GitHub issues/discussions

## 📞 Contact

**Team**: Computer Vision Bootcamp - Team 4  
**Project**: Brand Logo Detection System  
**Version**: 1.0.0  
**Last Updated**: January 2026  

---

## 🚀 Ready to Go!

Your Streamlit application is fully set up and ready to use. Start with:

```bash
python run_app.py
```

Then open http://localhost:8501 in your browser!

Enjoy analyzing videos and detecting brand logos! 🎯
