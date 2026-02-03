# Deployment & Running Guide

## Local Development

### Quick Start (Recommended for Development)

```bash
# Navigate to project directory
cd Proyecto_XII_equipo_4

# Run the app directly
python run_app.py
```

This script will:
- Verify Python version
- Check all dependencies
- Install missing packages automatically
- Create necessary directories
- Launch the Streamlit app

### Manual Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run streamlit_app.py
```

**Access**: Open browser to `http://localhost:8501`

## Docker Deployment

### Prerequisites
- Docker installed
- Docker Compose installed
- 4GB+ disk space

### Build & Run with Docker

```bash
# Build Docker image
docker build -t brand-detection:latest .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/data/input:/app/data/input \
  -v $(pwd)/data/output:/app/data/output \
  -v $(pwd)/models:/app/models \
  brand-detection:latest
```

**Access**: `http://localhost:8501`

### Docker Compose (Recommended)

```bash
# Start application
docker-compose up -d

# View logs
docker-compose logs -f brand-detection

# Stop application
docker-compose down

# Rebuild after changes
docker-compose up --build
```

### Docker Compose with GPU Support

Edit `docker-compose.yml`:

```yaml
services:
  brand-detection:
    # ... other config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Then run:
```bash
docker-compose up -d
```

## Cloud Deployment

### Streamlit Cloud (Free)

1. **Push to GitHub**
```bash
git add .
git commit -m "Add Streamlit app"
git push origin dev
```

2. **Deploy on Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub
   - Create new app
   - Select repository: `Bootcamp-IA-P5/Proyecto_XII_equipo_4`
   - Set main file path: `streamlit_app.py`
   - Click Deploy

3. **Access**: App will be available at `https://your-username-project-name.streamlit.app`

### Heroku Deployment

1. **Create Procfile**
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **Deploy**
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### AWS EC2 Deployment

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance: t3.medium or larger
   - Security Group: Open port 8501

2. **Setup**
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv

# Clone repository
git clone https://github.com/Bootcamp-IA-P5/Proyecto_XII_equipo_4.git
cd Proyecto_XII_equipo_4

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with nohup (background process)
nohup streamlit run streamlit_app.py > app.log 2>&1 &
```

3. **Access**: `http://your-instance-ip:8501`

### Azure Container Instances

```bash
# Create registry
az acr create --resource-group myResourceGroup \
  --name myRegistry --sku Basic

# Build image
az acr build --registry myRegistry \
  --image brand-detection:latest .

# Deploy container
az container create --resource-group myResourceGroup \
  --name brand-detection \
  --image myRegistry.azurecr.io/brand-detection:latest \
  --ports 8501 \
  --cpu 2 --memory 4 \
  --registry-login-server myRegistry.azurecr.io
```

### Google Cloud Run

```bash
# Build image
gcloud builds submit --tag gcr.io/your-project/brand-detection

# Deploy
gcloud run deploy brand-detection \
  --image gcr.io/your-project/brand-detection \
  --platform managed \
  --region us-central1 \
  --port 8501 \
  --memory 4Gi \
  --cpu 2
```

## Production Configuration

### Streamlit Config (.streamlit/config.toml)

```toml
[server]
port = 8501
headless = true
runOnSave = true
maxUploadSize = 200

[client]
showErrorDetails = false

[logger]
level = "info"

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 86400;
    }
}
```

### SSL/TLS (Let's Encrypt)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot certonly --nginx -d your-domain.com

# Update Nginx config to use SSL
sudo nano /etc/nginx/sites-available/default

# Restart Nginx
sudo systemctl restart nginx
```

## Environment Variables

Create `.env` file:

```env
# Model settings
DEFAULT_CONFIDENCE=0.5
DEFAULT_MODEL=yolov8n.pt
IOU_THRESHOLD=0.45

# Database
DATABASE_PATH=./data/detections.db

# Instagram (optional)
INSTAGRAM_USERNAME=your_username
INSTAGRAM_PASSWORD=your_password

# API Keys (for future integrations)
API_KEY=your_api_key
```

Load in app:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Performance Optimization

### For Video Processing
- Use GPU: Install CUDA + cuDNN
- Increase frame skip for faster processing
- Lower video resolution before upload

### For Web Server
- Use gunicorn with multiple workers
- Enable caching with Redis
- Use CDN for static files

### Memory Management
- Set memory limits in docker-compose
- Monitor with `docker stats`
- Clean old database records regularly

## Monitoring & Logging

### Streamlit Logs
```bash
# Local logs
cat ~/.streamlit/logs/2024-*.log

# Docker logs
docker-compose logs -f brand-detection
```

### Application Monitoring
```bash
# CPU/Memory usage
docker stats brand-detection

# Check port
netstat -tlnp | grep 8501
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run streamlit_app.py --server.port 8502
```

### Out of Memory
- Reduce batch size
- Increase frame skip
- Stop other applications
- Use GPU if available

### Video Download Fails
```bash
# Update yt-dlp
pip install --upgrade yt-dlp

# Clear cache
rm -rf ~/.cache/yt-dlp
```

### CORS Issues in Cloud
- Configure CORS headers in app
- Use reverse proxy with proper headers
- Check cross-origin settings

## Maintenance

### Regular Tasks
- Clear old videos from `data/output`
- Archive database periodically
- Update dependencies monthly
- Review logs for errors

### Backup Strategy
```bash
# Backup database
cp data/detections.db backups/detections_$(date +%Y%m%d).db

# Backup cropped images
tar -czf backups/crops_$(date +%Y%m%d).tar.gz data/output/crops/
```

### Automatic Cleanup
```bash
# Remove files older than 30 days
find data/output -type f -mtime +30 -delete
```

---

**Last Updated**: January 2026  
**Version**: 1.0.0
