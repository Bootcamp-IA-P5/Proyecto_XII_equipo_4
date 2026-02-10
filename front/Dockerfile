# Frontend Dockerfile for Streamlit Application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY front/requirements.txt /app/front/requirements.txt

# Install Python dependencies
# Install torch CPU-only first, then the rest of requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/front/requirements.txt

# Copy application code
COPY front/ /app/front/
COPY back/services/ /app/back/services/
COPY back/__init__.py /app/back/__init__.py
COPY database/ /app/database/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create Streamlit config directory
RUN mkdir -p /app/.streamlit

# Copy Streamlit config if exists
COPY front/.streamlit/ /app/.streamlit/

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

# Run Streamlit
CMD ["streamlit", "run", "front/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
