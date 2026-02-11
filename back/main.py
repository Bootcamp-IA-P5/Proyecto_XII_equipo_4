"""
FastAPI Backend for Brand Logo Detection
Main entry point for the API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from back.routers.detection import router as detection_router
from back.routers.videos import router as videos_router
from back.routers.visualization import router as visualization_router
from back.services.config import ensure_directories

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("🚀 Starting Brand Logo Detection API...")
    ensure_directories()
    logger.info("✅ Directories initialized")
    yield
    # Shutdown
    logger.info("👋 Shutting down API...")


app = FastAPI(
    title="Brand Logo Detection API",
    description="API for detecting brand logos in images and videos using YOLOv8",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detection_router, prefix="/api/detection", tags=["Detection"])
app.include_router(videos_router, prefix="/api/videos", tags=["Videos"])
app.include_router(visualization_router, prefix="/api/visualization", tags=["Visualization"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Brand Logo Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/test")
async def test():
    return {"message": "Backend is working"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("back.main:app", host="0.0.0.0", port=8000, reload=True)
