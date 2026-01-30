#!/usr/bin/env python
"""
Quick Start Script for Brand Logo Detection Streamlit App
Verifies installation and launches the application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Found: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_requirements():
    """Check if all required packages are installed."""
    required = [
        'streamlit',
        'opencv',
        'numpy',
        'torch',
        'ultralytics',
        'yt_dlp',
        'requests',
    ]
    
    missing = []
    
    for package in required:
        try:
            if package == 'opencv':
                import cv2
            elif package == 'torch':
                import torch
            elif package == 'ultralytics':
                from ultralytics import YOLO
            elif package == 'yt_dlp':
                import yt_dlp
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    return len(missing) == 0, missing

def install_requirements():
    """Install missing requirements."""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✅ Requirements installed successfully")
        return True
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def launch_streamlit():
    """Launch Streamlit application."""
    print("\n🚀 Launching Streamlit application...")
    print("   App will open in your default browser at http://localhost:8501")
    print("   Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.call([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        return False
    
    return True

def main():
    """Main startup routine."""
    print("=" * 60)
    print("🎯 BRAND LOGO DETECTION SYSTEM - STREAMLIT")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("\n📋 System Check:")
    print("-" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Please upgrade Python to version 3.8 or higher")
        return False
    
    # Check requirements
    print("\n📦 Checking dependencies:")
    all_installed, missing = check_requirements()
    
    if not all_installed:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nWould you like to install missing packages? (y/n)")
        
        response = input().strip().lower()
        if response == 'y':
            if not install_requirements():
                print("\n❌ Failed to install requirements")
                return False
        else:
            print("\n⚠️  Some packages are missing. Installation may fail.")
            return False
    
    print("\n✅ All checks passed!")
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    dirs = ['data/input', 'data/output', 'models']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")
    
    # Launch Streamlit
    print("\n" + "=" * 60)
    return launch_streamlit()

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Startup cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
