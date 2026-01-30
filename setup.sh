#!/bin/bash

# Brand Logo Detection - Setup Script for Linux/macOS
# This script sets up the development environment

echo ""
echo "===================================="
echo "Brand Logo Detection - Setup"
echo "===================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[OK] Python3 found"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/input
mkdir -p data/output
mkdir -p models

echo "[OK] Directories created"

# Make run_app.py executable
chmod +x run_app.py

# Done
echo ""
echo "===================================="
echo "Setup Complete!"
echo "===================================="
echo ""
echo "To start the application, run:"
echo "  python run_app.py"
echo ""
echo "Or for Streamlit directly:"
echo "  streamlit run streamlit_app.py"
echo ""
echo "Access the app at: http://localhost:8501"
echo ""
