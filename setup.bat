@echo off
REM Brand Logo Detection - Setup Script for Windows
REM This script sets up the development environment

echo.
echo ====================================
echo Brand Logo Detection - Setup
echo ====================================
echo.

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python found

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

REM Create necessary directories
echo.
echo Creating directories...
if not exist "data\input" mkdir data\input
if not exist "data\output" mkdir data\output
if not exist "models" mkdir models

echo [OK] Directories created

REM Done
echo.
echo ====================================
echo Setup Complete!
echo ====================================
echo.
echo To start the application, run:
echo   python run_app.py
echo.
echo Or for Streamlit directly:
echo   streamlit run streamlit_app.py
echo.
echo Access the app at: http://localhost:8501
echo.
pause
