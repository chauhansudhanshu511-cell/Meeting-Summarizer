@echo off
REM ==============================================================
REM  One-click setup for AI-Based Smart Meeting Summarization
REM  Double-click this file (or run it in Command Prompt).
REM ==============================================================
cd /d "%~dp0"

echo [1/5] Checking Python...
python --version
if errorlevel 1 (
    echo Python was not found. Install Python 3.11 from https://www.python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

echo [2/5] Creating virtual environment "venv"...
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [3/5] Installing Python packages (this can take 5-15 minutes)...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Package installation failed. See README - Troubleshooting.
    pause
    exit /b 1
)

echo [4/5] Installing spaCy English model...
python -m spacy download en_core_web_sm

echo [5/5] Downloading AI models (one time, needs internet)...
python download_models.py

echo.
echo Setup complete! Double-click run_app.bat to start the application.
pause
