@echo off
REM Starts the Streamlit app. Press Ctrl+C in this window to stop it.
cd /d "%~dp0"
if not exist venv (
    echo Virtual environment not found. Run setup_windows.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
streamlit run app.py
pause
