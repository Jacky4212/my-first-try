@echo off
chcp 65001 >/dev/null
cd /d "%~dp0"

echo ============================================
echo   Guitar Transcriber
echo ============================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    echo.
    echo [2/3] Installing dependencies (first time, ~3 min)...
    venv\Scripts\pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet
    echo.
) else (
    echo [OK] venv found
    echo.
)

echo [3/3] Starting server...
echo.
echo   Backend: http://localhost:8765
echo   API docs: http://localhost:8765/docs
echo.
echo   Press Ctrl+C to stop
echo ============================================

venv\Scripts\python server.py
pause
