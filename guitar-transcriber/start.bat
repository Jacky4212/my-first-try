@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   🎸 Guitar Transcriber - 启动中...
echo ============================================
echo.

REM Check if venv exists, create if not
if not exist "venv\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境...
    python -m venv venv
    echo.
    echo [2/3] 安装依赖（首次需要几分钟）...
    venv\Scripts\pip install -r requirements.txt --quiet
    echo.
) else (
    echo [✓] 虚拟环境已存在
    echo.
)

echo [3/3] 启动服务...
echo.
echo   后端地址: http://localhost:8765
echo   API 文档: http://localhost:8765/docs
echo.
echo   按 Ctrl+C 停止服务
echo ============================================

venv\Scripts\python server.py
pause
