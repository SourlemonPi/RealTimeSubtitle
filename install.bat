
@echo off
chcp 65001 > nul
title RealTimeSubtitle - Installer

:: 프로젝트 폴더로 이동
cd /d "%~dp0"

echo =====================================
echo       RealTimeSubtitle Installer
echo =====================================
echo.

:: Python 확인
echo [1/5] Checking Python...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to PATH.
    pause
    exit /b 1
)

:: 가상환경 생성
echo.
echo [2/5] Creating virtual environment...

if not exist "venv\Scripts\python.exe" (
    python -m venv venv
    if errorlevel 1 goto ERROR
)

:: pip 업데이트
echo.
echo [3/5] Updating pip...

venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto ERROR

:: 라이브러리 설치
echo.
echo [4/5] Installing Python packages...

venv\Scripts\python.exe -m pip install numpy scipy PyQt5 pyaudiowpatch ollama faster-whisper huggingface_hub
if errorlevel 1 goto ERROR

:: Whisper 모델 다운로드
echo.
echo [5/5] Downloading Whisper Small model...

if not exist "models" mkdir models

if not exist "models\whisper-small\model.bin" (
    venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-small', local_dir='models/whisper-small')"
    if errorlevel 1 goto ERROR
) else (
    echo Whisper model already exists. Skipping download.
)

echo.
echo =====================================
echo       Installation completed!
echo =====================================
echo.
echo You can now run run.bat.
pause
exit /b 0

:ERROR
echo.
echo =====================================
echo       Installation failed!
echo =====================================
echo Please check the error message above.
pause
exit /b 1
