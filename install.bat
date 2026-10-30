
@echo off
chcp 65001 > nul
title Voice Translator - Installer

echo =====================================
echo       Voice Translator Installer
echo =====================================

:: Python 확인
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed.
    pause
    exit /b 1
)

:: 가상환경 생성
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: 가상환경 활성화
call venv\Scripts\activate.bat

:: pip 업데이트
python -m pip install --upgrade pip

:: 라이브러리 설치
echo Installing Python packages...
pip install numpy scipy PyQt5 pyaudiowpatch ollama faster-whisper huggingface_hub

:: Whisper 모델 다운로드
echo Downloading Whisper Small model...

if not exist models\whisper-small (
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-small', local_dir='models/whisper-small')"
)

echo.
echo Installation completed!
pause
