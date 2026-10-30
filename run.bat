
@echo off
chcp 65001 > nul
title RealTimeSubtitle

:: 프로젝트 폴더로 이동
cd /d "%~dp0"

echo =====================================
echo       RealTimeSubtitle
echo =====================================
echo.

:: 설치 여부 확인
echo [1/3] Checking installation...

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Please run install.bat first.
    pause
    exit /b 1
)

:: Ollama 설치 여부 확인
echo.
echo [2/3] Checking Ollama...

where ollama > nul 2>&1

if errorlevel 1 (
    echo [ERROR] Ollama is not installed or not added to PATH.
    echo Please install Ollama first.
    pause
    exit /b 1
)

:: Ollama 서버 및 모델 확인
ollama list

if errorlevel 1 (
    echo.
    echo [ERROR] Cannot connect to Ollama.
    echo Please start Ollama and try again.
    pause
    exit /b 1
)

:: 프로그램 실행
echo.
echo [3/3] Starting RealTimeSubtitle...
echo.

venv\Scripts\python.exe main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Program terminated with an error.
)

pause
