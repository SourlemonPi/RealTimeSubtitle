
@echo off
chcp 65001 > nul
title Voice Translator

cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo [ERROR] Please run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: Ollama 실행 여부 확인
ollama list > nul 2>&1

if errorlevel 1 (
    echo [ERROR] Ollama is not running.
    echo Please start Ollama first.
    pause
    exit /b 1
)

:: 프로그램 실행
python main.py

pause
