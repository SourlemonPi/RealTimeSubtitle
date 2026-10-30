# RealTimeSubtitle
This is simple tool for using custom ollama models as an realtime translation.

# Installation Guide

## 1. System Requirements

- OS: Windows 10 / 11
- Python: 3.11 recommended
- NVIDIA GPU supporting CUDA
- Ollama
- Internet connection for initial setup and model downloads

## 2. Install Ollama

Download and install Ollama for Windows.

https://ollama.com/download/windows

Verify the installation:

```cmd
ollama --version
```

## 3. Download a Translation Model

Download a translation model using Ollama.

Example:

```cmd
ollama pull qwen2.5:7b
```

Check installed models:

```cmd
ollama list
```

You can install other compatible models and select them from the application.

## 4. Set Up Python Environment

Open CMD in the project directory.

Create a virtual environment:

```cmd
python -m venv venv
```

Activate the environment:

```cmd
venv\Scripts\activate
```

Update pip:

```cmd
python -m pip install --upgrade pip
```

## 5. Install Dependencies

Install the required Python packages:

```cmd
pip install numpy scipy PyQt5 pyaudiowpatch ollama faster-whisper huggingface_hub
```

## 6. Download the Whisper Model

The application uses a locally stored faster-whisper model.

Download the example Small model:

```cmd
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-small', local_dir='models/whisper-small')"
```

After downloading, select the model directory in the application.

## 7. NVIDIA GPU Configuration

The current application uses CUDA acceleration for speech recognition.

Required NVIDIA libraries:

- CUDA 12 compatible cuBLAS
- cuDNN 9 for CUDA 12

Refer to the faster-whisper documentation for Windows GPU configuration:

https://github.com/SYSTRAN/faster-whisper

## 8. Run the Application

Make sure Ollama is running and the required models are installed.

Run the application:

```cmd
python main.py
```

## 9. Troubleshooting

- **Ollama connection error:** Check whether Ollama is running.
- **Model not found:** Verify the selected Whisper model directory.
- **CUDA error:** Check the NVIDIA driver, CUDA libraries and cuDNN installation.
- **Audio input error:** Check the Windows audio device and WASAPI Loopback availability.

