# RealTimeSubtitle

A real-time speech recognition and translation application for Windows, using Faster-Whisper and locally hosted Ollama models.

RealTimeSubtitle captures system audio, recognizes speech, and translates the recognized text into your selected language.

## Features

- Real-time speech recognition using Faster-Whisper
- Translation using locally hosted Ollama models
- Support for custom Ollama models
- Multiple translation display modes
- System audio capture using WASAPI Loopback
- GPU acceleration for speech recognition
- Multilingual user interface

## System Requirements

- **OS:** Windows 10 (22H2 or later) / Windows 11
- **Python:** 3.11 recommended
- **GPU:** NVIDIA GPU with CUDA support (required by the current application configuration)
- **Ollama:** Required for translation
- **Internet connection:** Required for initial installation and model downloads

## Installation

### 1. Download the Repository

Download or clone this repository:

```cmd
git clone https://github.com/SourlemonPi/RealTimeSubtitle.git
cd RealTimeSubtitle
```

You can also download the repository as a ZIP file from GitHub.

### 2. Install Ollama

Download and install Ollama for Windows:

https://ollama.com/download/windows

Verify the installation:

```cmd
ollama --version
```

### 3. Download an Ollama Translation Model

Example:

```cmd
ollama pull MedAIBase/Tencent-HY-MT1.5:1.8b
```

Check installed models:

```cmd
ollama list
```

You can install other compatible models and select an installed model in the application.

Ollama runs in the background after installation. Make sure its service is available before launching RealTimeSubtitle.

### 4. Install RealTimeSubtitle

Make sure Python 3.11 is installed and available in PATH.

Run:

```cmd
install.bat
```

The installer automatically:

- Creates a Python virtual environment
- Updates pip
- Installs the required Python packages
- Downloads the Faster-Whisper Small model into `models/whisper-small`

**Note:** The installer does not install Ollama itself, Ollama translation models, or the NVIDIA CUDA libraries required for GPU speech recognition. These must be configured separately.

### 5. Configure NVIDIA GPU Libraries

The current application uses CUDA acceleration through Faster-Whisper.

The latest Faster-Whisper / CTranslate2 versions require:

- cuBLAS for CUDA 12
- cuDNN 9 for CUDA 12

For Windows GPU configuration, refer to the official documentation:

https://github.com/SYSTRAN/faster-whisper

The required NVIDIA libraries must be accessible to the application at runtime.

### 6. Run the Application

Make sure Ollama is installed, running, and has at least one translation model downloaded.

Run:

```cmd
run.bat
```

Alternatively, you can launch the application manually:

```cmd
venv\Scripts\python.exe main.py
```

## Whisper Model

The installer downloads the following example model:

- **Model:** faster-whisper-small
- **Repository:** Systran/faster-whisper-small
- **Local directory:** `models/whisper-small`

The application uses locally stored model files. Select the downloaded model directory in the application when prompted.

## Project Structure

```text
RealTimeSubtitle/
│
├── models/
│   └── whisper-small/
│
├── main.py
├── translations.py
├── install.bat
├── run.bat
├── README.md
└── LICENSE
```

The `venv/` directory is generated automatically by `install.bat` and is not included in the repository.

## Troubleshooting

### Python is not recognized

Make sure Python 3.11 is installed and added to PATH.

### Ollama connection error

- Make sure Ollama is installed.
- Check whether Ollama is running.
- Run `ollama list` to verify the connection.

### Translation model not found

Download a model using:

```cmd
ollama pull qwen2.5:7b
```

Make sure the selected model name matches an installed Ollama model.

### Whisper model not found

- Check whether `models/whisper-small` exists.
- Make sure the model download completed successfully.
- Select the correct model directory in the application.

### CUDA error

Check the following:

- NVIDIA GPU driver
- CUDA 12 compatible cuBLAS
- cuDNN 9 for CUDA 12
- NVIDIA library DLL paths

Refer to the Faster-Whisper documentation for further details.

### Audio input error

- Check your Windows audio output device.
- Make sure WASAPI Loopback is available.
- Check whether another application is interfering with audio capture.

## License

See the [LICENSE](LICENSE) file for license information.

## Acknowledgements

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Ollama](https://ollama.com/)
- [Hugging Face](https://huggingface.co/)
