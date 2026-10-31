
try:
    from scipy.signal import resample_poly
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import faulthandler

from translations import get_text

crash_log_file = open("crash.log", "w", encoding="utf-8")
faulthandler.enable(crash_log_file)

import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------------------------------------------
# DLL 경로 등록
# ------------------------------------------------------------

import sys
import site

_DLL_HANDLES = []


def _add_nvidia_dlls():
    roots = set(
        site.getsitepackages()
        + [site.getusersitepackages()]
        + sys.path
    )

    for sp in roots:
        for sub in ("cublas", "cudnn"):
            p = os.path.join(sp, "nvidia", sub, "bin")

            if os.path.isdir(p):
                _DLL_HANDLES.append(os.add_dll_directory(p))
                os.environ["PATH"] = (
                    p + os.pathsep + os.environ.get("PATH", "")
                )
                print("Added DLL path:", p)


_add_nvidia_dlls()


# ------------------------------------------------------------
# 라이브러리
# ------------------------------------------------------------

from faster_whisper import WhisperModel

import asyncio
import logging
import traceback
import numpy as np
import pyaudiowpatch as pyaudio
import ollama
import winsound
import threading

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QSlider,
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QLineEdit,
    QFileDialog,
    QProgressBar
)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QFont, QMouseEvent

from ollama import AsyncClient


# ------------------------------------------------------------
# 0. 디버깅 로그 설정
# ------------------------------------------------------------

logging.basicConfig(
    filename="subtitle_debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)


# ------------------------------------------------------------
# 1. 오디오 입력 테스트 쓰레드
# ------------------------------------------------------------

class AudioTestWorker(QThread):

    volume_signal = pyqtSignal(int)

    def __init__(self, device_index):
        super().__init__()

        self.device_index = device_index
        self.is_running = True
        self.stream = None
        self.p = None

    def run(self):

        self.p = pyaudio.PyAudio()
        CHUNK = 4096

        try:
            dev_info = self.p.get_device_info_by_index(
                self.device_index
            )

            native_rate = int(
                dev_info.get("defaultSampleRate", 48000)
            )

            channels = min(
                2,
                int(dev_info.get("maxInputChannels", 1))
            )

            if channels < 1:
                raise RuntimeError(
                    "The selected device has no input channels."
                )

            logging.info(
                f"Volume test device: {dev_info['name']}, "
                f"rate={native_rate}, channels={channels}"
            )

            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=native_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK
            )

            while self.is_running:

                data = self.stream.read(
                    CHUNK,
                    exception_on_overflow=False
                )

                if not self.is_running:
                    break

                audio_data = np.frombuffer(
                    data,
                    dtype=np.int16
                )

                if audio_data.size == 0:
                    continue

                peak = np.max(
                    np.abs(audio_data.astype(np.int32))
                )

                vol_percent = min(
                    100,
                    int((peak / 32768.0) * 100 * 5)
                )

                self.volume_signal.emit(vol_percent)

        except Exception as e:

            if self.is_running:
                logging.exception(
                    f"Audio Test Error: {e}"
                )

        finally:

            if self.stream is not None:

                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                except Exception:
                    pass

                try:
                    self.stream.close()
                except Exception:
                    pass

                self.stream = None

            if self.p is not None:
                self.p.terminate()
                self.p = None

            self.volume_signal.emit(0)

            logging.info(
                "Audio test thread terminated"
            )

    def stop(self):

        self.is_running = False

        if self.stream is not None:

            try:
                self.stream.abort_stream()

            except Exception as e:
                logging.warning(
                    f"Failed to abort audio stream: {e}"
                )


# ------------------------------------------------------------
# 2. Ollama 비동기 번역 클래스
# ------------------------------------------------------------

class OllamaTranslator:

    def __init__(
        self,
        model_name="qwen2.5:7b",
        host="http://localhost:11434"
    ):

        self.client = AsyncClient(host=host)
        self.model_name = model_name

    async def translate_stream(
        self,
        text,
        src_lang,
        target_lang,
        callback,
        display_mode="long_phrase"
    ):

        if not text.strip():
            return

        # HY-MT1.5가 이해하기 쉬운 정식 언어 이름

        target_names_zh = {
            "ko": "韩语",
            "en": "英语",
            "ja": "日语",
            "zh": "中文"
        }

        target_names_en = {
            "ko": "Korean",
            "en": "English",
            "ja": "Japanese",
            "zh": "Chinese"
        }

        # 번역 프롬프트

        if src_lang == "zh":

            target_name = target_names_zh.get(
                target_lang,
                "韩语"
            )

            prompt = (
                f"将以下文本翻译为{target_name}，"
                f"注意只需要输出翻译后的结果，不要额外解释：\n\n"
                f"{text}"
            )

        else:

            target_name = target_names_en.get(
                target_lang,
                "Korean"
            )

            prompt = (
                f"Translate the following segment into {target_name}, "
                f"without additional explanation.\n\n"
                f"{text}"
            )

        full_response = ""
        last_display_length = 0

        # 표시 단위별 갱신 기준

        display_limits = {
            "word": 1,
            "short_phrase": 8,
            "long_phrase": 20,
            "sentence": None
        }

        display_limit = display_limits.get(
            display_mode,
            20
        )

        try:

            async for chunk in await self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=True,
                options={
                    "temperature": 0.7,
                    "top_p": 0.6,
                    "top_k": 20,
                    "repeat_penalty": 1.05
                }
            ):

                token = chunk.get(
                    "message", {}
                ).get("content", "")

                if not token:
                    continue

                full_response += token

                new_text = full_response[
                    last_display_length:
                ]

                # 문장 단위

                if display_mode == "sentence":

                    if any(
                        mark in new_text
                        for mark in [
                            "。",
                            "！",
                            "？",
                            ".",
                            "!",
                            "?",
                            "\n"
                        ]
                    ):

                        callback(full_response)

                        last_display_length = len(
                            full_response
                        )

                # 단어 / 구절 단위

                else:

                    if len(new_text) >= display_limit:

                        callback(full_response)

                        last_display_length = len(
                            full_response
                        )

            # 마지막 남은 번역 내용 표시

            if (
                full_response
                and last_display_length < len(full_response)
            ):

                callback(full_response)

            logging.info(
                f"Translation completed | "
                f"Source: {text} | "
                f"Translation: {full_response}"
            )

        except Exception as e:

            err_msg = f"Ollama Error: {e}"

            logging.exception(err_msg)

            callback(err_msg)


# ------------------------------------------------------------
# 3. STT 및 번역 처리 쓰레드
# ------------------------------------------------------------

class SpeechTranslateWorker(QThread):

    subtitle_signal = pyqtSignal(str, str)

    @staticmethod
    def to_16k(x, in_rate):

        if in_rate == 16000:
            return x

        if HAS_SCIPY:

            from math import gcd

            g = gcd(16000, in_rate)

            return resample_poly(
                x,
                16000 // g,
                in_rate // g
            ).astype(np.float32)

        n_out = int(
            len(x) * 16000 / in_rate
        )

        return np.interp(
            np.linspace(0, len(x) - 1, n_out),
            np.arange(len(x)),
            x
        ).astype(np.float32)

    def __init__(
        self,
        device_index,
        src_lang,
        target_lang,
        ollama_model,
        stt_model_path,
        display_mode,
        ui_language
    ):

        super().__init__()

        self.device_index = device_index
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.ollama_model = ollama_model
        self.stt_model_path = stt_model_path
        self.display_mode = display_mode
        self.ui_language = ui_language

        self.is_running = True
        self.stt_model = None

        self.pause_event = threading.Event()

    def run(self):

        # STT 모델 로딩 메시지

        self.subtitle_signal.emit(
            f"⏳ {get_text(self.ui_language, 'loading')}",
            ""
        )

        try:

            model_path = os.path.abspath(
                self.stt_model_path
            )

            logging.info(
                f"STT model path: {model_path}"
            )

            if not os.path.isdir(model_path):

                raise FileNotFoundError(
                    f"Model folder does not exist: {model_path}"
                )

            required_files = [
                "config.json",
                "model.bin",
                "tokenizer.json"
            ]

            for filename in required_files:

                filepath = os.path.join(
                    model_path,
                    filename
                )

                if not os.path.isfile(filepath):

                    raise FileNotFoundError(
                        f"Model file not found: {filepath}"
                    )

            logging.info(
                "Required model files verified. "
                "Loading into GPU memory..."
            )

            # GPU CUDA 가속

            self.stt_model = WhisperModel(
                model_path,
                device="cuda",
                compute_type="float16",
                local_files_only=True
            )

            logging.info(
                "STT model loaded successfully on GPU!"
            )

        except Exception as e:

            logging.exception(
                "Error loading STT model"
            )

            self.subtitle_signal.emit(
                "❌ Failed to load STT model",
                str(e)
            )

            return

        # STT 준비 완료

        self.subtitle_signal.emit(
            "✅ " + get_text(self.ui_language, "ready"),
            ""
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        translator = OllamaTranslator(
            model_name=self.ollama_model
        )

        p = pyaudio.PyAudio()

        RATE = 16000
        CHUNK = 4096

        stream = None

        in_rate = RATE
        in_ch = 1

        try:

            dev_info = p.get_device_info_by_index(
                self.device_index
            )

            native_rate = int(
                dev_info.get(
                    "defaultSampleRate",
                    48000
                )
            )

            max_ch = int(
                dev_info.get(
                    "maxInputChannels",
                    2
                )
            )

            last_err = None

            # 장치 기본 레이트를 먼저 시도

            for rate in dict.fromkeys([
                native_rate,
                48000,
                44100,
                16000
            ]):

                for ch in dict.fromkeys([
                    min(2, max_ch),
                    1
                ]):

                    try:

                        stream = p.open(
                            format=pyaudio.paInt16,
                            channels=ch,
                            rate=rate,
                            input=True,
                            input_device_index=self.device_index,
                            frames_per_buffer=CHUNK
                        )

                        in_rate = rate
                        in_ch = ch

                        break

                    except Exception as e:

                        last_err = e

                        logging.warning(
                            f"Failed to open audio stream "
                            f"rate={rate} ch={ch}: {e}"
                        )

                if stream:
                    break

            if stream is None:
                raise last_err

            logging.info(
                f"Audio stream opened: "
                f"rate={in_rate}, channels={in_ch}"
            )

            buffer = np.zeros(
                0,
                dtype=np.float32
            )

            target_length = int(
                RATE * 1.8
            )

            STEP = CHUNK * 4

            while self.is_running:

                try:

                    data = stream.read(
                        CHUNK,
                        exception_on_overflow=False
                    )

                except IOError:
                    continue

                # 일시정지 상태

                if self.pause_event.is_set():

                    buffer = np.zeros(
                        0,
                        dtype=np.float32
                    )

                    continue

                audio = np.frombuffer(
                    data,
                    dtype=np.int16
                ).astype(np.float32) / 32768.0

                # 스테레오 -> 모노

                if in_ch > 1:

                    audio = audio.reshape(
                        -1,
                        in_ch
                    ).mean(axis=1)

                # 16kHz 변환

                audio = self.to_16k(
                    audio,
                    in_rate
                )

                buffer = np.concatenate([
                    buffer,
                    audio
                ])

                logging.info(
                    f"Input peak={np.abs(audio).max():.4f}"
                )

                if len(buffer) >= target_length:

                    audio_np = buffer[:target_length]

                    buffer = buffer[STEP:]

                    # Whisper 음성 인식

                    segments, _ = self.stt_model.transcribe(
                        audio_np,
                        language=(
                            self.src_lang
                            if self.src_lang != "auto"
                            else None
                        ),
                        beam_size=3,
                        vad_filter=True
                    )

                    text = "".join(
                        s.text for s in segments
                    ).strip()

                    logging.info(
                        f"STT result: '{text}'"
                    )

                    if text:

                        def on_stream_token(
                            partial_translation,
                            t=text
                        ):

                            self.subtitle_signal.emit(
                                t,
                                partial_translation
                            )

                        loop.run_until_complete(
                            translator.translate_stream(
                                text,
                                self.src_lang,
                                self.target_lang,
                                on_stream_token,
                                self.display_mode
                            )
                        )

        except Exception as e:

            logging.exception(
                "Audio/runtime error"
            )

            self.subtitle_signal.emit(
                "❌ Audio Device Error",
                str(e)
            )

        finally:

            if stream is not None:

                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

            p.terminate()

            loop.close()

    def toggle_pause(self):

        if self.pause_event.is_set():

            self.pause_event.clear()

            logging.info(
                "Translation resumed"
            )

            return False

        else:

            self.pause_event.set()

            logging.info(
                "Translation paused"
            )

            return True

    def stop(self):

        self.is_running = False


# ------------------------------------------------------------
# 4. 투명 자막 오버레이 창
# ------------------------------------------------------------

class SubtitleOverlay(QWidget):

    def __init__(self):

        super().__init__()

        self.oldPos = None

        self.initUI()

    def initUI(self):

        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )

        self.setAttribute(
            Qt.WA_TranslucentBackground
        )

        main_layout = QVBoxLayout()

        main_layout.setContentsMargins(
            10, 10, 10, 10
        )

        self.label_src = QLabel(
            "Waiting for subtitles...",
            self
        )

        self.label_trans = QLabel(
            "",
            self
        )

        for lbl in [
            self.label_src,
            self.label_trans
        ]:

            lbl.setAlignment(
                Qt.AlignCenter
            )

            lbl.setWordWrap(True)

            lbl.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Expanding
            )

            lbl.setStyleSheet(
                "color: white;"
                "background-color: rgba(0, 0, 0, 150);"
                "border-radius: 8px;"
                "padding: 10px;"
            )

        self.label_src.setFont(
            QFont("Arial", 14, QFont.Bold)
        )

        self.label_trans.setFont(
            QFont("Arial", 18, QFont.Bold)
        )

        self.label_trans.setStyleSheet(
            "color: #FFD700;"
            "background-color: rgba(0, 0, 0, 180);"
            "border-radius: 8px;"
            "padding: 10px;"
        )

        main_layout.addWidget(
            self.label_src
        )

        main_layout.addWidget(
            self.label_trans
        )

        # 크기 조절 핸들

        grip_layout = QHBoxLayout()

        grip_layout.setContentsMargins(
            0, 0, 0, 0
        )

        grip_layout.addStretch()

        size_grip = QSizeGrip(self)

        size_grip.setFixedSize(
            15,
            15
        )

        size_grip.setStyleSheet(
            "background-color: rgba(255, 255, 255, 100);"
            "border-radius: 3px;"
        )

        grip_layout.addWidget(
            size_grip
        )

        main_layout.addLayout(
            grip_layout
        )

        self.setLayout(
            main_layout
        )

        self.resize(
            800,
            180
        )

        self.setMinimumSize(
            400,
            100
        )

    def update_text(self, src_text, trans_text):

        self.label_src.setText(
            src_text
        )

        self.label_trans.setText(
            trans_text
        )

    def set_font_size(self, size):

        self.label_src.setFont(
            QFont(
                "Arial",
                int(size * 0.8),
                QFont.Bold
            )
        )

        self.label_trans.setFont(
            QFont(
                "Arial",
                size,
                QFont.Bold
            )
        )

    def set_opacity_val(self, opacity):

        self.setWindowOpacity(
            opacity
        )

    # 자막 창 드래그

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):

        if self.oldPos is not None:

            delta = (
                event.globalPos() - self.oldPos
            )

            self.move(
                self.x() + delta.x(),
                self.y() + delta.y()
            )

            self.oldPos = event.globalPos()

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.oldPos = None

    def set_overlay_width(self, width):

        self.resize(
            int(width),
            self.height()
        )

    def set_overlay_height(self, height):

        self.resize(
            self.width(),
            int(height)
        )

    def reset_overlay_position(self):

        self.resize(
            800,
            180
        )

        self.move(
            100,
            100
        )


# ------------------------------------------------------------
# 5. 설정 메인 패널 UI
# ------------------------------------------------------------

# ------------------------------------------------------------
# 5. 설정 메인 패널 UI
# ------------------------------------------------------------

class ControlPanel(QWidget):

    # 설정 파일을 저장할 경로
    CONFIG_FILE = "settings.json"

    def load_settings(self):
        """저장된 설정 불러와서 UI에 반영하기"""
        if not os.path.exists(self.CONFIG_FILE):
            return
        try:
            with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # 1. STT 모델 경로 복원
            if "stt_path" in config:
                self.stt_path_edit.setText(config["stt_path"])
            
            # 2. 오디오 장치 복원
            if "device_name" in config:
                for i in range(self.device_combo.count()):
                    if config["device_name"] in self.device_combo.itemText(i):
                        self.device_combo.setCurrentIndex(i)
                        break
            
            # 3. Ollama 모델 복원
            if "ollama_model" in config:
                idx = self.model_combo.findText(config["ollama_model"])
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)

            # 4. 표시 단위 복원
            if "display_mode" in config:
                idx = self.display_mode_combo.findData(config["display_mode"])
                if idx >= 0:
                    self.display_mode_combo.setCurrentIndex(idx)
            
            # 5. 언어 설정 복원
            if "src_lang" in config:
                idx = self.src_lang_combo.findText(config["src_lang"])
                if idx >= 0: self.src_lang_combo.setCurrentIndex(idx)
            if "target_lang" in config:
                idx = self.target_lang_combo.findText(config["target_lang"])
                if idx >= 0: self.target_lang_combo.setCurrentIndex(idx)

            # 6. 인터페이스 언어 복원
            if "ui_language" in config:
                for name, code in self.ui_languages.items():
                    if code == config["ui_language"]:
                        idx = self.ui_language_combo.findText(name)
                        if idx >= 0:
                            self.ui_language_combo.setCurrentIndex(idx)
                        break
            
            # 7. 슬라이더 (투명도, 크기, 너비, 높이) 복원
            if "opacity" in config:
                self.opacity_slider.setValue(config["opacity"])
                self.overlay.set_opacity_val(config["opacity"] / 100.0)
            if "font_size" in config:
                self.size_slider.setValue(config["font_size"])
                self.overlay.set_font_size(config["font_size"])
            if "width" in config:
                self.width_slider.setValue(config["width"])
                self.overlay.set_overlay_width(config["width"])
            if "height" in config:
                self.height_slider.setValue(config["height"])
                self.overlay.set_overlay_height(config["height"])
                
        except Exception as e:
            logging.exception(f"설정 불러오기 실패: {e}")

    def closeEvent(self, event):
        """프로그램이 꺼질 때 현재 설정 값을 JSON으로 저장 및 스레드 정리"""
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            self.worker.stop()
            self.worker.wait()

        if (
            self.test_worker is not None
            and self.test_worker.isRunning()
        ):
            self.test_worker.stop()
            self.test_worker.wait()

        try:
            config = {
                "stt_path": self.stt_path_edit.text(),
                "device_name": self.device_combo.currentText(),
                "ollama_model": self.model_combo.currentText(),
                "display_mode": self.display_mode_combo.currentData(),
                "src_lang": self.src_lang_combo.currentText(),
                "target_lang": self.target_lang_combo.currentText(),
                "ui_language": self.ui_language,
                "opacity": self.opacity_slider.value(),
                "font_size": self.size_slider.value(),
                "width": self.width_slider.value(),
                "height": self.height_slider.value()
            }
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logging.exception(f"설정 저장 실패: {e}")
        
        self.overlay.close()
        event.accept()

    def __init__(self, overlay):

        super().__init__()

        self.overlay = overlay

        self.worker = None
        self.test_worker = None

        # 기본 인터페이스 언어

        self.ui_language = "ko"

        self.initUI()

    # --------------------------------------------------------
    # 인터페이스 언어 변경
    # --------------------------------------------------------

    def change_ui_language(self):

        self.ui_language = (
            self.ui_language_combo.currentData()
        )

        self.retranslate_ui()

    # --------------------------------------------------------
    # UI 문구 갱신
    # --------------------------------------------------------

    def retranslate_ui(self):

        t = lambda key: get_text(
            self.ui_language,
            key
        )

        # 창 제목

        self.setWindowTitle(
            t("window_title")
        )

        # 라벨

        self.lbl_language.setText(
            t("language")
        )

        self.lbl_audio.setText(
            t("audio_source")
        )

        self.lbl_stt.setText(
            t("stt_folder")
        )

        self.lbl_model.setText(
            t("translation_model")
        )

        self.lbl_display.setText(
            t("display_unit")
        )

        self.lbl_src_lang.setText(
            t("recognition_language")
        )

        self.lbl_target_lang.setText(
            t("target_language")
        )

        self.lbl_opacity.setText(
            t("opacity")
        )

        self.lbl_width.setText(
            t("width")
        )

        self.lbl_height.setText(
            t("height")
        )

        self.lbl_font_size.setText(
            t("font_size")
        )

        # 폴더 버튼 및 입력창

        self.btn_browse.setText(
            t("browse")
        )

        self.stt_path_edit.setPlaceholderText(
            t("stt_placeholder")
        )

        # 오디오 테스트 버튼

        if self.btn_test.isChecked():

            self.btn_test.setText(
                t("audio_test_stop")
            )

        else:

            self.btn_test.setText(
                t("audio_test")
            )

        # 시작 / 중지 버튼

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):

            self.btn_start.setText(
                t("stop")
            )

            if self.worker.pause_event.is_set():

                self.btn_pause.setText(
                    t("resume")
                )

            else:

                self.btn_pause.setText(
                    t("pause")
                )

        else:

            self.btn_start.setText(
                t("start")
            )

            self.btn_pause.setText(
                t("pause")
            )

        # 번역 표시 단위

        current_mode = (
            self.display_mode_combo.currentData()
        )

        if current_mode is None:
            current_mode = "long_phrase"

        self.display_mode_combo.clear()

        display_modes = [
            "word",
            "short_phrase",
            "long_phrase",
            "sentence"
        ]

        for mode in display_modes:

            self.display_mode_combo.addItem(
                t(mode),
                mode
            )

        index = self.display_mode_combo.findData(
            current_mode
        )

        if index >= 0:

            self.display_mode_combo.setCurrentIndex(
                index
            )

    # --------------------------------------------------------
    # UI 초기화
    # --------------------------------------------------------

    def initUI(self):

        layout = QVBoxLayout()

        # 인터페이스 언어 선택

        self.ui_language_combo = QComboBox()

        self.ui_languages = {
            "한국어": "ko",
            "English": "en",
            "中文": "zh",
            "日本語": "ja"
        }

        for name, code in self.ui_languages.items():

            self.ui_language_combo.addItem(
                name,
                code
            )

        self.ui_language_combo.setCurrentIndex(
            0
        )

        language_layout = QHBoxLayout()

        self.lbl_language = QLabel()

        language_layout.addWidget(
            self.lbl_language
        )

        language_layout.addWidget(
            self.ui_language_combo
        )

        layout.addLayout(
            language_layout
        )

        # 언어 변경 이벤트

        self.ui_language_combo.currentIndexChanged.connect(
            self.change_ui_language
        )

        # ----------------------------------------------------
        # 오디오 입력
        # ----------------------------------------------------

        audio_layout = QHBoxLayout()

        p = pyaudio.PyAudio()

        self.device_combo = QComboBox()

        try:

            wasapi_info = p.get_host_api_info_by_type(
                pyaudio.paWASAPI
            )

            wasapi_index = wasapi_info["index"]

            loopback_devices = [
                dev
                for dev in p.get_loopback_device_info_generator()
                if dev["hostApi"] == wasapi_index
            ]

            for dev in loopback_devices:

                self.device_combo.addItem(
                    dev["name"],
                    dev["index"]
                )

            if not loopback_devices:

                self.device_combo.addItem(
                    "No WASAPI Loopback Device Found",
                    None
                )

        except Exception as e:

            logging.exception(
                "WASAPI Loopback device search failed"
            )

            self.device_combo.addItem(
                f"Device search failed: {e}",
                None
            )

        finally:

            p.terminate()

        self.btn_test = QPushButton()

        self.btn_test.setCheckable(
            True
        )

        self.btn_test.clicked.connect(
            self.toggle_audio_test
        )

        self.vol_bar = QProgressBar()

        self.vol_bar.setRange(
            0,
            100
        )

        self.vol_bar.setValue(
            0
        )

        self.vol_bar.setTextVisible(
            False
        )

        self.vol_bar.setFixedHeight(
            10
        )

        self.lbl_audio = QLabel()

        audio_layout.addWidget(
            self.lbl_audio
        )

        audio_layout.addWidget(
            self.device_combo
        )

        audio_layout.addWidget(
            self.btn_test
        )

        # ----------------------------------------------------
        # STT 모델 폴더
        # ----------------------------------------------------

        stt_layout = QHBoxLayout()

        self.stt_path_edit = QLineEdit()

        self.btn_browse = QPushButton()

        self.btn_browse.clicked.connect(
            self.browse_folder
        )

        self.lbl_stt = QLabel()

        stt_layout.addWidget(
            self.lbl_stt
        )

        stt_layout.addWidget(
            self.stt_path_edit
        )

        stt_layout.addWidget(
            self.btn_browse
        )

        # ----------------------------------------------------
        # Ollama 번역 모델
        # ----------------------------------------------------

        self.model_combo = QComboBox()

        self.lbl_model = QLabel()

        try:

            response = ollama.list()

            models = (
                response.models
                if hasattr(response, "models")
                else response.get("models", [])
            )

            for m in models:

                model_name = (
                    m.model
                    if hasattr(m, "model")
                    else m.get(
                        "name",
                        m.get("model", "")
                    )
                )

                if model_name:

                    self.model_combo.addItem(
                        model_name
                    )

            if self.model_combo.count() == 0:

                self.model_combo.addItem(
                    "No models installed (run pull in Terminal)"
                )

        except Exception as e:

            logging.exception(
                f"Failed to load Ollama models: {e}"
            )

            self.model_combo.addItem(
                "qwen2.5:7b"
            )

        # ----------------------------------------------------
        # 번역 표시 단위
        # ----------------------------------------------------

        self.display_mode_combo = QComboBox()

        self.lbl_display = QLabel()

        # ----------------------------------------------------
        # 인식 / 번역 언어
        # ----------------------------------------------------

        self.src_lang_combo = QComboBox()

        self.src_lang_combo.addItems([
            "auto",
            "en",
            "zh",
            "ja",
            "ko"
        ])

        self.target_lang_combo = QComboBox()

        self.target_lang_combo.addItems([
            "ko",
            "en",
            "zh",
            "ja"
        ])

        self.lbl_src_lang = QLabel()
        self.lbl_target_lang = QLabel()

        # ----------------------------------------------------
        # 투명도 슬라이더
        # ----------------------------------------------------

        self.opacity_slider = QSlider(
            Qt.Horizontal
        )

        self.opacity_slider.setRange(
            20,
            100
        )

        self.opacity_slider.setValue(
            100
        )

        self.opacity_slider.valueChanged.connect(
            lambda v: self.overlay.set_opacity_val(
                v / 100.0
            )
        )

        self.lbl_opacity = QLabel()

        # ----------------------------------------------------
        # 글씨 크기 슬라이더
        # ----------------------------------------------------

        self.size_slider = QSlider(
            Qt.Horizontal
        )

        self.size_slider.setRange(
            12,
            48
        )

        self.size_slider.setValue(
            18
        )

        self.size_slider.valueChanged.connect(
            self.overlay.set_font_size
        )

        self.lbl_font_size = QLabel()

        # ----------------------------------------------------
        # 자막 창 너비
        # ----------------------------------------------------

        self.width_slider = QSlider(
            Qt.Horizontal
        )

        self.width_slider.setRange(
            400,
            1600
        )

        self.width_slider.setValue(
            800
        )

        self.width_slider.valueChanged.connect(
            self.overlay.set_overlay_width
        )

        self.lbl_width = QLabel()

        # ----------------------------------------------------
        # 자막 창 높이
        # ----------------------------------------------------

        self.height_slider = QSlider(
            Qt.Horizontal
        )

        self.height_slider.setRange(
            100,
            500
        )

        self.height_slider.setValue(
            180
        )

        self.height_slider.valueChanged.connect(
            self.overlay.set_overlay_height
        )

        self.lbl_height = QLabel()

        # ----------------------------------------------------
        # 시작 / 중지 버튼
        # ----------------------------------------------------

        self.btn_start = QPushButton()

        self.btn_start.setStyleSheet(
            "font-weight: bold; padding: 10px;"
        )

        self.btn_start.clicked.connect(
            self.toggle_process
        )

        self.btn_pause = QPushButton()

        self.btn_pause.setEnabled(
            False
        )

        self.btn_pause.clicked.connect(
            self.toggle_pause
        )

        # ----------------------------------------------------
        # 전체 레이아웃 구성
        # ----------------------------------------------------

        layout.addLayout(
            audio_layout
        )

        layout.addWidget(
            self.vol_bar
        )

        layout.addLayout(
            stt_layout
        )

        layout.addWidget(
            self.lbl_model
        )

        layout.addWidget(
            self.model_combo
        )

        layout.addWidget(
            self.lbl_display
        )

        layout.addWidget(
            self.display_mode_combo
        )

        lang_layout = QHBoxLayout()

        lang_layout.addWidget(
            self.lbl_src_lang
        )

        lang_layout.addWidget(
            self.src_lang_combo
        )

        lang_layout.addWidget(
            self.lbl_target_lang
        )

        lang_layout.addWidget(
            self.target_lang_combo
        )

        layout.addLayout(
            lang_layout
        )

        layout.addWidget(
            self.lbl_opacity
        )

        layout.addWidget(
            self.opacity_slider
        )

        layout.addWidget(
            self.lbl_width
        )

        layout.addWidget(
            self.width_slider
        )

        layout.addWidget(
            self.lbl_height
        )

        layout.addWidget(
            self.height_slider
        )

        layout.addWidget(
            self.lbl_font_size
        )

        layout.addWidget(
            self.size_slider
        )

        layout.addWidget(
            self.btn_pause
        )

        layout.addWidget(
            self.btn_start
        )

        self.setLayout(
            layout
        )

        self.resize(
            500,
            300
        )

        self.load_settings()

        # 초기 언어 적용

        self.retranslate_ui()

    # --------------------------------------------------------
    # 폴더 선택
    # --------------------------------------------------------

    def browse_folder(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            get_text(
                self.ui_language,
                "stt_folder"
            )
        )

        if folder:

            self.stt_path_edit.setText(
                folder
            )

    # --------------------------------------------------------
    # 오디오 테스트
    # --------------------------------------------------------

    def toggle_audio_test(self):

        if self.btn_test.isChecked():

            if (
                self.worker is not None
                and self.worker.isRunning()
            ):

                self.btn_test.setChecked(
                    False
                )

                return

            device_idx = (
                self.device_combo.currentData()
            )

            if device_idx is None:

                self.btn_test.setChecked(
                    False
                )

                self.btn_test.setText(
                    get_text(
                        self.ui_language,
                        "audio_test"
                    )
                )

                return

            self.btn_test.setText(
                get_text(
                    self.ui_language,
                    "audio_test_stop"
                )
            )

            self.test_worker = AudioTestWorker(
                device_idx
            )

            self.test_worker.volume_signal.connect(
                self.vol_bar.setValue
            )

            self.test_worker.finished.connect(
                self.on_audio_test_finished
            )

            self.test_worker.start()

        else:

            self.btn_test.setText(
                "Stopping..."
            )

            self.btn_test.setEnabled(
                False
            )

            if self.test_worker is not None:

                self.test_worker.stop()

            else:

                self.btn_test.setEnabled(
                    True
                )

                self.btn_test.setText(
                    get_text(
                        self.ui_language,
                        "audio_test"
                    )
                )

    def on_audio_test_finished(self):

        self.btn_test.setText(
            get_text(
                self.ui_language,
                "audio_test"
            )
        )

        self.btn_test.setChecked(
            False
        )

        self.btn_test.setEnabled(
            True
        )

        self.vol_bar.setValue(
            0
        )

        self.test_worker = None

    # --------------------------------------------------------
    # 번역 시작 / 중지
    # --------------------------------------------------------

    def toggle_process(self):

        if (
            self.worker is None
            or not self.worker.isRunning()
        ):

            stt_path = (
                self.stt_path_edit.text().strip()
            )

            if not stt_path:

                self.overlay.update_text(
                    "Configuration Error",
                    "Please select an STT model folder first!"
                )

                return

            device_idx = (
                self.device_combo.currentData()
            )

            if device_idx is None:

                self.overlay.update_text(
                    "Configuration Error",
                    "Please select an available WASAPI loopback device."
                )

                return

            if self.btn_test.isChecked():

                self.btn_test.click()

            self.btn_test.setEnabled(
                False
            )

            src_lang = (
                self.src_lang_combo.currentText()
            )

            target_lang = (
                self.target_lang_combo.currentText()
            )

            selected_model = (
                self.model_combo.currentText()
            )

            # 내부 key를 가져옴

            selected_display_mode = (
                self.display_mode_combo.currentData()
            )

            selected_ui_language = (
                self.ui_language
            )

            self.worker = SpeechTranslateWorker(
                device_idx,
                src_lang,
                target_lang,
                selected_model,
                stt_path,
                selected_display_mode,
                selected_ui_language
            )

            self.worker.subtitle_signal.connect(
                self.overlay.update_text
            )

            self.worker.finished.connect(
                self.on_worker_finished
            )

            self.worker.start()

            self.btn_start.setText(
                get_text(
                    self.ui_language,
                    "stop"
                )
            )

            self.btn_pause.setEnabled(
                True
            )

            self.btn_pause.setText(
                get_text(
                    self.ui_language,
                    "pause"
                )
            )

        else:

            # wait() 대신 finished 시그널 사용

            self.worker.stop()

            self.btn_start.setEnabled(
                False
            )

            self.btn_start.setText(
                "Stopping..."
            )

    # --------------------------------------------------------
    # 번역 일시정지 / 재개
    # --------------------------------------------------------

    def toggle_pause(self):

        if (
            self.worker is None
            or not self.worker.isRunning()
        ):
            return

        is_paused = (
            self.worker.toggle_pause()
        )

        if is_paused:

            self.btn_pause.setText(
                get_text(
                    self.ui_language,
                    "resume"
                )
            )

        else:

            self.btn_pause.setText(
                get_text(
                    self.ui_language,
                    "pause"
                )
            )

    # --------------------------------------------------------
    # Worker 종료 후 UI 복구
    # --------------------------------------------------------

    def on_worker_finished(self):

        self.btn_start.setText(
            get_text(
                self.ui_language,
                "start"
            )
        )

        self.btn_start.setEnabled(
            True
        )

        self.btn_test.setEnabled(
            True
        )

        self.btn_pause.setEnabled(
            False
        )

        self.btn_pause.setText(
            get_text(
                self.ui_language,
                "pause"
            )
        )

        self.worker = None

# ------------------------------------------------------------
# 6. 프로그램 실행
# ------------------------------------------------------------

if __name__ == "__main__":

    app = QApplication(sys.argv)

    overlay = SubtitleOverlay()
    overlay.show()

    panel = ControlPanel(overlay)
    panel.show()

    sys.exit(
        app.exec_()
    )
