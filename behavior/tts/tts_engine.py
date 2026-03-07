"""
behavior/tts/tts_engine.py
─────────────────────────────────────────────────────────────────
Text-to-Speech engine abstraction.
Supports pyttsx3 (offline, lightweight) and Coqui TTS (higher quality).
"""

from __future__ import annotations

import threading
from typing import Optional

from config.config import BehaviorConfig
from utils.logger import get_logger

logger = get_logger(__name__)

TONE_RATE_MAP = {
    "warm": 165,
    "playful": 185,
    "concerned": 155,
    "professional": 170,
    "excited": 195,
    "gentle": 150,
    "neutral": 175,
}


class TTSEngine:
    """
    Thread-safe TTS abstraction supporting pyttsx3 and Coqui TTS.
    Speech runs in a background thread to avoid blocking the pipeline.
    """

    def __init__(self, config: BehaviorConfig):
        self.config = config
        self._engine = None
        self._coqui = None
        self._initialized = False
        self._lock = threading.Lock()
        self._speak_thread: Optional[threading.Thread] = None

    def initialize(self) -> None:
        if self.config.tts_engine == "pyttsx3":
            self._init_pyttsx3()
        elif self.config.tts_engine == "coqui":
            self._init_coqui()
        else:
            logger.warning(f"Unknown TTS engine: {self.config.tts_engine} — using pyttsx3")
            self._init_pyttsx3()

    def _init_pyttsx3(self) -> None:
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.config.tts_rate)
            self._engine.setProperty("volume", self.config.tts_volume)
            voices = self._engine.getProperty("voices")
            if voices:
                # Prefer female voice if available
                for voice in voices:
                    if "female" in voice.name.lower() or "zira" in voice.name.lower():
                        self._engine.setProperty("voice", voice.id)
                        break
            self._initialized = True
            logger.info("TTS initialized: pyttsx3")
        except Exception as e:
            logger.error(f"pyttsx3 init failed: {e}")

    def _init_coqui(self) -> None:
        try:
            from TTS.api import TTS
            self._coqui = TTS(model_name=self.config.coqui_model, progress_bar=False)
            self._initialized = True
            logger.info(f"TTS initialized: Coqui ({self.config.coqui_model})")
        except Exception as e:
            logger.error(f"Coqui TTS init failed: {e} — falling back to pyttsx3")
            self._init_pyttsx3()

    def speak(self, text: str, tone: str = "neutral", blocking: bool = False) -> None:
        """
        Speak the given text.
        If blocking=False (default), speech runs in a background thread.
        """
        if not self._initialized:
            logger.warning(f"TTS not initialized — suppressing: '{text}'")
            return

        if not text:
            return

        def _do_speak():
            with self._lock:
                try:
                    if self._engine:  # pyttsx3
                        rate = TONE_RATE_MAP.get(tone, self.config.tts_rate)
                        self._engine.setProperty("rate", rate)
                        self._engine.say(text)
                        self._engine.runAndWait()
                    elif self._coqui:
                        import sounddevice as sd
                        import numpy as np
                        wav = self._coqui.tts(text=text)
                        sd.play(np.array(wav), samplerate=22050, blocking=True)
                except Exception as e:
                    logger.error(f"TTS speak error: {e}")

        if blocking:
            _do_speak()
        else:
            self._speak_thread = threading.Thread(target=_do_speak, daemon=True)
            self._speak_thread.start()

    def is_speaking(self) -> bool:
        return self._speak_thread is not None and self._speak_thread.is_alive()

    def stop(self) -> None:
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
