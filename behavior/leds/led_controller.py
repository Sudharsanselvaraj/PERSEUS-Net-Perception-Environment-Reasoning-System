"""
behavior/leds/led_controller.py
─────────────────────────────────────────────────────────────────
LED ring controller.
Sends color/animation commands to the microcontroller via serial,
or simulates LED output in console mode for development.
"""

from __future__ import annotations

import time
import threading
from typing import Tuple

from config.config import HardwareConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Tone → RGB color mapping
TONE_COLORS: dict = {
    "warm":         (255, 140, 40),
    "playful":      (80,  200, 255),
    "concerned":    (255, 80,  80),
    "professional": (180, 180, 255),
    "excited":      (80,  255, 80),
    "gentle":       (180, 100, 255),
    "neutral":      (200, 200, 200),
    "idle":         (20,  20,  40),
    "listening":    (0,   120, 255),
    "thinking":     (255, 200, 0),
}


class LEDController:
    """
    Controls the NeoPixel LED ring.
    In development mode (serial_enabled=False), logs state to console.
    """

    def __init__(self, config: HardwareConfig):
        self.config = config
        self._serial = None
        self._initialized = False
        self._current_color: Tuple[int, int, int] = (20, 20, 40)
        self._animation_thread: threading.Thread | None = None
        self._animate = False

    def initialize(self) -> None:
        if not self.config.serial_enabled:
            logger.info("LEDController: running in simulation mode (no serial)")
            self._initialized = True
            return
        try:
            import serial
            self._serial = serial.Serial(
                self.config.serial_port,
                self.config.serial_baud,
                timeout=1,
            )
            time.sleep(2)  # Allow microcontroller to reset
            self._initialized = True
            logger.info(f"LEDController: connected to {self.config.serial_port}")
        except Exception as e:
            logger.warning(f"Serial init failed (simulation mode): {e}")
            self._initialized = True

    def set_color(self, r: int, g: int, b: int) -> None:
        self._current_color = (r, g, b)
        self._send(f"LED:{r},{g},{b}\n")

    def set_tone(self, tone: str) -> None:
        color = TONE_COLORS.get(tone, TONE_COLORS["neutral"])
        self.set_color(*color)

    def pulse(self, r: int, g: int, b: int, duration: float = 2.0) -> None:
        """Fade in/out animation for the given duration."""
        self._animate = True

        def _pulse_loop():
            start = time.time()
            while self._animate and time.time() - start < duration:
                for brightness in list(range(0, 100, 5)) + list(range(100, 0, -5)):
                    if not self._animate:
                        break
                    scaled = tuple(int(c * brightness / 100) for c in (r, g, b))
                    self._send(f"LED:{scaled[0]},{scaled[1]},{scaled[2]}\n")
                    time.sleep(0.03)
            self.set_color(*self._current_color)

        self._animation_thread = threading.Thread(target=_pulse_loop, daemon=True)
        self._animation_thread.start()

    def stop_animation(self) -> None:
        self._animate = False

    def off(self) -> None:
        self._animate = False
        self.set_color(0, 0, 0)

    def idle(self) -> None:
        self.set_tone("idle")

    def _send(self, command: str) -> None:
        if self._serial:
            try:
                self._serial.write(command.encode())
            except Exception as e:
                logger.debug(f"LED serial write failed: {e}")
        else:
            logger.debug(f"[LED SIM] {command.strip()}")
