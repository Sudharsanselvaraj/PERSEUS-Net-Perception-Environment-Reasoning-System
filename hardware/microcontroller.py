"""
hardware/microcontroller.py
─────────────────────────────────────────────────────────────────
Serial communication bridge to ESP32 / Arduino microcontroller.
Sends JSON command packets over USB serial for LED, servo, and
display control. Operates in simulation mode when serial is disabled.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

from config.config import HardwareConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class MicrocontrollerBridge:
    """
    Abstracts serial communication with the companion device's
    ESP32/Arduino microcontroller over USB.

    Command protocol: newline-delimited JSON strings
    {"cmd": "led_color", "r": 255, "g": 100, "b": 50}
    {"cmd": "servo_move", "yaw": 90, "pitch": 75}
    {"cmd": "play_sound", "file": "chime.wav"}
    {"cmd": "display_text", "text": "Hello!", "color": "white"}
    """

    def __init__(self, config: HardwareConfig):
        self.config = config
        self._serial = None
        self._lock = threading.Lock()
        self._connected = False

    def connect(self) -> bool:
        if not self.config.serial_enabled:
            logger.info("MicrocontrollerBridge: simulation mode (serial disabled)")
            self._connected = True
            return True
        try:
            import serial
            self._serial = serial.Serial(
                port=self.config.serial_port,
                baudrate=self.config.serial_baud,
                timeout=1,
            )
            time.sleep(2)  # Wait for microcontroller boot
            self._connected = True
            logger.info(f"Microcontroller connected on {self.config.serial_port} "
                        f"@ {self.config.serial_baud} baud")
            return True
        except Exception as e:
            logger.warning(f"Microcontroller connection failed: {e} (running without hardware)")
            self._connected = True  # Allow sim mode
            return False

    def disconnect(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._connected = False

    # ── Command Interface ─────────────────────────────────────

    def send_command(self, cmd: str, **params) -> bool:
        payload = json.dumps({"cmd": cmd, **params}) + "\n"
        with self._lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.write(payload.encode("utf-8"))
                    return True
                except Exception as e:
                    logger.error(f"Serial write error: {e}")
                    return False
            else:
                logger.debug(f"[MCU SIM] {payload.strip()}")
                return True

    def set_led_color(self, r: int, g: int, b: int) -> bool:
        return self.send_command("led_color", r=r, g=g, b=b)

    def set_led_brightness(self, brightness: int) -> bool:
        """brightness: 0–255"""
        return self.send_command("led_brightness", value=brightness)

    def play_animation(self, name: str) -> bool:
        return self.send_command("play_animation", name=name)

    def move_head(self, yaw_degrees: float, pitch_degrees: Optional[float] = None) -> bool:
        if not self.config.servo_enabled:
            return True
        pitch = pitch_degrees if pitch_degrees is not None else self.config.head_pitch_neutral
        return self.send_command("servo_move", yaw=yaw_degrees, pitch=pitch)

    def reset_head(self) -> bool:
        return self.move_head(
            yaw_degrees=self.config.head_yaw_neutral,
            pitch_degrees=self.config.head_pitch_neutral,
        )

    def play_sound(self, filename: str) -> bool:
        return self.send_command("play_sound", file=filename)

    def display_text(self, text: str, color: str = "white") -> bool:
        return self.send_command("display_text", text=text[:64], color=color)

    def display_clear(self) -> bool:
        return self.send_command("display_clear")

    @property
    def is_connected(self) -> bool:
        return self._connected
