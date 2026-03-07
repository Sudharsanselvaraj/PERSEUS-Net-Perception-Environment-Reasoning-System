"""
behavior/executor.py
─────────────────────────────────────────────────────────────────
Behavior execution coordinator.
Translates AgentAction decisions into concrete output:
TTS speech, LED color, display updates, animations, serial commands.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional

from agent.aura_agent import AgentAction
from behavior.tts.tts_engine import TTSEngine
from behavior.leds.led_controller import LEDController
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    action_type: str
    message_spoken: Optional[str]
    led_color_set: Optional[tuple]
    animation_played: Optional[str]
    execution_time_ms: float
    success: bool


class BehaviorExecutor:
    """
    Coordinates execution of all output modalities for a given AgentAction.
    Handles urgency (immediate vs. delayed) and prevents output during speech.
    """

    def __init__(
        self,
        tts: TTSEngine,
        leds: LEDController,
    ):
        self.tts = tts
        self.leds = leds
        self._execution_lock = threading.Lock()

    def execute(self, action: AgentAction) -> ExecutionResult:
        """
        Execute an AgentAction across all output modalities.
        Returns an ExecutionResult with execution metadata.
        """
        if action.action_type == "silence":
            return ExecutionResult(
                action_type="silence",
                message_spoken=None,
                led_color_set=None,
                animation_played=None,
                execution_time_ms=0.0,
                success=True,
            )

        t0 = time.perf_counter()

        # Apply urgency delay
        if action.urgency == "delayed_5s":
            time.sleep(5.0)
        elif action.urgency == "next_opportunity":
            # Wait until TTS is not speaking
            for _ in range(50):
                if not self.tts.is_speaking():
                    break
                time.sleep(0.1)

        led_color = None
        animation_played = None

        with self._execution_lock:
            # ── LED update ────────────────────────────────────
            self.leds.set_tone(action.tone)
            from behavior.leds.led_controller import TONE_COLORS
            led_color = TONE_COLORS.get(action.tone, (200, 200, 200))

            # ── Animation ─────────────────────────────────────
            if action.animation:
                self._trigger_animation(action.animation, action.tone)
                animation_played = action.animation

            # ── Speech ────────────────────────────────────────
            if action.message:
                logger.info(f"[Aura speaks] {action.message}")
                self.tts.speak(action.message, tone=action.tone)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ExecutionResult(
            action_type=action.action_type,
            message_spoken=action.message,
            led_color_set=led_color,
            animation_played=animation_played,
            execution_time_ms=elapsed_ms,
            success=True,
        )

    def _trigger_animation(self, animation_name: str, tone: str) -> None:
        """
        Trigger a named animation. Animations are short LED sequences
        that run concurrently with speech.
        """
        from behavior.leds.led_controller import TONE_COLORS
        color = TONE_COLORS.get(tone, (200, 200, 200))

        pulse_animations = {
            "wave_back", "happy_bounce", "calm_nod",
            "wave_goodbye", "gentle_pulse", "soft_glow_purple",
        }
        if animation_name in pulse_animations:
            self.leds.pulse(*color, duration=2.0)
        else:
            self.leds.set_color(*color)

        logger.debug(f"Animation triggered: {animation_name}")

    def idle(self) -> None:
        """Set device to idle state (dim LEDs, no speech)."""
        self.leds.idle()
