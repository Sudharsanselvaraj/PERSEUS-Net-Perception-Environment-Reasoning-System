"""
config/config.py
─────────────────────────────────────────────────────────────────
Central configuration loader.
Reads settings.yaml, overlays with .env / environment variables,
and exposes typed Pydantic models consumed by all modules.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

# ── Sub-models ────────────────────────────────────────────────

class AppConfig(BaseModel):
    name: str = "Aura"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    log_dir: str = "data/logs"
    timezone: str = "UTC"


class CameraConfig(BaseModel):
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "v4l2"
    buffer_size: int = 1
    codec: str = "MJPG"


class FrameConfig(BaseModel):
    target_width: int = 640
    target_height: int = 480
    blur_threshold: float = 80.0
    min_brightness: int = 20
    max_brightness: int = 240
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8


class DetectionConfig(BaseModel):
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.45
    device: str = "cpu"
    half_precision: bool = False
    close_area_threshold: int = 100_000


class RecognitionConfig(BaseModel):
    model_name: str = "buffalo_sc"
    det_size: Tuple[int, int] = (320, 320)
    ctx_id: int = -1
    recognition_threshold: float = 0.45
    database_path: str = "data/face_db/face_db.pkl"


class EmotionConfig(BaseModel):
    backend: str = "opencv"
    enforce_detection: bool = False
    smoothing_window_frames: int = 10


class GestureConfig(BaseModel):
    max_num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    pose_model_complexity: int = 1
    wave_window_frames: int = 15
    wave_direction_threshold: float = 0.02


class ObjectsConfig(BaseModel):
    model_path: str = "yolov8s.pt"
    confidence_threshold: float = 0.40
    device: str = "cpu"


class SceneConfig(BaseModel):
    enabled: bool = True
    model_name: str = "Qwen/Qwen-VL-Chat"
    device: str = "auto"
    max_new_tokens: int = 256
    interval_seconds: float = 10.0
    use_quantization: bool = True


class MemoryConfig(BaseModel):
    short_term_window_seconds: float = 30.0
    session_db_path: str = "data/sessions/sessions.db"
    profile_db_path: str = "data/profiles/profiles.db"
    vector_db_path: str = "data/memory_vectors"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_episodic_memories: int = 500


class PipelineConfig(BaseModel):
    tier2_interval_seconds: float = 3.0
    tier3_interval_seconds: float = 10.0
    max_worker_threads: int = 4
    frame_drop_threshold: int = 5


class AgentConfig(BaseModel):
    backend: Literal["ollama", "anthropic", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    openai_model: str = "gpt-4o-mini"
    max_tokens: int = 512
    temperature: float = 0.7
    timeout_seconds: float = 10.0


class CooldownConfig(BaseModel):
    greet: int = 300
    reminder: int = 600
    comment: int = 180
    question: int = 300
    reaction: int = 30
    silence: int = 0


class BehaviorConfig(BaseModel):
    tts_engine: str = "pyttsx3"
    tts_rate: int = 175
    tts_volume: float = 0.9
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    display_enabled: bool = False
    display_width: int = 480
    display_height: int = 320


class HardwareConfig(BaseModel):
    serial_enabled: bool = False
    serial_port: str = "/dev/ttyUSB0"
    serial_baud: int = 115200
    led_count: int = 24
    servo_enabled: bool = False
    head_yaw_neutral: int = 90
    head_pitch_neutral: int = 75


# ── Root Config ───────────────────────────────────────────────

class AuraConfig(BaseModel):
    app: AppConfig = AppConfig()
    camera: CameraConfig = CameraConfig()
    frame: FrameConfig = FrameConfig()
    detection: DetectionConfig = DetectionConfig()
    recognition: RecognitionConfig = RecognitionConfig()
    emotion: EmotionConfig = EmotionConfig()
    gesture: GestureConfig = GestureConfig()
    objects: ObjectsConfig = ObjectsConfig()
    scene: SceneConfig = SceneConfig()
    memory: MemoryConfig = MemoryConfig()
    pipeline: PipelineConfig = PipelineConfig()
    agent: AgentConfig = AgentConfig()
    cooldowns: CooldownConfig = CooldownConfig()
    behavior: BehaviorConfig = BehaviorConfig()
    hardware: HardwareConfig = HardwareConfig()

    # Apply env-var overrides after YAML load
    def apply_env_overrides(self) -> "AuraConfig":
        if (v := os.getenv("AURA_AGENT_BACKEND")):
            self.agent.backend = v  # type: ignore
        if (v := os.getenv("OLLAMA_BASE_URL")):
            self.agent.ollama_base_url = v
        if (v := os.getenv("OLLAMA_MODEL")):
            self.agent.ollama_model = v
        if (v := os.getenv("CAMERA_DEVICE_ID")):
            self.camera.device_id = int(v)
        if (v := os.getenv("AURA_DEBUG")):
            self.app.debug = v.lower() == "true"
        if (v := os.getenv("AURA_LOG_LEVEL")):
            self.app.log_level = v.upper()
        if (v := os.getenv("SERIAL_PORT")):
            self.hardware.serial_port = v
        if (v := os.getenv("SERIAL_ENABLED")):
            self.hardware.serial_enabled = v.lower() == "true"
        return self


@lru_cache(maxsize=1)
def load_config(config_path: str = "config/settings.yaml") -> AuraConfig:
    """
    Load and cache the global AuraConfig singleton.
    Called once at startup; all modules import `get_config()`.
    """
    path = Path(config_path)
    if not path.exists():
        # Return defaults if config file not found
        return AuraConfig().apply_env_overrides()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = AuraConfig(**raw)
    config.apply_env_overrides()
    return config


def get_config() -> AuraConfig:
    """Convenience accessor for the cached config singleton."""
    return load_config()
