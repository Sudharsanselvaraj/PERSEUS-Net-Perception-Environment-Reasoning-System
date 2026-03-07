"""
perception/scene/vlm_analyzer.py
─────────────────────────────────────────────────────────────────
Vision-Language Model scene understanding.
Tier-3 model: runs every ~10 seconds or on scene-change events.
Generates natural-language descriptions of the full scene.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config.config import SceneConfig
from utils.logger import get_logger
from utils.timing import timed

logger = get_logger(__name__)


@dataclass
class SceneUnderstandingResult:
    scene_description: str
    activity_summary: str
    environment_notes: str
    notable_events: Optional[str]
    raw_response: str = ""


_SCENE_PROMPT = """You are a perceptive AI assistant analyzing a live camera feed.
Describe what you observe concisely and factually. Focus on:
1. The person's current activity and posture
2. Visible objects and their arrangement
3. The apparent mood or energy level of the environment
4. Anything unusual or noteworthy

Respond ONLY with a valid JSON object — no preamble, no markdown fences:
{"scene_description": "...", "activity_summary": "...", "environment_notes": "...", "notable_events": "..."}
"""

_FALLBACK = SceneUnderstandingResult(
    scene_description="Scene analysis unavailable",
    activity_summary="unknown",
    environment_notes="",
    notable_events=None,
)


class VLMSceneAnalyzer:
    """
    Wraps a Vision-Language Model to produce structured scene descriptions.
    Supports Qwen-VL-Chat, LLaVA, and MiniCPM-V backends.
    Falls back gracefully if the model is unavailable.
    """

    def __init__(self, config: SceneConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._initialized = False

    def initialize(self) -> bool:
        if not self.config.enabled:
            logger.info("VLM scene analysis disabled in config")
            return False
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq

            quant_kwargs = {}
            if self.config.use_quantization:
                try:
                    from transformers import BitsAndBytesConfig
                    quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                except Exception:
                    logger.warning("BitsAndBytes not available — loading without quantization")

            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map=self.config.device,
                trust_remote_code=True,
                **quant_kwargs,
            )
            self._model.eval()
            self._initialized = True
            logger.info(f"VLMSceneAnalyzer initialized: {self.config.model_name}")
            return True

        except Exception as e:
            logger.warning(f"VLM initialization failed (scene analysis disabled): {e}")
            return False

    @timed("vlm_scene_analysis")
    def analyze_scene(self, frame: np.ndarray) -> SceneUnderstandingResult:
        """
        Generate a structured scene description from a BGR/RGB frame.
        Returns a fallback result if VLM is unavailable.
        """
        if not self._initialized:
            return _FALLBACK

        try:
            import torch
            from PIL import Image

            image = Image.fromarray(frame)
            inputs = self._processor(
                text=_SCENE_PROMPT,
                images=image,
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )

            raw = self._processor.decode(output[0], skip_special_tokens=True)
            return self._parse_response(raw)

        except Exception as e:
            logger.error(f"VLM scene analysis error: {e}")
            return _FALLBACK

    # ── Response Parsing ──────────────────────────────────────

    @staticmethod
    def _parse_response(text: str) -> SceneUnderstandingResult:
        # Strip markdown fences if present
        clean = re.sub(r"```json|```", "", text).strip()

        # Try to extract JSON object
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return SceneUnderstandingResult(
                    scene_description=data.get("scene_description", ""),
                    activity_summary=data.get("activity_summary", ""),
                    environment_notes=data.get("environment_notes", ""),
                    notable_events=data.get("notable_events"),
                    raw_response=text,
                )
            except json.JSONDecodeError:
                pass

        # Fallback: treat entire text as description
        return SceneUnderstandingResult(
            scene_description=clean[:500],
            activity_summary="",
            environment_notes="",
            notable_events=None,
            raw_response=text,
        )
