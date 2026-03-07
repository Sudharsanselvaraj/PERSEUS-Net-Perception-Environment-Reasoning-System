"""
perception/emotion/emotion_detector.py
─────────────────────────────────────────────────────────────────
DeepFace-based facial emotion recognition.
Tier-2 model: runs every ~3 seconds on the face crop region.
Includes temporal smoothing to reduce per-frame noise.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from config.config import EmotionConfig
from utils.logger import get_logger
from utils.timing import timed

logger = get_logger(__name__)

# 7 basic emotions (FER+ / AffectNet label set)
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

VALENCE_MAP: Dict[str, str] = {
    "happy": "positive",
    "surprised": "positive",
    "neutral": "neutral",
    "sad": "negative",
    "angry": "negative",
    "fearful": "negative",
    "disgusted": "negative",
}

AROUSAL_MAP: Dict[str, str] = {
    "angry": "high",
    "surprised": "high",
    "fearful": "high",
    "happy": "medium",
    "disgusted": "medium",
    "sad": "low",
    "neutral": "low",
}


@dataclass
class EmotionResult:
    dominant_emotion: str
    emotion_scores: Dict[str, float]   # Sum to 1.0
    valence: str                        # positive / negative / neutral
    arousal: str                        # high / medium / low
    smoothed: bool = False             # True if result was from smoothing buffer


class EmotionDetector:
    """
    Wraps DeepFace for per-face emotion analysis.
    Maintains a temporal smoothing buffer to stabilize noisy per-frame outputs.
    """

    def __init__(self, config: EmotionConfig):
        self.config = config
        self._score_history: Deque[Dict[str, float]] = deque(
            maxlen=config.smoothing_window_frames
        )

    @timed("emotion_detection")
    def detect_emotion(self, face_crop: np.ndarray) -> Optional[EmotionResult]:
        """
        Analyze emotion from a pre-cropped face image (BGR or RGB).
        Returns None if no face is detected in the crop.
        """
        if face_crop is None or face_crop.size == 0:
            return self._get_smoothed_result()

        try:
            from deepface import DeepFace
            results = DeepFace.analyze(
                img_path=face_crop,
                actions=["emotion"],
                detector_backend=self.config.backend,
                enforce_detection=self.config.enforce_detection,
                silent=True,
            )

            if not results:
                return self._get_smoothed_result()

            raw_scores: Dict[str, float] = results[0]["emotion"]
            total = sum(raw_scores.values()) or 1.0
            normalized = {k: v / total for k, v in raw_scores.items()}

            self._score_history.append(normalized)
            return self._build_result(normalized, smoothed=False)

        except Exception as e:
            logger.debug(f"EmotionDetector error: {e}")
            return self._get_smoothed_result()

    # ── Internal ──────────────────────────────────────────────

    def _get_smoothed_result(self) -> Optional[EmotionResult]:
        """Return smoothed estimate from history buffer, or None if empty."""
        if not self._score_history:
            return None
        averaged = self._average_scores(list(self._score_history))
        return self._build_result(averaged, smoothed=True)

    @staticmethod
    def _average_scores(history: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute element-wise mean across score history."""
        combined: Dict[str, float] = {e: 0.0 for e in EMOTIONS}
        for scores in history:
            for emotion, score in scores.items():
                if emotion in combined:
                    combined[emotion] += score
        n = len(history)
        return {k: v / n for k, v in combined.items()}

    @staticmethod
    def _build_result(scores: Dict[str, float], smoothed: bool) -> EmotionResult:
        dominant = max(scores, key=scores.get)
        return EmotionResult(
            dominant_emotion=dominant,
            emotion_scores=scores,
            valence=VALENCE_MAP.get(dominant, "neutral"),
            arousal=AROUSAL_MAP.get(dominant, "low"),
            smoothed=smoothed,
        )
