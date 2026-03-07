"""
perception/camera/processor.py
─────────────────────────────────────────────────────────────────
Frame pre-processing: resize, CLAHE enhancement, quality gating.
Converts raw BGR frames into pipeline-ready inputs and computes
quality metadata to gate unusable frames early.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from config.config import FrameConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameMetadata:
    timestamp: float
    frame_id: int
    original_shape: tuple
    processed_shape: tuple
    brightness: float
    blur_score: float
    is_usable: bool


@dataclass
class ProcessedFrame:
    bgr: np.ndarray          # Enhanced BGR frame (for display / detection)
    rgb: np.ndarray          # RGB frame (for ML models expecting RGB)
    gray: np.ndarray         # Grayscale (for face/blob processing)
    metadata: FrameMetadata


class FrameProcessor:
    """
    Applies deterministic pre-processing to raw camera frames.

    Pipeline:
      Raw BGR → Resize → CLAHE enhancement → RGB conversion
      → Quality scoring → Usability gate
    """

    def __init__(self, config: FrameConfig):
        self.config = config
        self._clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_size, config.clahe_tile_size)
        )
        self._frame_id = 0

    def process(self, raw_frame: np.ndarray) -> ProcessedFrame:
        """
        Process a single raw camera frame.
        Always returns a ProcessedFrame; check `metadata.is_usable`
        before running expensive downstream models.
        """
        self._frame_id += 1
        ts = time.time()
        original_shape = raw_frame.shape

        # ── Step 1: Resize to standard processing resolution ──
        resized = cv2.resize(
            raw_frame,
            (self.config.target_width, self.config.target_height),
            interpolation=cv2.INTER_LINEAR
        )

        # ── Step 2: CLAHE on L channel (Lab color space) ──────
        enhanced_bgr = resized
        if self.config.apply_clahe:
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = self._clahe.apply(l_ch)
            lab_enhanced = cv2.merge([l_ch, a_ch, b_ch])
            enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # ── Step 3: Derive color space variants ───────────────
        rgb_frame = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

        # ── Step 4: Quality metrics ───────────────────────────
        blur_score = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())
        brightness = float(np.mean(gray_frame))

        is_usable = (
            blur_score >= self.config.blur_threshold
            and self.config.min_brightness <= brightness <= self.config.max_brightness
        )

        metadata = FrameMetadata(
            timestamp=ts,
            frame_id=self._frame_id,
            original_shape=original_shape,
            processed_shape=resized.shape,
            brightness=brightness,
            blur_score=blur_score,
            is_usable=is_usable,
        )

        return ProcessedFrame(
            bgr=enhanced_bgr,
            rgb=rgb_frame,
            gray=gray_frame,
            metadata=metadata,
        )
