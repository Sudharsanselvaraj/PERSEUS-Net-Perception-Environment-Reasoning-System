"""
perception/detection/human_detector.py
─────────────────────────────────────────────────────────────────
YOLOv8-based human presence detector.
Tier-1 model: runs every frame for presence gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from config.config import DetectionConfig
from utils.logger import get_logger
from utils.timing import timed

logger = get_logger(__name__)


@dataclass
class HumanDetection:
    bbox: tuple            # (x1, y1, x2, y2)
    confidence: float
    center: tuple          # (cx, cy)
    area: int
    relative_position: str # e.g. "close_center", "far_left"
    track_id: int = -1     # Populated if ByteTrack is enabled


class HumanDetector:
    """
    Wraps YOLOv8 for single-class (person) detection.
    Computes spatial metadata on each bounding box.
    """

    PERSON_CLASS_ID = 0

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._model = None
        self._initialized = False

    def initialize(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.config.model_path)
            if self.config.half_precision and self.config.device != "cpu":
                self._model.model.half()
            logger.info(f"HumanDetector initialized: {self.config.model_path} "
                        f"on {self.config.device}")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize HumanDetector: {e}")
            raise

    @timed("human_detection")
    def detect(self, frame: np.ndarray) -> List[HumanDetection]:
        """
        Run person detection on a BGR/RGB frame.
        Returns list of HumanDetection (may be empty).
        """
        if not self._initialized:
            raise RuntimeError("HumanDetector not initialized. Call initialize() first.")

        results = self._model(
            frame,
            classes=[self.PERSON_CLASS_ID],
            conf=self.config.confidence_threshold,
            device=self.config.device,
            verbose=False,
        )

        detections: List[HumanDetection] = []
        frame_w = frame.shape[1]

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)

                # Horizontal zone
                if cx < frame_w * 0.33:
                    h_zone = "left"
                elif cx > frame_w * 0.66:
                    h_zone = "right"
                else:
                    h_zone = "center"

                proximity = "close" if area > self.config.close_area_threshold else "far"

                detections.append(HumanDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    center=(cx, cy),
                    area=area,
                    relative_position=f"{proximity}_{h_zone}",
                ))

        return detections

    @property
    def human_present(self) -> bool:
        """Convenience: True if last detect() returned at least one person."""
        return self._last_count > 0

    def detect_with_presence(self, frame: np.ndarray):
        """Returns (detections, is_present) tuple."""
        dets = self.detect(frame)
        self._last_count = len(dets)
        return dets, len(dets) > 0

    _last_count: int = 0
