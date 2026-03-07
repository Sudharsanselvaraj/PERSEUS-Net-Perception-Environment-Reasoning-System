"""
perception/objects/object_detector.py
─────────────────────────────────────────────────────────────────
YOLOv8-based contextual object detection.
Tier-2 model: runs every ~3 seconds.
Detects objects relevant to activity inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List

import numpy as np

from config.config import ObjectsConfig
from utils.logger import get_logger
from utils.timing import timed

logger = get_logger(__name__)


# COCO class indices → context-relevant label
CONTEXT_OBJECTS: Dict[int, str] = {
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "cell_phone",
    62: "tv",
    72: "refrigerator",
    73: "book",
    74: "clock",
    39: "bottle",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    24: "backpack",
    26: "handbag",
    56: "chair",
    57: "couch",
    58: "potted_plant",
    76: "scissors",
    77: "teddy_bear",
    78: "hair_drier",
    79: "toothbrush",
}

# Object-set → inferred activity (order matters — first match wins)
ACTIVITY_RULES: List[tuple] = [
    (frozenset(["laptop", "keyboard", "mouse"]), "working_at_computer"),
    (frozenset(["laptop", "keyboard"]),           "working_at_computer"),
    (frozenset(["laptop"]),                        "using_laptop"),
    (frozenset(["book"]),                          "reading"),
    (frozenset(["bowl", "fork"]),                  "eating"),
    (frozenset(["bowl", "spoon"]),                 "eating"),
    (frozenset(["cup"]),                           "having_a_drink"),
    (frozenset(["bottle"]),                        "having_a_drink"),
    (frozenset(["cell_phone"]),                    "using_phone"),
    (frozenset(["tv"]),                            "watching_tv"),
    (frozenset(["couch"]),                         "relaxing"),
]


@dataclass
class ObjectDetectionResult:
    detected_objects: List[str]
    object_details: List[Dict]   # {name, confidence, bbox} per detection
    inferred_activity: str
    scene_complexity: str        # sparse | moderate | cluttered


class ContextObjectDetector:
    """
    Reuses YOLOv8 (larger variant than Tier-1) to detect contextual
    objects and infer current user activity.
    """

    def __init__(self, config: ObjectsConfig):
        self.config = config
        self._model = None
        self._initialized = False

    def initialize(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.config.model_path)
            self._initialized = True
            logger.info(f"ContextObjectDetector initialized: {self.config.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ContextObjectDetector: {e}")
            raise

    @timed("object_detection")
    def detect(self, frame: np.ndarray) -> ObjectDetectionResult:
        if not self._initialized:
            raise RuntimeError("ContextObjectDetector not initialized.")

        target_classes = list(CONTEXT_OBJECTS.keys())
        results = self._model(
            frame,
            classes=target_classes,
            conf=self.config.confidence_threshold,
            device=self.config.device,
            verbose=False,
        )

        seen: set = set()
        detected_objects: List[str] = []
        object_details: List[Dict] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = CONTEXT_OBJECTS.get(cls_id, "unknown")
                conf = float(box.conf[0])
                bbox = tuple(map(int, box.xyxy[0].tolist()))

                object_details.append({"name": name, "confidence": conf, "bbox": bbox})
                if name not in seen:
                    detected_objects.append(name)
                    seen.add(name)

        inferred_activity = self._infer_activity(frozenset(detected_objects))
        n = len(object_details)
        complexity = "sparse" if n < 3 else ("moderate" if n < 8 else "cluttered")

        return ObjectDetectionResult(
            detected_objects=detected_objects,
            object_details=object_details,
            inferred_activity=inferred_activity,
            scene_complexity=complexity,
        )

    @staticmethod
    def _infer_activity(detected: FrozenSet[str]) -> str:
        for pattern, activity in ACTIVITY_RULES:
            if pattern.issubset(detected):
                return activity
        return "idle"
