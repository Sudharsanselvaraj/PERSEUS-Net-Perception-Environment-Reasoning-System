"""
perception/orchestrator.py
─────────────────────────────────────────────────────────────────
Perception pipeline orchestrator.
Manages the three-tier processing schedule and shared state
between the realtime loop and background inference threads.

Tier 1 — Every frame  (human detection, gesture)
Tier 2 — Every ~3s    (face recognition, emotion, objects)
Tier 3 — Every ~10s   (VLM scene analysis → context build → agent)
"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from config.config import PipelineConfig
from perception.detection.human_detector import HumanDetector, HumanDetection
from perception.recognition.face_recognizer import FaceRecognitionSystem, FaceRecognitionResult
from perception.emotion.emotion_detector import EmotionDetector, EmotionResult
from perception.gesture.gesture_recognizer import GestureRecognizer, GestureResult
from perception.objects.object_detector import ContextObjectDetector, ObjectDetectionResult
from perception.scene.vlm_analyzer import VLMSceneAnalyzer, SceneUnderstandingResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerceptionState:
    """Thread-safe snapshot of the latest outputs from all tiers."""
    human_detections: List[HumanDetection] = field(default_factory=list)
    human_present: bool = False
    gesture: Optional[GestureResult] = None
    face_results: Optional[List[FaceRecognitionResult]] = None
    emotion: Optional[EmotionResult] = None
    objects: Optional[ObjectDetectionResult] = None
    scene: Optional[SceneUnderstandingResult] = None
    last_tier2_ts: float = 0.0
    last_tier3_ts: float = 0.0


class PerceptionOrchestrator:
    """
    Central orchestrator for the full perception pipeline.

    Usage:
        orchestrator = PerceptionOrchestrator(...)
        orchestrator.initialize()

        # In main loop:
        state = orchestrator.process_frame(frame)
    """

    def __init__(
        self,
        config: PipelineConfig,
        detector: HumanDetector,
        face_recognizer: FaceRecognitionSystem,
        emotion_detector: EmotionDetector,
        gesture_recognizer: GestureRecognizer,
        object_detector: ContextObjectDetector,
        scene_analyzer: VLMSceneAnalyzer,
        on_tier3_complete: Optional[Callable] = None,
    ):
        self.config = config
        self.detector = detector
        self.face_recognizer = face_recognizer
        self.emotion_detector = emotion_detector
        self.gesture_recognizer = gesture_recognizer
        self.object_detector = object_detector
        self.scene_analyzer = scene_analyzer
        self.on_tier3_complete = on_tier3_complete  # Callback for context engine

        self._executor = ThreadPoolExecutor(
            max_workers=config.max_worker_threads,
            thread_name_prefix="PerceptionWorker",
        )
        self._state = PerceptionState()
        self._lock = threading.Lock()
        self._tier2_pending = False
        self._tier3_pending = False

    def initialize(self) -> None:
        """Initialize all perception models."""
        self.detector.initialize()
        self.gesture_recognizer.initialize()
        self.face_recognizer.initialize()
        self.emotion_detector  # No init needed (lazy via DeepFace)
        self.object_detector.initialize()
        self.scene_analyzer.initialize()
        logger.info("PerceptionOrchestrator fully initialized")

    # ── Main Entry Point ──────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> PerceptionState:
        """
        Process a single frame through the tiered pipeline.
        Tier 1 runs synchronously; Tiers 2 & 3 are scheduled async.
        Returns a copy of the latest PerceptionState.
        """
        now = time.time()

        # ── Tier 1: Synchronous, every frame ─────────────────
        detections, present = self.detector.detect_with_presence(frame)
        gesture = self.gesture_recognizer.process(frame)

        with self._lock:
            self._state.human_detections = detections
            self._state.human_present = present
            self._state.gesture = gesture

        # ── Tier 2: Schedule if interval elapsed ─────────────
        if (now - self._state.last_tier2_ts >= self.config.tier2_interval_seconds
                and not self._tier2_pending):
            self._tier2_pending = True
            self._executor.submit(self._run_tier2, frame.copy(), now)

        # ── Tier 3: Schedule if interval elapsed ─────────────
        if (now - self._state.last_tier3_ts >= self.config.tier3_interval_seconds
                and not self._tier3_pending):
            self._tier3_pending = True
            self._executor.submit(self._run_tier3, frame.copy(), now)

        # Return snapshot
        with self._lock:
            return PerceptionState(
                human_detections=self._state.human_detections,
                human_present=self._state.human_present,
                gesture=self._state.gesture,
                face_results=self._state.face_results,
                emotion=self._state.emotion,
                objects=self._state.objects,
                scene=self._state.scene,
                last_tier2_ts=self._state.last_tier2_ts,
                last_tier3_ts=self._state.last_tier3_ts,
            )

    # ── Background Workers ────────────────────────────────────

    def _run_tier2(self, frame: np.ndarray, ts: float) -> None:
        try:
            face_results = None
            emotion = None
            objects = None

            try:
                face_results = self.face_recognizer.recognize(frame)
            except Exception as fe:
                logger.warning(f"Face recognition failed: {fe}")

            # Crop face region for emotion analysis
            if face_results:
                try:
                    x1, y1, x2, y2 = face_results[0].face_bbox
                    # Clamp to frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        emotion = self.emotion_detector.detect_emotion(face_crop)
                except Exception as ee:
                    logger.warning(f"Emotion detection failed: {ee}")

            try:
                objects = self.object_detector.detect(frame)
            except Exception as oe:
                logger.warning(f"Object detection failed: {oe}")

            with self._lock:
                self._state.face_results = face_results
                self._state.emotion = emotion
                self._state.objects = objects
                self._state.last_tier2_ts = ts

            logger.debug(f"Tier 2 complete: faces={len(face_results) if face_results else 0}, "
                        f"emotion={emotion.dominant_emotion if emotion else None}, "
                        f"objects={len(objects.detected_objects) if objects else 0}")

        except Exception as e:
            logger.error(f"Tier 2 processing error: {e}", exc_info=True)
        finally:
            self._tier2_pending = False

    def _run_tier3(self, frame: np.ndarray, ts: float) -> None:
        try:
            scene = self.scene_analyzer.analyze_scene(frame)

            with self._lock:
                self._state.scene = scene
                self._state.last_tier3_ts = ts

            logger.debug("Tier 3 scene analysis complete")

            # Trigger context engine + agent via callback
            if self.on_tier3_complete:
                try:
                    self.on_tier3_complete()
                except Exception as cb_err:
                    logger.error(f"Tier 3 callback error: {cb_err}", exc_info=True)

        except Exception as e:
            logger.error(f"Tier 3 processing error: {e}", exc_info=True)
        finally:
            self._tier3_pending = False

    def get_state(self) -> PerceptionState:
        with self._lock:
            return PerceptionState(**vars(self._state))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
        logger.info("PerceptionOrchestrator shutdown complete")
