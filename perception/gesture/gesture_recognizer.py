"""
perception/gesture/gesture_recognizer.py
─────────────────────────────────────────────────────────────────
MediaPipe Hands + Pose gesture and body orientation recognition.
Tier-1 model: runs every frame (CPU-optimized).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from config.config import GestureConfig
from utils.logger import get_logger
from utils.timing import timed

logger = get_logger(__name__)


@dataclass
class GestureResult:
    gesture_name: str         # Classified gesture label
    confidence: float
    is_waving: bool
    body_orientation: str     # facing | turned_left | turned_right | turned_away | unknown
    hand_detected: bool
    pose_detected: bool


class GestureRecognizer:
    """
    MediaPipe Hands + Pose gesture recognition pipeline.

    Gesture classification uses heuristic landmark rules.
    Waving is detected by tracking lateral wrist oscillation.
    Body orientation is inferred from shoulder visibility & width.
    """

    GESTURE_LABELS = {
        "thumbs_up", "fist", "open_palm", "pointing",
        "peace_sign", "waving", "other", "none",
    }

    def __init__(self, config: GestureConfig):
        self.config = config
        self._hands = None
        self._pose = None
        self._initialized = False
        self._wrist_x_history: Deque[float] = deque(maxlen=config.wave_window_frames)

    # ── Lifecycle ─────────────────────────────────────────────

    def initialize(self) -> None:
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._mp_pose = mp.solutions.pose

            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config.max_num_hands,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.config.pose_model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._initialized = True
            logger.info("GestureRecognizer initialized (MediaPipe Hands + Pose)")
        except Exception as e:
            logger.error(f"Failed to initialize GestureRecognizer: {e}")
            raise

    # ── Main Interface ────────────────────────────────────────

    @timed("gesture_recognition")
    def process(self, rgb_frame: np.ndarray) -> GestureResult:
        """
        Process an RGB frame and return gesture + orientation result.
        """
        if not self._initialized:
            raise RuntimeError("GestureRecognizer not initialized.")

        hand_results = self._hands.process(rgb_frame)
        pose_results = self._pose.process(rgb_frame)

        gesture_name = "none"
        confidence = 0.0
        is_waving = False
        hand_detected = bool(hand_results.multi_hand_landmarks)

        if hand_detected:
            primary = hand_results.multi_hand_landmarks[0]
            gesture_name, confidence = self._classify_hand(primary)
            wrist_x = primary.landmark[0].x
            is_waving = self._check_waving(wrist_x)
            if is_waving:
                gesture_name = "waving"
                confidence = 0.88

        pose_landmarks = (pose_results.pose_landmarks
                          if pose_results and pose_results.pose_landmarks else None)
        body_orientation = self._classify_orientation(pose_landmarks)

        return GestureResult(
            gesture_name=gesture_name,
            confidence=confidence,
            is_waving=is_waving,
            body_orientation=body_orientation,
            hand_detected=hand_detected,
            pose_detected=pose_landmarks is not None,
        )

    # ── Gesture Classification ────────────────────────────────

    @staticmethod
    def _classify_hand(landmarks) -> Tuple[str, float]:
        """
        Rule-based classification from 21 hand landmarks.
        MediaPipe landmark convention:
          0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip,
          16=ring_tip, 20=pinky_tip
        PIP joints: index=6, middle=10, ring=14, pinky=18
        """
        lm = landmarks.landmark

        thumb_up = lm[4].y < lm[3].y < lm[2].y

        index_up  = lm[8].y  < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up   = lm[16].y < lm[14].y
        pinky_up  = lm[20].y < lm[18].y

        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

        if thumb_up and fingers_up == 0:
            return "thumbs_up", 0.90
        if not thumb_up and fingers_up == 0:
            return "fist", 0.85
        if fingers_up >= 4:
            return "open_palm", 0.85
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "pointing", 0.82
        if index_up and middle_up and not ring_up and not pinky_up:
            return "peace_sign", 0.80
        return "other", 0.50

    def _check_waving(self, wrist_x: float) -> bool:
        """
        Waving = lateral wrist oscillation with ≥2 direction reversals
        in the tracking window, each reversal exceeding the threshold.
        """
        self._wrist_x_history.append(wrist_x)
        if len(self._wrist_x_history) < 6:
            return False

        history = list(self._wrist_x_history)
        changes = 0
        for i in range(2, len(history)):
            prev = history[i - 1] - history[i - 2]
            curr = history[i] - history[i - 1]
            if prev * curr < 0 and abs(curr) > self.config.wave_direction_threshold:
                changes += 1

        return changes >= 2

    @staticmethod
    def _classify_orientation(pose_landmarks) -> str:
        """
        Infer body orientation from shoulder landmark visibility.
        """
        if pose_landmarks is None:
            return "unknown"

        lm = pose_landmarks.landmark
        left_vis  = lm[11].visibility  # LEFT_SHOULDER
        right_vis = lm[12].visibility  # RIGHT_SHOULDER

        if left_vis > 0.7 and right_vis > 0.7:
            shoulder_width = abs(lm[11].x - lm[12].x)
            return "facing" if shoulder_width > 0.15 else "leaning_away"
        if left_vis > 0.7:
            return "turned_right"
        if right_vis > 0.7:
            return "turned_left"
        return "turned_away"
