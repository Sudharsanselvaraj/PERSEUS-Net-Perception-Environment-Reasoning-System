"""
context/context_engine.py
─────────────────────────────────────────────────────────────────
Context synthesis engine.
Combines perception outputs + memory summaries into a single,
structured ContextObject consumed by the AI agent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from memory.temporal_buffer import BufferSummary
from memory.session_memory import SessionMemory
from perception.recognition.face_recognizer import FaceRecognitionResult
from perception.emotion.emotion_detector import EmotionResult
from perception.gesture.gesture_recognizer import GestureResult
from perception.objects.object_detector import ObjectDetectionResult
from perception.scene.vlm_analyzer import SceneUnderstandingResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContextObject:
    # Identity
    user_id: Optional[str]
    user_display_name: Optional[str]
    identity_confidence: float

    # Emotional State
    emotion: str
    emotion_scores: Dict[str, float]
    emotion_trend: str
    valence: str

    # Activity
    activity: str
    gesture: str
    body_orientation: str

    # Environment
    detected_objects: List[str]
    scene_description: str
    environment_notes: str

    # Temporal
    presence_duration_seconds: float
    session_summary: str
    time_of_day: str

    # Events
    event: str
    notable_events: List[str]

    # Metadata
    timestamp: float
    context_version: str = "1.0"

    # ── Serialization ─────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_prompt_string(self) -> str:
        """
        Compact natural-language summary for LLM system prompt injection.
        """
        parts = [
            f"User: {self.user_display_name or 'unrecognized'} "
            f"(confidence: {self.identity_confidence:.0%})",
            f"Emotion: {self.emotion} ({self.valence}, trend: {self.emotion_trend})",
            f"Activity: {self.activity}",
            f"Gesture: {self.gesture}",
            f"Body orientation: {self.body_orientation}",
            f"Objects in scene: {', '.join(self.detected_objects) or 'none'}",
            f"Event: {self.event}",
            f"Time of day: {self.time_of_day}",
            f"Presence duration: {int(self.presence_duration_seconds // 60)} min",
            f"Session: {self.session_summary}",
        ]
        if self.scene_description:
            parts.append(f"Scene: {self.scene_description}")
        if self.notable_events:
            parts.append(f"Notable events: {', '.join(self.notable_events)}")
        return "\n".join(parts)


# ── Time-of-Day Helper ─────────────────────────────────────────

_TIME_BANDS = [
    (5,  12, "morning"),
    (12, 17, "afternoon"),
    (17, 21, "evening"),
    (21, 24, "night"),
    (0,   5, "night"),
]


def _get_time_of_day() -> str:
    hour = time.localtime().tm_hour
    for start, end, label in _TIME_BANDS:
        if start <= hour < end:
            return label
    return "night"


# ── Event Classifier ──────────────────────────────────────────

def _classify_event(
    face_results: Optional[List[FaceRecognitionResult]],
    gesture: GestureResult,
    buffer_summary: BufferSummary,
) -> str:
    if face_results and any(f.is_new_face for f in face_results):
        return "new_person_detected"
    if gesture.is_waving:
        return "user_waving"
    if gesture.gesture_name == "thumbs_up":
        return "user_approval_gesture"
    if gesture.gesture_name == "open_palm":
        return "user_stop_gesture"
    if buffer_summary.dominant_emotion in ("sad", "angry", "fearful"):
        return "user_appears_distressed"
    if buffer_summary.presence_duration_seconds >= 90 * 60:
        return "extended_session"
    if not buffer_summary.user_id and (face_results is None or len(face_results) == 0):
        return "no_user_detected"
    if buffer_summary.user_id:
        return "user_present"
    return "unknown_person_present"


# ── Context Engine ────────────────────────────────────────────

class ContextEngine:
    """
    Synthesizes all perception and memory outputs into a ContextObject.
    Stateless: build_context() can be called any time with fresh inputs.
    """

    def build_context(
        self,
        face_results: Optional[List[FaceRecognitionResult]],
        emotion: Optional[EmotionResult],
        gesture: Optional[GestureResult],
        objects: Optional[ObjectDetectionResult],
        scene: Optional[SceneUnderstandingResult],
        buffer_summary: BufferSummary,
        session: SessionMemory,
    ) -> ContextObject:

        # ── Identity ──────────────────────────────────────────
        user_id = None
        user_name = None
        id_confidence = 0.0
        if face_results:
            best = max(face_results, key=lambda f: f.confidence)
            user_id = best.user_id
            user_name = best.display_name
            id_confidence = best.confidence

        # ── Emotion ───────────────────────────────────────────
        emo_label = emotion.dominant_emotion if emotion else buffer_summary.dominant_emotion
        emo_scores = emotion.emotion_scores if emotion else {}
        valence = emotion.valence if emotion else "neutral"
        trend = buffer_summary.emotion_trend

        # ── Activity ──────────────────────────────────────────
        object_activity = objects.inferred_activity if objects else "idle"
        activity = buffer_summary.primary_activity
        if activity == "idle" and object_activity != "idle":
            activity = object_activity

        # ── Gesture ───────────────────────────────────────────
        gest_name = gesture.gesture_name if gesture else "none"
        body_orient = gesture.body_orientation if gesture else "unknown"

        # ── Objects ───────────────────────────────────────────
        obj_list = objects.detected_objects if objects else buffer_summary.unique_objects

        # ── Scene ─────────────────────────────────────────────
        scene_desc = scene.scene_description if scene else ""
        env_notes = scene.environment_notes if scene else ""

        # ── Event ─────────────────────────────────────────────
        event = _classify_event(face_results, gesture or GestureResult(
            gesture_name="none", confidence=0.0, is_waving=False,
            body_orientation="unknown", hand_detected=False, pose_detected=False
        ), buffer_summary)

        # Log to session memory
        if user_id and not session.user_id:
            session.user_id = user_id
        session.log_activity(activity)
        if emotion:
            session.log_emotion(emo_label, valence)
        for evt in buffer_summary.notable_events:
            session.add_notable_event(evt)
        if scene_desc:
            session.log_scene(scene_desc)

        return ContextObject(
            user_id=user_id,
            user_display_name=user_name,
            identity_confidence=id_confidence,
            emotion=emo_label,
            emotion_scores=emo_scores,
            emotion_trend=trend,
            valence=valence,
            activity=activity,
            gesture=gest_name,
            body_orientation=body_orient,
            detected_objects=obj_list,
            scene_description=scene_desc,
            environment_notes=env_notes,
            presence_duration_seconds=buffer_summary.presence_duration_seconds,
            session_summary=session.get_session_summary(),
            time_of_day=_get_time_of_day(),
            event=event,
            notable_events=buffer_summary.notable_events,
            timestamp=time.time(),
        )
