"""
tests/unit/test_context_engine.py
─────────────────────────────────────────────────────────────────
Unit tests for the ContextEngine.
"""

import time
import pytest
from unittest.mock import MagicMock

from context.context_engine import ContextEngine
from memory.temporal_buffer import BufferSummary
from memory.session_memory import SessionMemory
from perception.gesture.gesture_recognizer import GestureResult


def _make_buffer_summary(**kwargs):
    defaults = dict(
        dominant_emotion="neutral",
        emotion_trend="stable",
        primary_activity="idle",
        presence_duration_seconds=60.0,
        recent_gestures=[],
        unique_objects=[],
        buffer_size=5,
        user_id="user_001",
        notable_events=[],
    )
    defaults.update(kwargs)
    return BufferSummary(**defaults)


def _make_gesture(name="none", waving=False, orientation="facing"):
    return GestureResult(
        gesture_name=name,
        confidence=0.8,
        is_waving=waving,
        body_orientation=orientation,
        hand_detected=name != "none",
        pose_detected=True,
    )


class TestContextEngine:

    def setup_method(self):
        self.engine = ContextEngine()
        self.session = SessionMemory()

    def test_build_context_basic(self):
        summary = _make_buffer_summary()
        ctx = self.engine.build_context(
            face_results=None,
            emotion=None,
            gesture=_make_gesture(),
            objects=None,
            scene=None,
            buffer_summary=summary,
            session=self.session,
        )
        assert ctx.emotion == "neutral"
        assert ctx.activity == "idle"
        assert ctx.event in ("no_user_detected", "unknown_person_present", "user_present")

    def test_waving_detected_as_event(self):
        summary = _make_buffer_summary(user_id="user_001")
        ctx = self.engine.build_context(
            face_results=None,
            emotion=None,
            gesture=_make_gesture("waving", waving=True),
            objects=None,
            scene=None,
            buffer_summary=summary,
            session=self.session,
        )
        assert ctx.event == "user_waving"

    def test_distress_event_classified(self):
        summary = _make_buffer_summary(dominant_emotion="sad", user_id="user_001")
        ctx = self.engine.build_context(
            face_results=None,
            emotion=None,
            gesture=_make_gesture(),
            objects=None,
            scene=None,
            buffer_summary=summary,
            session=self.session,
        )
        assert ctx.event == "user_appears_distressed"

    def test_context_to_prompt_string_contains_key_fields(self):
        summary = _make_buffer_summary(dominant_emotion="happy")
        ctx = self.engine.build_context(
            face_results=None,
            emotion=None,
            gesture=_make_gesture(),
            objects=None,
            scene=None,
            buffer_summary=summary,
            session=self.session,
        )
        prompt_str = ctx.to_prompt_string()
        assert "happy" in prompt_str
        assert "idle" in prompt_str

    def test_context_has_timestamp(self):
        summary = _make_buffer_summary()
        before = time.time()
        ctx = self.engine.build_context(
            face_results=None,
            emotion=None,
            gesture=_make_gesture(),
            objects=None,
            scene=None,
            buffer_summary=summary,
            session=self.session,
        )
        assert ctx.timestamp >= before

    def test_extended_session_event(self):
        summary = _make_buffer_summary(
            presence_duration_seconds=91 * 60,
            user_id="user_001",
        )
        ctx = self.engine.build_context(
            face_results=None,
            emotion=None,
            gesture=_make_gesture(),
            objects=None,
            scene=None,
            buffer_summary=summary,
            session=self.session,
        )
        assert ctx.event == "extended_session"
