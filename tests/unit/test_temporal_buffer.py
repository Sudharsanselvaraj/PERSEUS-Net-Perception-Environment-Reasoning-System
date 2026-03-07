"""
tests/unit/test_temporal_buffer.py
─────────────────────────────────────────────────────────────────
Unit tests for the TemporalMemoryBuffer.
"""

import time
import pytest
from memory.temporal_buffer import TemporalMemoryBuffer, PerceptionSnapshot


def _snap(emotion="neutral", activity="idle", present=True,
          objects=None, gesture="none", ts=None) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        timestamp=ts or time.time(),
        user_id="user_001",
        emotion=emotion,
        gesture=gesture,
        detected_objects=objects or [],
        activity=activity,
        body_orientation="facing",
        human_present=present,
        face_confidence=0.85,
    )


class TestTemporalMemoryBuffer:

    def test_buffer_is_initially_empty(self):
        buf = TemporalMemoryBuffer(window_seconds=10)
        assert len(buf._buffer) == 0

    def test_add_and_retrieve(self):
        buf = TemporalMemoryBuffer(window_seconds=10)
        buf.add(_snap("happy"))
        assert len(buf._buffer) == 1

    def test_dominant_emotion_single(self):
        buf = TemporalMemoryBuffer(window_seconds=10)
        buf.add(_snap("happy"))
        assert buf.dominant_emotion == "happy"

    def test_dominant_emotion_majority(self):
        buf = TemporalMemoryBuffer(window_seconds=10)
        for _ in range(5):
            buf.add(_snap("happy"))
        for _ in range(2):
            buf.add(_snap("sad"))
        assert buf.dominant_emotion == "happy"

    def test_emotion_trend_improving(self):
        buf = TemporalMemoryBuffer(window_seconds=60)
        base = time.time() - 30
        for i, emo in enumerate(["sad", "sad", "neutral", "neutral", "happy", "happy"]):
            s = _snap(emo, ts=base + i * 5)
            buf.add(s)
        assert buf.emotion_trend == "improving"

    def test_emotion_trend_stable(self):
        buf = TemporalMemoryBuffer(window_seconds=60)
        base = time.time() - 30
        for i in range(6):
            buf.add(_snap("neutral", ts=base + i * 5))
        assert buf.emotion_trend == "stable"

    def test_primary_activity(self):
        buf = TemporalMemoryBuffer(window_seconds=10)
        for _ in range(4):
            buf.add(_snap(activity="working_at_computer"))
        buf.add(_snap(activity="idle"))
        assert buf.primary_activity == "working_at_computer"

    def test_unique_objects_aggregated(self):
        buf = TemporalMemoryBuffer(window_seconds=10)
        buf.add(_snap(objects=["laptop", "cup"]))
        buf.add(_snap(objects=["cup", "phone"]))
        objs = buf.unique_objects
        assert "laptop" in objs
        assert "cup" in objs
        assert "phone" in objs

    def test_pruning_removes_old_snapshots(self):
        buf = TemporalMemoryBuffer(window_seconds=1)
        past_ts = time.time() - 5
        buf.add(_snap(ts=past_ts))
        buf.add(_snap())  # Current — triggers prune
        assert len(buf._buffer) == 1

    def test_summarize_returns_buffer_summary(self):
        buf = TemporalMemoryBuffer(window_seconds=10)
        buf.add(_snap("happy", activity="working_at_computer"))
        summary = buf.summarize()
        assert summary.dominant_emotion == "happy"
        assert summary.primary_activity == "working_at_computer"
        assert summary.buffer_size == 1
