"""
memory/temporal_buffer.py
─────────────────────────────────────────────────────────────────
Rolling temporal memory buffer.
Stores PerceptionSnapshots over a configurable time window and
computes smoothed, aggregated summaries for the context engine.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

VALENCE_SCORE: Dict[str, float] = {
    "happy": 1.0,
    "surprised": 0.5,
    "neutral": 0.0,
    "disgusted": -0.5,
    "fearful": -1.0,
    "sad": -1.0,
    "angry": -1.5,
}


@dataclass
class PerceptionSnapshot:
    timestamp: float
    user_id: Optional[str]
    emotion: Optional[str]
    gesture: str
    detected_objects: List[str]
    activity: str
    body_orientation: str
    human_present: bool
    face_confidence: float = 0.0


@dataclass
class BufferSummary:
    dominant_emotion: str
    emotion_trend: str        # improving | stable | declining
    primary_activity: str
    presence_duration_seconds: float
    recent_gestures: List[str]
    unique_objects: List[str]
    buffer_size: int
    user_id: Optional[str]
    notable_events: List[str]


class TemporalMemoryBuffer:
    """
    Rolling time-windowed buffer of perception snapshots.
    Thread-safe via GIL on list operations (Python deque is thread-safe).
    """

    SUSTAINED_NEGATIVE_THRESHOLD_S = 15 * 60   # 15 min
    EXTENDED_SESSION_THRESHOLD_S   = 90 * 60   # 90 min
    LONG_SESSION_THRESHOLD_S       = 6 * 60 * 60  # 6 hr

    def __init__(self, window_seconds: float = 30.0):
        self.window_seconds = window_seconds
        self._buffer: Deque[PerceptionSnapshot] = deque()

    # ── Write ─────────────────────────────────────────────────

    def add(self, snapshot: PerceptionSnapshot) -> None:
        self._buffer.append(snapshot)
        self._prune()

    def _prune(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    # ── Computed Properties ───────────────────────────────────

    @property
    def dominant_emotion(self) -> str:
        emotions = [s.emotion for s in self._buffer if s.emotion]
        if not emotions:
            return "neutral"
        return max(set(emotions), key=emotions.count)

    @property
    def emotion_trend(self) -> str:
        scored = [
            (s.timestamp, VALENCE_SCORE.get(s.emotion or "neutral", 0.0))
            for s in self._buffer
            if s.emotion
        ]
        if len(scored) < 5:
            return "stable"

        times  = np.array([t for t, _ in scored])
        scores = np.array([v for _, v in scored])
        times -= times[0]  # Normalize

        slope = float(np.polyfit(times, scores, 1)[0])
        if slope > 0.05:
            return "improving"
        if slope < -0.05:
            return "declining"
        return "stable"

    @property
    def primary_activity(self) -> str:
        activities = [s.activity for s in self._buffer]
        if not activities:
            return "idle"
        return max(set(activities), key=activities.count)

    @property
    def presence_duration(self) -> float:
        """Seconds user has been continuously present."""
        snapshots = list(self._buffer)
        if not snapshots:
            return 0.0
        latest_ts = snapshots[-1].timestamp
        for i in range(len(snapshots) - 1, -1, -1):
            if not snapshots[i].human_present:
                return latest_ts - snapshots[i + 1].timestamp if i + 1 < len(snapshots) else 0.0
        # All snapshots show presence
        return latest_ts - snapshots[0].timestamp if snapshots else 0.0

    @property
    def recent_gestures(self) -> List[str]:
        recent = list(self._buffer)[-10:]
        return list({s.gesture for s in recent if s.gesture not in ("none", "")})

    @property
    def unique_objects(self) -> List[str]:
        seen: Set[str] = set()
        for s in self._buffer:
            seen.update(s.detected_objects)
        return list(seen)

    @property
    def primary_user_id(self) -> Optional[str]:
        ids = [s.user_id for s in self._buffer if s.user_id]
        if not ids:
            return None
        return max(set(ids), key=ids.count)

    # ── Notable Events ────────────────────────────────────────

    def detect_notable_events(self) -> List[str]:
        events = []
        presence = self.presence_duration

        if self.emotion_trend == "declining" and self.dominant_emotion in ("sad", "angry"):
            # Check if sustained
            negative_count = sum(
                1 for s in self._buffer
                if s.emotion in ("sad", "angry", "fearful")
            )
            if negative_count / max(len(self._buffer), 1) > 0.6:
                events.append("sustained_negative_emotion")

        if presence >= self.EXTENDED_SESSION_THRESHOLD_S:
            events.append(f"extended_session_{int(presence // 60)}min")

        if presence >= self.LONG_SESSION_THRESHOLD_S:
            events.append("no_breaks_detected_long_session")

        return events

    # ── Summary ───────────────────────────────────────────────

    def summarize(self) -> BufferSummary:
        return BufferSummary(
            dominant_emotion=self.dominant_emotion,
            emotion_trend=self.emotion_trend,
            primary_activity=self.primary_activity,
            presence_duration_seconds=self.presence_duration,
            recent_gestures=self.recent_gestures,
            unique_objects=self.unique_objects,
            buffer_size=len(self._buffer),
            user_id=self.primary_user_id,
            notable_events=self.detect_notable_events(),
        )
