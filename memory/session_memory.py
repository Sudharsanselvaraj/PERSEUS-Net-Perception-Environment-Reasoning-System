"""
memory/session_memory.py
─────────────────────────────────────────────────────────────────
Session-scoped memory: persists for one device session.
Tracks activity log, emotion log, and interaction history
from device wake until extended absence.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ActivityEntry:
    activity: str
    timestamp: float


@dataclass
class EmotionEntry:
    emotion: str
    valence: str
    timestamp: float


@dataclass
class InteractionEntry:
    interaction_type: str   # greet | comment | reminder | reaction | question
    content: str
    tone: str
    timestamp: float
    user_response: Optional[str] = None  # positive | negative | neutral (filled later)


@dataclass
class SessionMemory:
    """
    In-memory log of everything that happened in the current session.
    Serializable to JSON for persistence in the session database.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    activity_log: List[ActivityEntry] = field(default_factory=list)
    emotion_log: List[EmotionEntry] = field(default_factory=list)
    interaction_log: List[InteractionEntry] = field(default_factory=list)
    scene_descriptions: List[str] = field(default_factory=list)
    notable_events: List[str] = field(default_factory=list)

    # ── Logging ───────────────────────────────────────────────

    def log_activity(self, activity: str) -> None:
        """Log activity only when it changes."""
        if not self.activity_log or self.activity_log[-1].activity != activity:
            self.activity_log.append(ActivityEntry(activity=activity,
                                                    timestamp=time.time()))

    def log_emotion(self, emotion: str, valence: str) -> None:
        self.emotion_log.append(EmotionEntry(emotion=emotion, valence=valence,
                                              timestamp=time.time()))

    def log_interaction(self, interaction_type: str, content: str, tone: str) -> str:
        """Returns interaction_id for later feedback recording."""
        entry = InteractionEntry(
            interaction_type=interaction_type,
            content=content,
            tone=tone,
            timestamp=time.time(),
        )
        self.interaction_log.append(entry)
        return f"{self.session_id}_{len(self.interaction_log)}"

    def record_interaction_response(self, index: int, response: str) -> None:
        """Record user's response (positive/neutral/negative) to an interaction."""
        if 0 <= index < len(self.interaction_log):
            self.interaction_log[index].user_response = response

    def log_scene(self, description: str) -> None:
        if description:
            self.scene_descriptions.append(description)
            if len(self.scene_descriptions) > 20:
                self.scene_descriptions.pop(0)

    def add_notable_event(self, event: str) -> None:
        if event not in self.notable_events:
            self.notable_events.append(event)

    # ── Summaries ─────────────────────────────────────────────

    def get_session_summary(self) -> str:
        duration_min = (time.time() - self.started_at) / 60
        activities = [e.activity for e in self.activity_log]
        dominant_activity = (max(set(activities), key=activities.count)
                              if activities else "idle")
        emotions = [e.emotion for e in self.emotion_log]
        dominant_emotion = (max(set(emotions), key=emotions.count)
                             if emotions else "neutral")
        return (
            f"Session duration: {duration_min:.0f} min. "
            f"Primary activity: {dominant_activity}. "
            f"Dominant mood: {dominant_emotion}. "
            f"Interactions: {len(self.interaction_log)}."
        )

    def get_dominant_activity(self) -> str:
        activities = [e.activity for e in self.activity_log]
        return max(set(activities), key=activities.count) if activities else "idle"

    def get_dominant_emotion(self) -> str:
        emotions = [e.emotion for e in self.emotion_log]
        return max(set(emotions), key=emotions.count) if emotions else "neutral"

    def duration_seconds(self) -> float:
        return time.time() - self.started_at

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "user_id": self.user_id,
            "duration_seconds": self.duration_seconds(),
            "dominant_activity": self.get_dominant_activity(),
            "dominant_emotion": self.get_dominant_emotion(),
            "interaction_count": len(self.interaction_log),
            "notable_events": self.notable_events,
            "summary": self.get_session_summary(),
        }
