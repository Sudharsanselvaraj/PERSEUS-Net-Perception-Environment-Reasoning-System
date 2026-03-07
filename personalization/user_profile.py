"""
personalization/user_profile.py
─────────────────────────────────────────────────────────────────
User profile data model and persistent storage.
Profiles are stored as JSON files in data/profiles/.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class UserProfile:
    user_id: str
    display_name: str
    enrolled_at: float = field(default_factory=time.time)

    # Interaction style preferences
    preferred_interaction_style: str = "balanced"  # brief | balanced | conversational
    sensitivity_to_emotion: bool = True
    reminder_tolerance: str = "medium"             # high | medium | low
    humor_preference: bool = True
    preferred_topics: List[str] = field(default_factory=list)

    # Learned behavioral patterns
    typical_arrival_hour: Optional[int] = None
    typical_departure_hour: Optional[int] = None
    primary_use_location: str = "desk"
    average_session_duration_minutes: float = 0.0
    most_common_emotion: str = "neutral"
    most_common_activity: str = "idle"

    # Interaction action weights (0.0–1.0, higher = more likely)
    action_weights: Dict[str, float] = field(default_factory=lambda: {
        "greet": 0.8,
        "comment": 0.5,
        "reminder": 0.5,
        "question": 0.4,
        "reaction": 0.7,
    })

    # Feedback counters
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction_ts: Optional[float] = None

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


class ProfileStore:
    """
    JSON file-based profile persistence.
    One file per user: data/profiles/{user_id}.json
    """

    def __init__(self, profile_dir: str = "data/profiles"):
        self._dir = Path(profile_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, UserProfile] = {}

    def save(self, profile: UserProfile) -> None:
        path = self._dir / f"{profile.user_id}.json"
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        self._cache[profile.user_id] = profile

    def load(self, user_id: str) -> Optional[UserProfile]:
        if user_id in self._cache:
            return self._cache[user_id]
        path = self._dir / f"{user_id}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        profile = UserProfile.from_dict(data)
        self._cache[user_id] = profile
        return profile

    def get_or_create(self, user_id: str, display_name: str) -> UserProfile:
        existing = self.load(user_id)
        if existing:
            return existing
        new_profile = UserProfile(user_id=user_id, display_name=display_name)
        self.save(new_profile)
        logger.info(f"Created new profile for {display_name} ({user_id})")
        return new_profile

    def list_users(self) -> List[str]:
        return [p.stem for p in self._dir.glob("*.json")]

    def delete(self, user_id: str) -> bool:
        path = self._dir / f"{user_id}.json"
        if path.exists():
            path.unlink()
            self._cache.pop(user_id, None)
            return True
        return False
