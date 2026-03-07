"""
personalization/personalization_engine.py
─────────────────────────────────────────────────────────────────
Preference learning engine.
Updates user profiles based on interaction outcomes and generates
behavioral instructions for the agent.
"""

from __future__ import annotations

import time
from typing import Optional

from context.context_engine import ContextObject
from personalization.user_profile import UserProfile, ProfileStore
from utils.logger import get_logger

logger = get_logger(__name__)

FEEDBACK_DECAY = 0.95
FEEDBACK_WEIGHT = {"positive": +0.1, "neutral": 0.0, "negative": -0.1}


class PersonalizationEngine:
    """
    Learns user preferences from interaction feedback and generates
    behavior instructions that the agent includes in its prompt.
    """

    def __init__(self, profile_store: ProfileStore):
        self.store = profile_store

    # ── Preference Learning ───────────────────────────────────

    def record_feedback(
        self,
        user_id: str,
        action_type: str,
        outcome: str,          # "positive" | "neutral" | "negative"
        context: ContextObject,
    ) -> None:
        """
        Update action weight for the given user based on outcome.
        Uses exponential moving average with decay.
        """
        profile = self.store.load(user_id)
        if not profile:
            logger.warning(f"No profile found for {user_id} — skipping feedback")
            return

        current = profile.action_weights.get(action_type, 0.5)
        delta = FEEDBACK_WEIGHT.get(outcome, 0.0)
        updated = current * FEEDBACK_DECAY + (0.5 + delta) * (1 - FEEDBACK_DECAY)
        profile.action_weights[action_type] = max(0.0, min(1.0, updated))

        if outcome == "positive":
            profile.positive_interactions += 1
        elif outcome == "negative":
            profile.negative_interactions += 1
        profile.last_interaction_ts = time.time()

        self.store.save(profile)
        logger.debug(f"Feedback recorded for {user_id}: {action_type} → {outcome} "
                     f"(weight now {updated:.2f})")

    def update_behavioral_patterns(
        self,
        user_id: str,
        session_duration_minutes: float,
        dominant_emotion: str,
        dominant_activity: str,
    ) -> None:
        """Update long-term behavioral pattern fields at session end."""
        profile = self.store.load(user_id)
        if not profile:
            return

        # Exponential moving average for session duration
        if profile.average_session_duration_minutes == 0:
            profile.average_session_duration_minutes = session_duration_minutes
        else:
            profile.average_session_duration_minutes = (
                0.8 * profile.average_session_duration_minutes
                + 0.2 * session_duration_minutes
            )

        profile.most_common_emotion = dominant_emotion
        profile.most_common_activity = dominant_activity

        now_hour = time.localtime().tm_hour
        if profile.typical_arrival_hour is None:
            profile.typical_arrival_hour = now_hour

        self.store.save(profile)

    # ── Instruction Generation ────────────────────────────────

    def get_behavior_instructions(
        self,
        profile: Optional[UserProfile],
        context: ContextObject,
    ) -> str:
        """
        Generate natural-language behavioral instructions for the agent
        based on the user's profile and current context.
        """
        if not profile:
            return "This is an unrecognized user. Be welcoming and offer to learn their name."

        instructions: list[str] = []

        # Interaction style
        if profile.preferred_interaction_style == "brief":
            instructions.append("Keep responses short and to the point (1–2 sentences max).")
        elif profile.preferred_interaction_style == "conversational":
            instructions.append("Engage warmly and conversationally. Expand on topics.")

        # Emotional sensitivity
        if not profile.sensitivity_to_emotion:
            instructions.append("Do not comment on the user's emotional state unless directly asked.")
        elif context.emotion in ("sad", "angry", "fearful"):
            instructions.append(
                f"The user appears {context.emotion}. Be gentle, supportive, and non-intrusive."
            )

        # Reminder tolerance
        if profile.reminder_tolerance == "low":
            instructions.append("Avoid sending reminders unless absolutely necessary.")
        elif profile.reminder_tolerance == "high":
            instructions.append("User is receptive to proactive reminders and suggestions.")

        # Humor
        if not profile.humor_preference:
            instructions.append("Avoid humor or playful remarks.")
        else:
            instructions.append("Occasional light humor is welcome.")

        # Action weight guidance
        for action, weight in profile.action_weights.items():
            if weight < 0.25:
                instructions.append(f"Avoid '{action}' interactions — user has responded poorly.")
            elif weight > 0.75:
                instructions.append(f"User responds well to '{action}' interactions.")

        return " | ".join(instructions) if instructions else "Use balanced, friendly behavior."

    def get_profile_for_context(self, context: ContextObject) -> Optional[UserProfile]:
        if not context.user_id:
            return None
        return self.store.load(context.user_id)
