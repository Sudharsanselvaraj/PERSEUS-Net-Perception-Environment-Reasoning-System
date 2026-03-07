"""
tests/unit/test_agent.py
─────────────────────────────────────────────────────────────────
Unit tests for the AuraAgent with mocked LLM backends.
"""

import time
import json
import pytest
from unittest.mock import patch, MagicMock

from agent.aura_agent import AuraAgent, AgentAction
from config.config import AgentConfig, CooldownConfig
from context.context_engine import ContextObject


def _make_config(backend="ollama"):
    return AgentConfig(
        backend=backend,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1:8b",
        max_tokens=256,
        temperature=0.7,
        timeout_seconds=5.0,
    )


def _make_cooldowns():
    return CooldownConfig(
        greet=0, reminder=0, comment=0,
        question=0, reaction=0, silence=0,
    )


def _make_context(event="user_present", emotion="neutral"):
    return ContextObject(
        user_id="user_001",
        user_display_name="Alex",
        identity_confidence=0.88,
        emotion=emotion,
        emotion_scores={emotion: 1.0},
        emotion_trend="stable",
        valence="neutral",
        activity="idle",
        gesture="none",
        body_orientation="facing",
        detected_objects=[],
        scene_description="",
        environment_notes="",
        presence_duration_seconds=60.0,
        session_summary="Session started.",
        time_of_day="morning",
        event=event,
        notable_events=[],
        timestamp=time.time(),
    )


def _make_valid_response(action_type="greet") -> str:
    return json.dumps({
        "action_type": action_type,
        "message": "Hello Alex!",
        "tone": "warm",
        "urgency": "immediate",
        "animation": "wave_back",
        "reasoning": "User just arrived.",
    })


class TestAuraAgent:

    def setup_method(self):
        self.agent = AuraAgent(_make_config(), _make_cooldowns())

    def _mock_llm_response(self, response_text: str):
        """Patch the internal _call_llm to return a fixed string."""
        self.agent._call_llm = lambda prompt: response_text

    def test_greet_action_parsed(self):
        self._mock_llm_response(_make_valid_response("greet"))
        ctx = _make_context(event="user_waving")
        action = self.agent.decide(ctx, "Be warm and friendly.")
        assert action.action_type == "greet"
        assert action.message == "Hello Alex!"
        assert action.tone == "warm"

    def test_silence_action_passes_through(self):
        self._mock_llm_response(_make_valid_response("silence"))
        ctx = _make_context()
        action = self.agent.decide(ctx, "")
        # silence still returns AgentAction with action_type="silence"
        assert action.action_type == "silence"

    def test_invalid_json_returns_silence(self):
        self._mock_llm_response("This is not valid JSON at all!")
        ctx = _make_context()
        action = self.agent.decide(ctx, "")
        assert action.action_type == "silence"

    def test_invalid_action_type_returns_silence(self):
        bad = json.dumps({
            "action_type": "dance",  # Not in ACTION_TYPES
            "message": "Let's dance!",
            "tone": "playful",
            "urgency": "immediate",
            "animation": None,
            "reasoning": "Test",
        })
        self._mock_llm_response(bad)
        ctx = _make_context()
        action = self.agent.decide(ctx, "")
        # Invalid type normalizes to silence
        assert action.action_type == "silence"

    def test_cooldown_prevents_repeat_greet(self):
        # Use real cooldowns (non-zero)
        agent = AuraAgent(_make_config(), CooldownConfig(greet=300))
        agent._call_llm = lambda p: _make_valid_response("greet")
        ctx = _make_context()

        # First call succeeds
        action1 = agent.decide(ctx, "")
        assert action1.action_type == "greet"

        # Second call should be silenced by cooldown
        action2 = agent.decide(ctx, "")
        assert action2.action_type == "silence"

    def test_action_history_recorded(self):
        self._mock_llm_response(_make_valid_response("comment"))
        ctx = _make_context()
        self.agent.decide(ctx, "")
        assert self.agent.get_last_action() is not None
        assert self.agent.get_last_action().action_type == "comment"
