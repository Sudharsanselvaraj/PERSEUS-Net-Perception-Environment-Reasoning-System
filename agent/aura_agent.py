"""
agent/aura_agent.py
─────────────────────────────────────────────────────────────────
Aura AI Agent — LLM-based reasoning and action decision engine.
Receives enriched ContextObject + behavioral instructions,
produces a structured AgentAction via structured LLM output.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from context.context_engine import ContextObject
from config.config import AgentConfig, CooldownConfig
from utils.logger import get_logger
from utils.timing import timed

logger = get_logger(__name__)

ACTION_TYPES = ["greet", "comment", "reminder", "reaction", "question", "silence"]
TONES = ["warm", "playful", "concerned", "professional", "excited", "gentle"]


@dataclass
class AgentAction:
    action_type: str            # One of ACTION_TYPES
    message: Optional[str]
    tone: str
    urgency: str                # immediate | delayed_5s | next_opportunity
    animation: Optional[str]
    reasoning: str


_SYSTEM_TEMPLATE = """\
You are Aura, an empathetic AI companion device with a warm, perceptive personality.
You observe a user through a camera and decide how to respond to their current situation.

Your core traits:
- Observant but never intrusive
- Warm and supportive without being clingy
- Occasionally playful, always appropriate
- Prioritize the user's comfort, focus, and wellbeing above all else

═══════════════════ CURRENT CONTEXT ═══════════════════
{context_string}

═══════════════════ USER BEHAVIORAL PROFILE ═══════════
{behavior_instructions}

═══════════════════ AVAILABLE ACTIONS ═════════════════
- greet: Acknowledge the user's presence warmly
- comment: Observe something relevant about activity/environment
- reminder: Offer a helpful reminder based on time/activity duration
- reaction: React appropriately to a detected gesture
- question: Ask a brief, relevant conversational question
- silence: Take no action (often the correct choice)

═══════════════════ DECISION RULES ════════════════════
1. If the user just arrived or was just recognized for the first time → greet
2. If the user is in deep focus (working 30+ min) → silence (do NOT interrupt)
3. If the user looks distressed and has for > 10 min → gentle comment or question
4. If a meaningful gesture was detected → reaction
5. If the user has been at desk ≥ 90 min with no break → reminder
6. If nothing significant is happening → silence
7. If you already took an action recently → lean strongly toward silence

═══════════════════ RESPONSE FORMAT ═══════════════════
Respond ONLY with a single valid JSON object. No preamble, no markdown fences.
{
  "action_type": "<one of: greet|comment|reminder|reaction|question|silence>",
  "message": "<spoken message, or null for silence>",
  "tone": "<one of: warm|playful|concerned|professional|excited|gentle>",
  "urgency": "<one of: immediate|delayed_5s|next_opportunity>",
  "animation": "<animation name or null>",
  "reasoning": "<internal reasoning in 1–2 sentences>"
}
"""

_SILENCE_ACTION = AgentAction(
    action_type="silence",
    message=None,
    tone="neutral",
    urgency="next_opportunity",
    animation=None,
    reasoning="Default silence — no significant trigger.",
)


class AuraAgent:
    """
    LLM-based agent that decides companion actions from context.
    Supports Ollama (local), Anthropic Claude, and OpenAI backends.
    Enforces per-action cooldowns to prevent interaction spam.
    """

    def __init__(self, agent_config: AgentConfig, cooldown_config: CooldownConfig):
        self.config = agent_config
        self.cooldowns = cooldown_config
        self._last_action_times: Dict[str, float] = {}
        self._action_history: List[AgentAction] = []

    # ── Main Decision Interface ───────────────────────────────

    @timed("agent_decision")
    def decide(self, context: ContextObject, behavior_instructions: str) -> AgentAction:
        """
        Reason about the context and return an action decision.
        Returns silence on LLM error or cooldown hit.
        """
        system_prompt = _SYSTEM_TEMPLATE.format(
            context_string=context.to_prompt_string(),
            behavior_instructions=behavior_instructions,
        )

        try:
            raw_response = self._call_llm(system_prompt)
            action = self._parse_response(raw_response)
        except Exception as e:
            logger.error(f"Agent LLM call failed: {e}")
            return _SILENCE_ACTION

        # Enforce cooldown
        if not self._check_cooldown(action.action_type):
            logger.debug(f"Action '{action.action_type}' in cooldown — silencing")
            return _SILENCE_ACTION

        # Record action
        self._last_action_times[action.action_type] = time.time()
        self._action_history.append(action)
        if len(self._action_history) > 100:
            self._action_history.pop(0)

        logger.info(f"Agent decided: {action.action_type} | "
                    f"tone={action.tone} | urgency={action.urgency}")
        return action

    # ── LLM Backends ─────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        backend = self.config.backend
        if backend == "ollama":
            return self._call_ollama(prompt)
        elif backend == "anthropic":
            return self._call_anthropic(prompt)
        elif backend == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown agent backend: {backend}")

    def _call_ollama(self, prompt: str) -> str:
        import httpx
        response = httpx.post(
            f"{self.config.ollama_base_url}/api/generate",
            json={
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        import os
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=self.config.anthropic_model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _call_openai(self, prompt: str) -> str:
        import openai
        import os
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.config.openai_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    # ── Response Parsing ──────────────────────────────────────

    @staticmethod
    def _parse_response(text: str) -> AgentAction:
        import re
        # Strip markdown code fences
        clean = re.sub(r"```json|```", "", text).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in agent response")

        data = json.loads(match.group())

        action_type = data.get("action_type", "silence")
        if action_type not in ACTION_TYPES:
            action_type = "silence"

        tone = data.get("tone", "warm")
        if tone not in TONES and tone != "neutral":
            tone = "warm"

        return AgentAction(
            action_type=action_type,
            message=data.get("message"),
            tone=tone,
            urgency=data.get("urgency", "next_opportunity"),
            animation=data.get("animation"),
            reasoning=data.get("reasoning", ""),
        )

    # ── Cooldown ─────────────────────────────────────────────

    def _check_cooldown(self, action_type: str) -> bool:
        """Returns True if the action is allowed (not in cooldown window)."""
        cooldown_seconds = getattr(self.cooldowns, action_type, 60)
        last = self._last_action_times.get(action_type, 0)
        return (time.time() - last) >= cooldown_seconds

    def get_last_action(self) -> Optional[AgentAction]:
        return self._action_history[-1] if self._action_history else None
