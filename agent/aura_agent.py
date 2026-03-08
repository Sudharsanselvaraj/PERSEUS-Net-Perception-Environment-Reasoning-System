from __future__ import annotations
import json, time, re
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
    action_type: str
    message: Optional[str]
    tone: str
    urgency: str
    animation: Optional[str]
    reasoning: str

_SILENCE_ACTION = AgentAction(action_type="silence", message=None, tone="neutral", urgency="next_opportunity", animation=None, reasoning="Default silence.")

_PROMPT = """You are Aura, an AI companion. Respond ONLY with a JSON object, no other text.
Context: {context_string}
Profile: {behavior_instructions}
Rules: greet if user just arrived, silence if focused, reminder if 90+ min at desk.
Required JSON fields: action_type (greet/comment/reminder/reaction/question/silence), message (string or null), tone (warm/playful/concerned/professional/excited/gentle), urgency (immediate/delayed_5s/next_opportunity), animation (string or null), reasoning (string)
Example: {"action_type":"greet","message":"Hello!","tone":"warm","urgency":"immediate","animation":null,"reasoning":"User arrived."}"""

class AuraAgent:
    def __init__(self, agent_config: AgentConfig, cooldown_config: CooldownConfig):
        self.config = agent_config
        self.cooldowns = cooldown_config
        self._last_action_times: Dict[str, float] = {}
        self._action_history: List[AgentAction] = []

    @timed("agent_decision")
    def decide(self, context: ContextObject, behavior_instructions: str) -> AgentAction:
        prompt = _PROMPT.format(context_string=context.to_prompt_string(), behavior_instructions=behavior_instructions)
        try:
            raw = self._call_llm(prompt)
            action = self._parse(raw)
        except Exception as e:
            logger.error(f"Agent LLM call failed: {e}")
            return _SILENCE_ACTION
        if not self._check_cooldown(action.action_type):
            return _SILENCE_ACTION
        self._last_action_times[action.action_type] = time.time()
        self._action_history.append(action)
        logger.info(f"Agent decided: {action.action_type} | tone={action.tone}")
        return action

    def _call_llm(self, prompt: str) -> str:
        if self.config.backend == "ollama": return self._ollama(prompt)
        if self.config.backend == "anthropic": return self._anthropic(prompt)
        if self.config.backend == "openai": return self._openai(prompt)
        raise ValueError(f"Unknown backend: {self.config.backend}")

    def _ollama(self, prompt: str) -> str:
        import httpx
        r = httpx.post(f"{self.config.ollama_base_url}/api/generate",
            json={"model": self.config.ollama_model, "prompt": prompt, "stream": False, "format": "json",
                  "options": {"temperature": self.config.temperature, "num_predict": self.config.max_tokens}},
            timeout=self.config.timeout_seconds)
        r.raise_for_status()
        result = r.json().get("response", "")
        logger.debug(f"Ollama raw response: {repr(result[:200])}")
        return result

    def _anthropic(self, prompt: str) -> str:
        import anthropic, os
        c = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        m = c.messages.create(model=self.config.anthropic_model, max_tokens=self.config.max_tokens, messages=[{"role":"user","content":prompt}])
        return m.content[0].text

    def _openai(self, prompt: str) -> str:
        import openai, os
        c = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        r = c.chat.completions.create(model=self.config.openai_model, max_tokens=self.config.max_tokens, temperature=self.config.temperature, messages=[{"role":"user","content":prompt}])
        return r.choices[0].message.content or ""

    def _parse(self, text: str) -> AgentAction:
        if not text or not text.strip():
            return _SILENCE_ACTION
        clean = re.sub(r"```json\s*|```", "", text).strip()
        s, e = clean.find('{'), clean.rfind('}')
        if s == -1 or e == -1:
            logger.warning(f"No JSON in response: {text[:100]}")
            return _SILENCE_ACTION
        try:
            data = json.loads(clean[s:e+1])
        except Exception:
            try:
                fixed = re.sub(r',(\s*[}\]])', r'\1', clean[s:e+1]).replace("'", '"')
                data = json.loads(fixed)
            except Exception as err:
                logger.warning(f"JSON parse failed: {err} | text: {text[:200]}")
                return _SILENCE_ACTION
        at = data.get("action_type", "silence")
        if at not in ACTION_TYPES: at = "silence"
        tone = data.get("tone", "warm")
        if tone not in TONES: tone = "warm"
        return AgentAction(action_type=at, message=data.get("message"), tone=tone,
                           urgency=data.get("urgency","next_opportunity"),
                           animation=data.get("animation"), reasoning=data.get("reasoning",""))

    def _check_cooldown(self, action_type: str) -> bool:
        return (time.time() - self._last_action_times.get(action_type, 0)) >= getattr(self.cooldowns, action_type, 60)

    def get_last_action(self) -> Optional[AgentAction]:
        return self._action_history[-1] if self._action_history else None
