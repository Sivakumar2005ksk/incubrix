"""
Swappable LLM / NLP Intent Adapter (IncuBrix Rule #3 & Must-Have #4).

Principles:
1. An AI model may ONLY translate natural language words into structured operations.
   All timeline math, projection, and word anchoring remain deterministic Python code.
2. Swappable Provider: supports Mock (default, offline, zero API keys), Google Gemini,
   OpenAI, and local Ollama.
3. Zero secrets in code: API keys are loaded strictly from environment variables.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger("incubrix.nlp.llm")


class ParsedIntent(BaseModel):
    """Structured intent representation produced by the AI model."""
    action: str = Field(..., description="Action: add_caption, cut, speed_change, reframe, punch_in, lut, transition, overlay, bgm, silence_removal, filler_removal, auto_caption, undo, redo, restore, modify_caption")
    caption_text: Optional[str] = None
    trigger_phrase: Optional[str] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    speed: Optional[float] = None
    color: Optional[str] = None
    animation: Optional[str] = None
    aspect_ratio: Optional[str] = None
    scale: Optional[float] = None
    target: Optional[str] = None
    topic: Optional[str] = None
    media_path: Optional[str] = None
    confidence: float = 1.0
    reasoning: Optional[str] = None


SYSTEM_PROMPT = """You are an expert video editing intent parser for the IncuBrix Editing Engine.
Your job is ONLY to translate the creator's plain-English instruction into a structured JSON operation.
DO NOT perform clock math. DO NOT build timeline clips. Simply extract the intent and parameters.

Output ONLY valid JSON matching this schema:
{
  "action": "add_caption" | "cut" | "speed_change" | "reframe" | "punch_in" | "lut" | "transition" | "overlay" | "bgm" | "silence_removal" | "filler_removal" | "auto_caption" | "undo" | "redo" | "restore" | "modify_caption",
  "caption_text": string or null,
  "trigger_phrase": string or null,
  "time_start": string or null,
  "time_end": string or null,
  "speed": float or null,
  "color": string or null,
  "animation": string or null,
  "aspect_ratio": string or null,
  "scale": float or null,
  "target": string or null,
  "topic": string or null,
  "media_path": string or null
}

Rules:
- For adding captions (e.g. "when speaker says welcome add caption welcome", "put welcome caption when speaker says welcome"):
  action="add_caption", caption_text="welcome", trigger_phrase="welcome".
- For cutting ranges (e.g. "cut 0:30 to 0:40", "cut the first 30 seconds"):
  action="cut", time_start="0:30", time_end="0:40".
- For cutting topics (e.g. "remove the part about pricing"):
  action="cut", topic="pricing".
- For speed change (e.g. "speed up from 1:00 to 1:30 by 2x"):
  action="speed_change", time_start="1:00", time_end="1:30", speed=2.0.
- For restoring (e.g. "put the intro back", "actually put pricing back"):
  action="restore", target="intro" or target="pricing".
- For aspect ratio (e.g. "convert to 9:16 for TikTok"):
  action="reframe", aspect_ratio="9:16".
- For punch-in/zoom (e.g. "punch in 1.2x on welcome"):
  action="punch_in", scale=1.2, trigger_phrase="welcome".
- For background music (e.g. "add background music"):
  action="bgm".
- For silences (e.g. "remove pauses"):
  action="silence_removal".
- For fillers (e.g. "remove filler words"):
  action="filler_removal".
- For auto captions (e.g. "auto caption whole video"):
  action="auto_caption".
- For styling existing caption (e.g. "make that caption red"):
  action="modify_caption", color="red".
"""


class BaseLLMAdapter(ABC):
    """Abstract interface for swappable LLM intent classifiers."""

    @abstractmethod
    def parse_instruction(self, prompt: str) -> Optional[ParsedIntent]:
        """Translates user natural language prompt into a structured intent."""
        pass


class MockLLMAdapter(BaseLLMAdapter):
    """
    Offline Mock Adapter (Default).
    Ensures 100% determinism and allows all test suites to run with ZERO API keys.
    """

    def parse_instruction(self, prompt: str) -> Optional[ParsedIntent]:
        clean = prompt.strip()

        # Welcome caption match
        if re.search(r"\bwelcome\b", clean, re.I) and (
            re.search(r"\bcaption\b", clean, re.I) or
            re.search(r"\bwhen\b", clean, re.I) or
            re.search(r"\badd\b", clean, re.I)
        ):
            anim = "karaoke" if "karaoke" in clean.lower() else None
            return ParsedIntent(
                action="add_caption",
                caption_text="Welcome!",
                trigger_phrase="welcome",
                animation=anim,
                confidence=1.0
            )

        # Range cuts & trims (e.g. "trim video from 1:00 to 2:00", "cut 0:30 to 0:40")
        cut_range_m = re.search(
            r"(?:cut|remove|delete|trim)\s+(?:out\s+)?(?:(?:the\s+|this\s+)?(?:video|audio|clip|part|section)\s+)?(?:(?:from|between)\s+)?(\d+(?::\d+)?\s*(?:s|sec|seconds)?)\s*(?:to|-|and)\s*(\d+(?::\d+)?\s*(?:s|sec|seconds)?)",
            clean,
            re.I
        )
        if cut_range_m:
            return ParsedIntent(
                action="cut",
                time_start=cut_range_m.group(1),
                time_end=cut_range_m.group(2),
                confidence=1.0
            )

        # Cut intro / first N seconds / trim first N seconds
        cut_first_m = re.search(r"(?:cut|remove|delete|trim)\s+(?:(?:the\s+|this\s+)?(?:video|audio|clip)\s+)?(?:the\s+)?first\s+(\d+)\s*(?:s|sec|seconds)?", clean, re.I)
        if cut_first_m:
            dur = cut_first_m.group(1)
            return ParsedIntent(
                action="cut",
                time_start="0",
                time_end=f"{dur}s",
                confidence=1.0
            )

        # Restore intro
        if re.search(r"put\s+(the\s+)?intro\s+back|restore\s+(the\s+)?intro", clean, re.I):
            return ParsedIntent(
                action="restore",
                target="intro",
                confidence=1.0
            )

        # Background music
        if re.search(r"background\s+music|bgm", clean, re.I):
            return ParsedIntent(
                action="bgm",
                confidence=1.0
            )

        # Silence removal
        if re.search(r"(?:remove|cut|delete)\s+(?:all\s+)?(?:silences?|pauses?)", clean, re.I):
            return ParsedIntent(
                action="silence_removal",
                confidence=1.0
            )

        # Filler removal
        if re.search(r"(?:remove|cut|delete)\s+(?:all\s+)?(?:filler|ums?|filler\s+words?)", clean, re.I):
            return ParsedIntent(
                action="filler_removal",
                confidence=1.0
            )

        # Auto captioning
        if re.search(r"auto[- ]?caption|caption\s+(?:the\s+)?(?:whole\s+)?video|generate\s+captions?\s+from\s+audio", clean, re.I):
            return ParsedIntent(
                action="auto_caption",
                confidence=1.0
            )

        return None


class GeminiLLMAdapter(BaseLLMAdapter):
    """
    Google Gemini Adapter (Gemini 1.5 Flash / Pro).
    Uses lightweight urllib.request with structured JSON output mode.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model

    def parse_instruction(self, prompt: str) -> Optional[ParsedIntent]:
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. Falling back to MockLLMAdapter.")
            return MockLLMAdapter().parse_instruction(prompt)

        m_name = self.model.replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": SYSTEM_PROMPT},
                        {"text": f"User instruction: {prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                parsed_json = json.loads(text)
                return ParsedIntent(**parsed_json)
        except Exception as e:
            logger.error(f"Gemini API error ({e}). Falling back to deterministic matching.")
            return MockLLMAdapter().parse_instruction(prompt)


class OpenAILLMAdapter(BaseLLMAdapter):
    """
    OpenAI Adapter (GPT-4o / GPT-4o-mini).
    Uses lightweight urllib.request with json_object response format.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model

    def parse_instruction(self, prompt: str) -> Optional[ParsedIntent]:
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set. Falling back to MockLLMAdapter.")
            return MockLLMAdapter().parse_instruction(prompt)

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["choices"][0]["message"]["content"]
                parsed_json = json.loads(text)
                return ParsedIntent(**parsed_json)
        except Exception as e:
            logger.error(f"OpenAI API error ({e}). Falling back to deterministic matching.")
            return MockLLMAdapter().parse_instruction(prompt)


class OllamaLLMAdapter(BaseLLMAdapter):
    """
    Local Ollama Adapter (e.g. LLaMA 3, Mistral, Qwen).
    Runs completely on-device without cloud API keys.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def parse_instruction(self, prompt: str) -> Optional[ParsedIntent]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": f"{SYSTEM_PROMPT}\n\nUser instruction: {prompt}",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0}
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data.get("response", "{}")
                parsed_json = json.loads(text)
                return ParsedIntent(**parsed_json)
        except Exception as e:
            logger.warning(f"Ollama local error ({e}). Falling back to deterministic matching.")
            return MockLLMAdapter().parse_instruction(prompt)


def get_llm_adapter(provider_name: Optional[str] = None) -> BaseLLMAdapter:
    """Factory function providing the configured LLM adapter."""
    provider = (provider_name or settings.LLM_PROVIDER or "mock").lower()

    if provider == "gemini":
        return GeminiLLMAdapter(api_key=settings.GEMINI_API_KEY, model=settings.LLM_MODEL)
    elif provider == "openai":
        return OpenAILLMAdapter(api_key=settings.OPENAI_API_KEY, model=settings.LLM_MODEL)
    elif provider == "ollama":
        return OllamaLLMAdapter(base_url=settings.OLLAMA_BASE_URL, model=settings.LLM_MODEL)
    else:
        return MockLLMAdapter()
