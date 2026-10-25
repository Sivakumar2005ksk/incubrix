"""
Tests for Swappable LLM Intent Adapter (IncuBrix Rule #3 & Must-Have #4).

Verifies:
1. MockLLMAdapter parses conversational instructions offline with 0 API keys.
2. get_llm_adapter() returns Mock adapter by default.
3. IntentCompiler seamlessly integrates with LLM adapter while keeping timeline math deterministic.
4. Fallback to mock works when API keys are absent.
"""

import pytest
from src.core.models import SourceMedia, StepStatus
from src.core.state_machine import TimelineEngine
from src.nlp.intent_compiler import IntentCompiler
from src.nlp.llm_adapter import (
    BaseLLMAdapter,
    MockLLMAdapter,
    GeminiLLMAdapter,
    OpenAILLMAdapter,
    OllamaLLMAdapter,
    ParsedIntent,
    get_llm_adapter
)
from src.stt.mock_stt import MockSTTProvider


@pytest.fixture
def sample_engine():
    source = SourceMedia(
        source_url_or_path="fixtures/test_video.mp4",
        sha256_hash="sample_sha256",
        duration=300.0,
        fps=30.0
    )
    stt = MockSTTProvider()
    transcript = stt.transcribe("fixtures/test_video.mp4", source)
    return TimelineEngine.create_session("sess_llm_test", source, transcript)


def test_mock_llm_adapter_parsing():
    """Verify MockLLMAdapter offline intent extraction."""
    adapter = MockLLMAdapter()

    # Caption intent
    p1 = adapter.parse_instruction("in adding welcome caption when speaker says welcome that when you add welcome")
    assert p1 is not None
    assert p1.action == "add_caption"
    assert p1.trigger_phrase == "welcome"

    # Cut intent
    p2 = adapter.parse_instruction("cut the first 45 seconds")
    assert p2 is not None
    assert p2.action == "cut"
    assert p2.time_start == "0"
    assert p2.time_end == "45s"

    # Silence removal intent
    p3 = adapter.parse_instruction("please cut all pauses")
    assert p3 is not None
    assert p3.action == "silence_removal"


def test_get_llm_adapter_factory():
    """Verify provider factory returns correct adapter instance."""
    adapter_mock = get_llm_adapter("mock")
    assert isinstance(adapter_mock, MockLLMAdapter)

    adapter_gemini = get_llm_adapter("gemini")
    assert isinstance(adapter_gemini, GeminiLLMAdapter)

    adapter_openai = get_llm_adapter("openai")
    assert isinstance(adapter_openai, OpenAILLMAdapter)

    adapter_ollama = get_llm_adapter("ollama")
    assert isinstance(adapter_ollama, OllamaLLMAdapter)


def test_gemini_openai_graceful_fallback_without_keys():
    """Verify that cloud adapters gracefully fall back to Mock when keys are missing."""
    gemini = GeminiLLMAdapter(api_key="")
    res = gemini.parse_instruction("when speaker says welcome add caption welcome")
    assert res is not None
    assert res.action == "add_caption"

    openai = OpenAILLMAdapter(api_key="")
    res2 = openai.parse_instruction("remove all filler words")
    assert res2 is not None
    assert res2.action == "filler_removal"


def test_intent_compiler_with_custom_llm_adapter(sample_engine):
    """
    Verify that IntentCompiler receives LLM parsed intent and executes
    deterministic timeline operations via TimelineEngine.
    """
    class TestLLM(BaseLLMAdapter):
        def parse_instruction(self, prompt: str):
            return ParsedIntent(
                action="add_caption",
                caption_text="AI Generated Caption",
                trigger_phrase="welcome",
                animation="karaoke"
            )

    compiler = IntentCompiler(sample_engine, llm_adapter=TestLLM())
    resp = compiler.process_step("some completely random conversational sentence that normally fails regex")
    assert resp.status == StepStatus.APPLIED
    assert "AI to 1 words" in resp.summary or "Caption 'AI Generated Caption'" in resp.summary
    assert len(sample_engine.state.projected_captions) == 1
    assert sample_engine.state.projected_captions[0].style.animation == "karaoke"
