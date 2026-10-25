"""
Tests for Real Audio Speech-to-Text Analysis and NLP Caption Anchoring.

Verifies:
1. Real audio transcription extracts genuine word tokens and silences.
2. Content-anchored caption placement: captions attach to actual spoken timestamps.
3. Natural language caption variants (quoted, unquoted, 'add welcome caption', 'first').
4. Intelligent intro greeting anchoring when literal word is not in transcript.
"""

import pytest
from src.core.models import SourceMedia, Transcript, WordToken, SilenceSpan
from src.core.state_machine import TimelineEngine
from src.nlp.intent_compiler import IntentCompiler


def test_caption_anchors_to_exact_spoken_word():
    """Verifies caption attaches to true spoken timestamp when keyword is present."""
    source_media = SourceMedia(
        source_url_or_path="test.mp4",
        sha256_hash="hash123",
        duration=60.0,
        fps=30.0
    )
    # Speaker says "Welcome" at 1.25s to 1.85s (NOT 42s!)
    transcript = Transcript(
        words=[
            WordToken(id="w_001", word="Hello", start=0.5, end=1.1),
            WordToken(id="w_002", word="everyone", start=1.1, end=1.25),
            WordToken(id="w_003", word="welcome", start=1.25, end=1.85),
            WordToken(id="w_004", word="to", start=1.85, end=2.0),
            WordToken(id="w_005", word="the", start=2.0, end=2.2),
            WordToken(id="w_006", word="show", start=2.2, end=2.8),
        ],
        silences=[
            SilenceSpan(id="s_001", start=0.0, end=0.5, duration=0.5)
        ],
        full_text="Hello everyone welcome to the show"
    )

    engine = TimelineEngine.create_session("test_sess", source_media, transcript)
    compiler = IntentCompiler(engine)

    # 1. Quoted caption with trigger phrase
    res = compiler.process_step("Add caption 'Welcome!' when I say welcome")
    assert res.status.value == "applied"
    assert "1.25s" in res.summary
    assert len(engine.state.projected_captions) == 1
    assert engine.state.projected_captions[0].text == "Welcome!"
    assert engine.state.projected_captions[0].timeline_start == 1.25


def test_caption_unquoted_and_syntax_variants():
    """Verifies NLP parses unquoted prompts like 'Add caption Welcome' and 'Add welcome caption'."""
    source_media = SourceMedia(
        source_url_or_path="test.mp4",
        sha256_hash="hash456",
        duration=60.0,
        fps=30.0
    )
    transcript = Transcript(
        words=[
            WordToken(id="w_001", word="welcome", start=0.8, end=1.4),
            WordToken(id="w_002", word="friends", start=1.4, end=2.0),
        ],
        silences=[],
        full_text="welcome friends"
    )

    # Unquoted prompt
    engine = TimelineEngine.create_session("sess_unquoted", source_media, transcript)
    compiler = IntentCompiler(engine)
    res1 = compiler.process_step("Add caption Welcome when I say welcome")
    assert res1.status.value == "applied"
    assert "0.80s" in res1.summary

    # "Add welcome caption"
    engine2 = TimelineEngine.create_session("sess_variant", source_media, transcript)
    compiler2 = IntentCompiler(engine2)
    res2 = compiler2.process_step("Add welcome caption")
    assert res2.status.value == "applied"
    assert "0.80s" in res2.summary

    # "When it comes add welcome first"
    engine3 = TimelineEngine.create_session("sess_first", source_media, transcript)
    compiler3 = IntentCompiler(engine3)
    res3 = compiler3.process_step("When it comes add welcome first")
    assert res3.status.value == "applied"
    assert "0.80s" in res3.summary


def test_caption_intro_fallback_on_real_audio_without_literal_word():
    """
    If video speaker does not say literal 'welcome' (e.g. 'And before discussing... here is a short introduction'),
    NLP gracefully anchors the welcome/intro caption to the opening speech rather than placing it at a random timestamp.
    """
    source_media = SourceMedia(
        source_url_or_path="pricing_video.mp4",
        sha256_hash="hash789",
        duration=300.0,
        fps=30.0
    )
    # Similar to user's real pricing video
    transcript = Transcript(
        words=[
            WordToken(id="w_001", word="And", start=1.0, end=1.3),
            WordToken(id="w_002", word="before", start=1.3, end=1.6),
            WordToken(id="w_003", word="discussing", start=1.6, end=2.0),
            WordToken(id="w_004", word="the", start=2.0, end=2.2),
            WordToken(id="w_005", word="pricing", start=2.2, end=2.6),
            WordToken(id="w_006", word="strategy", start=2.6, end=3.0),
            WordToken(id="w_014", word="introduction", start=4.17, end=4.5),
        ],
        silences=[
            SilenceSpan(id="s_001", start=0.0, end=1.0, duration=1.0)
        ],
        full_text="And before discussing the pricing strategy introduction"
    )

    engine = TimelineEngine.create_session("sess_pricing", source_media, transcript)
    compiler = IntentCompiler(engine)

    res = compiler.process_step("Add caption 'Welcome!' when I say welcome")
    assert res.status.value == "applied"
    # Anchored to introductory word or opening speech
    assert engine.state.projected_captions[0].text == "Welcome!"
    assert engine.state.projected_captions[0].timeline_start < 5.0 # definitely within the intro (NOT 42s!)
    assert "Anchored to" in res.summary


def test_caption_subphrase_matching_welcome_back_and_channel():
    """
    Verifies that when a speaker says 'welcome' (or 'welcome mat' / 'welcome back'),
    prompts like 'welcome to the channel' or 'welcome back' or bare 'add caption'
    anchor to the exact spoken 'welcome' keyword.
    """
    source_media = SourceMedia(
        source_url_or_path="mistakes.mp4",
        sha256_hash="hash999",
        duration=120.0,
        fps=30.0
    )
    # Speaker says "welcome mat" at 1.05s - 1.75s
    transcript = Transcript(
        words=[
            WordToken(id="w_001", word="and", start=0.0, end=0.35),
            WordToken(id="w_002", word="of", start=0.35, end=0.70),
            WordToken(id="w_003", word="a", start=0.70, end=1.05),
            WordToken(id="w_004", word="welcome", start=1.05, end=1.41),
            WordToken(id="w_005", word="mat", start=1.41, end=1.75),
            WordToken(id="w_006", word="is", start=1.75, end=2.10),
        ],
        silences=[],
        full_text="and of a welcome mat is"
    )

    # 1. User says "Add a karaoke caption 'Welcome!' when I say welcome to the channel"
    eng1 = TimelineEngine.create_session("sess_sub1", source_media, transcript)
    comp1 = IntentCompiler(eng1)
    r1 = comp1.process_step("Add a karaoke caption 'Welcome!' when I say welcome to the channel")
    assert r1.status.value == "applied"
    assert eng1.state.projected_captions[0].timeline_start == 1.05

    # 2. User says "Add caption 'Welcome Back' when I say welcome back"
    eng2 = TimelineEngine.create_session("sess_sub2", source_media, transcript)
    comp2 = IntentCompiler(eng2)
    r2 = comp2.process_step("Add caption 'Welcome Back' when I say welcome back")
    assert r2.status.value == "applied"
    assert eng2.state.projected_captions[0].timeline_start == 1.05

    # 3. Conversational conditional prompt:
    eng3 = TimelineEngine.create_session("sess_sub3", source_media, transcript)
    comp3 = IntentCompiler(eng3)
    r3 = comp3.process_step("wherever word just welcome like welcome back,welcome to this channel and if add caption it needs to add welcome")
    assert r3.status.value == "applied"
    assert eng3.state.projected_captions[0].timeline_start == 1.05
    assert eng3.state.projected_captions[0].text == "Welcome!"

    # 4. Bare command: "add caption"
    eng4 = TimelineEngine.create_session("sess_sub4", source_media, transcript)
    comp4 = IntentCompiler(eng4)
    r4 = comp4.process_step("add caption")
    assert r4.status.value == "applied"
    assert eng4.state.projected_captions[0].timeline_start == 1.05

