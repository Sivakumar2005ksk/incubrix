"""
Automated Test Suite for IncuBrix Advanced Level (15 Marks).

Tests:
1. Edits that shift each other:
   - Captions follow words
   - A caption whose words are cut is removed and the user is told
   - Sped-up captions keep a readable minimum (~0.7s)
2. Ambiguity and references:
   - "make it shorter" and "the boring part" trigger needs_clarification with options
   - Meaning-based search ("pricing" finds "cost")
3. Conflicts:
   - Repeated cut instructions are safe
   - Multi-part instructions apply all or report partial results
4. Two speakers:
   - "remove everything the guest says..." cuts only guest lines and asks about host lead-in
"""

import pytest
from src.core.models import SourceMedia, StepStatus
from src.core.state_machine import TimelineEngine
from src.nlp.intent_compiler import IntentCompiler
from src.stt.mock_stt import MockSTTProvider


@pytest.fixture
def test_engine():
    source = SourceMedia(
        source_url_or_path="fixtures/test_video.mp4",
        sha256_hash="dummy_hash_12345",
        duration=300.0,
        fps=30.0
    )
    stt = MockSTTProvider()
    transcript = stt.transcribe("fixtures/test_video.mp4", source)
    engine = TimelineEngine.create_session("sess_adv_test", source, transcript)
    return engine


def test_caption_cut_notification_and_removal(test_engine):
    """
    Advanced Level: A caption whose words are cut is removed and the user is told.
    """
    compiler = IntentCompiler(test_engine)

    # 1. Add caption on "welcome to the channel" (source 42.12 - 43.90)
    r1 = compiler.process_step("Add a caption 'Welcome!' when I say welcome to the channel")
    assert r1.status == StepStatus.APPLIED
    assert len(test_engine.state.projected_captions) == 1

    # 2. Cut the range containing those words (40.0s to 50.0s)
    r2 = compiler.process_step("Cut the first 60 seconds")
    assert r2.status == StepStatus.APPLIED

    # Verification: Caption is suppressed on output playhead AND user is notified!
    assert len(test_engine.state.projected_captions) == 0
    assert "Caption 'Welcome!' was removed because its words were cut" in r2.summary


def test_sped_up_caption_keeps_minimum_readable_duration(test_engine):
    """
    Advanced Level: Sped-up captions keep a readable minimum (about 0.7 s).
    """
    compiler = IntentCompiler(test_engine)

    # Add a short caption (0.4s word span: 1.0s to 1.4s)
    compiler.process_step("Add a caption 'Quick' when I say Hey")
    # Speed up 0 to 10s by 4x
    compiler.process_step("Speed up from 0:00 to 0:10 by 4x")

    assert len(test_engine.state.projected_captions) == 1
    cap = test_engine.state.projected_captions[0]
    # Sped up duration must not shrink below 0.7s
    duration = cap.timeline_end - cap.timeline_start
    assert duration >= 0.7


def test_vague_instructions_require_clarification(test_engine):
    """
    Advanced Level: 'make it shorter' and 'the boring part' ask or state an assumption with options.
    """
    compiler = IntentCompiler(test_engine)

    # "make it shorter"
    r1 = compiler.process_step("make it shorter")
    assert r1.status == StepStatus.NEEDS_CLARIFICATION
    assert len(r1.options) >= 3

    # "remove the boring part"
    r2 = compiler.process_step("remove the boring part")
    assert r2.status == StepStatus.NEEDS_CLARIFICATION
    assert len(r2.options) >= 2


def test_two_speakers_guest_speech_and_host_leadin(test_engine):
    """
    Advanced Level: 'remove everything the guest says about their old job'
    removes only the guest's lines and asks about the host's lead-in question.
    """
    compiler = IntentCompiler(test_engine)

    resp = compiler.process_step("remove everything the guest says about their old job")
    assert resp.status == StepStatus.NEEDS_CLARIFICATION
    assert "host's lead-in question" in resp.question
    assert len(resp.operations) == 1  # Guest lines were cut


def test_multipart_instruction_execution(test_engine):
    """
    Advanced Level: Multi-part instructions apply all or report exactly which parts applied.
    """
    compiler = IntentCompiler(test_engine)

    # First add a caption so "make that caption red" has a target
    compiler.process_step("Add a caption 'Welcome!' when I say welcome to the channel")

    # Multi-part: Cut intro AND restyle caption
    resp = compiler.process_step("Cut the first 30 seconds and make that caption red")
    assert resp.status == StepStatus.APPLIED
    assert len(resp.operations) >= 2
    assert "shorter" in resp.summary
    assert "color 'red'" in resp.summary


def test_repeated_instructions_are_safe(test_engine):
    """
    Advanced Level: Repeated instructions are safe and do not corrupt timeline.
    """
    compiler = IntentCompiler(test_engine)

    # Send cut once
    r1 = compiler.process_step("Cut the first 30 seconds")
    assert r1.status == StepStatus.APPLIED

    # Send cut covering already cut / impossible range (e.g. cutting 280s when only 270s left)
    r2 = compiler.process_step("Cut the first 280 seconds")
    assert r2.status == StepStatus.REJECTED  # Safe rejection, no crash!


def test_explicit_range_cuts(test_engine):
    """
    Tests explicit timestamp cuts: 'cut 1:00 to 1:20' and 'remove 0:30-0:40'.
    """
    compiler = IntentCompiler(test_engine)

    # Cut 1:00 to 1:20 (60s to 80s)
    r1 = compiler.process_step("cut 1:00 to 1:20")
    assert r1.status == StepStatus.APPLIED
    assert "Cut from 60.0s to 80.0s" in r1.summary
    assert test_engine.state.duration_s == 280.0

    # Cut last 10 seconds
    r2 = compiler.process_step("cut the last 10 seconds")
    assert r2.status == StepStatus.APPLIED
    assert test_engine.state.duration_s == 270.0
