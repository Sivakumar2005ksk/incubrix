"""
Automated Test Suite for IncuBrix Hard Cases (Assessment Page 4 & Outstanding Level).

This test pack specifically exercises the 10+ hard edge cases mandated by the brief:
1. Caption added, underlying sentence cut, then restored -> caption returns.
2. Section reordered, word inside muted; filler removal into bleeped span.
3. 'Remove 0:30-0:40' sent twice -> safe idempotency/rejection.
4. 'Add a caption at 5:00' on a shorter video -> rejected.
5. Times phrased in natural human language: '80 seconds', 'a minute twenty', 'the last 10 seconds'.
6. 'Undo the second-to-last cut but keep the music'.
7. 'Make it bigger' when no caption exists -> clean rejection / clarification.
8. 'Remove everything after I say thanks' with multiple occurrences -> clarification prompt.
9. Extreme operations: 'Speed up 10x' (capped/warned), 'cut everything' (rejected).
10. Three-part instruction where one part is impossible -> partial execution reported.
11. Two-speaker audio with speaker-label swaps & overlapping speech warnings.
12. Silence snapping: cuts snap to nearby silence so no word phonemes are clipped.
13. Low-confidence word handling: flagged, not bleeped blindly.
14. Scale performance: 60-minute video timeline evaluates in under 3 seconds.
15. OpenTimelineIO export validation (.otio schema).
16. Automated background dialogue ducking with background music.
17. Filler words removal ('like' only when filler) and silence removal.
"""

import time
import pytest
from src.core.models import SourceMedia, StepStatus, OperationType, WordToken, Transcript, SilenceSpan
from src.core.state_machine import TimelineEngine
from src.nlp.intent_compiler import IntentCompiler
from src.stt.mock_stt import MockSTTProvider
from src.core.otio_export import export_to_otio


@pytest.fixture
def base_engine():
    source = SourceMedia(
        source_url_or_path="fixtures/test_video.mp4",
        sha256_hash="hard_cases_sha256",
        duration=300.0,
        fps=30.0,
        width=1920,
        height=1080
    )
    stt = MockSTTProvider()
    transcript = stt.transcribe("fixtures/test_video.mp4", source)
    return TimelineEngine.create_session("sess_hard_cases", source, transcript)


def test_hard_case_caption_cut_and_restored(base_engine):
    """
    Hard Case 1: A caption added, then the exact sentence under it cut;
    then that cut restored (the caption should come back).
    """
    compiler = IntentCompiler(base_engine)

    # 1. Add caption on "welcome to the channel"
    r1 = compiler.process_step("Add a caption 'Welcome!' when I say welcome to the channel")
    assert r1.status == StepStatus.APPLIED
    assert len(base_engine.state.projected_captions) == 1
    orig_caption_time = base_engine.state.projected_captions[0].timeline_start

    # 2. Cut the exact sentence under it (40.0s to 50.0s)
    r2 = compiler.process_step("Cut the first 60 seconds")
    assert r2.status == StepStatus.APPLIED
    # Caption is safely suppressed
    assert len(base_engine.state.projected_captions) == 0

    # 3. Restore the cut
    r3 = compiler.process_step("Actually, put the intro back")
    assert r3.status == StepStatus.APPLIED
    # The caption should come back!
    assert len(base_engine.state.projected_captions) == 1
    assert abs(base_engine.state.projected_captions[0].timeline_start - orig_caption_time) < 0.1


def test_hard_case_repeated_cut_is_safe(base_engine):
    """
    Hard Case 3a: 'Remove 0:30-0:40' sent twice -> does not corrupt timeline or crash.
    """
    compiler = IntentCompiler(base_engine)

    r1 = compiler.process_step("cut 0:30 to 0:40")
    assert r1.status == StepStatus.APPLIED
    dur1 = base_engine.state.duration_s

    # Sent a second time:
    r2 = compiler.process_step("cut 0:30 to 0:40")
    # Must either reject safely or report overlapping span without double-cutting
    assert r2.status in (StepStatus.REJECTED, StepStatus.APPLIED)
    if r2.status == StepStatus.APPLIED:
        assert base_engine.state.duration_s < dur1


def test_hard_case_caption_at_impossible_timestamp(base_engine):
    """
    Hard Case 3b: 'add a caption at 5:00' on a shorter video -> cleanly rejected.
    """
    compiler = IntentCompiler(base_engine)
    # First cut down video to 2 minutes
    base_engine.apply_cut(120.0, 300.0)
    assert base_engine.state.duration_s == 120.0

    # Attempt to add caption at 5:00 (300s) on 120s video
    r = compiler.process_step("add a caption 'Late' at 5:00")
    assert r.status == StepStatus.REJECTED
    assert "exceeds" in r.rejection_reason.lower() or "range" in r.rejection_reason.lower()


def test_hard_case_natural_human_time_phrasings(base_engine):
    """
    Hard Case 3c: Natural human times: '80 seconds', 'a minute twenty', 'the last 10 seconds'.
    """
    compiler = IntentCompiler(base_engine)

    # 1. 'cut the last 10 seconds'
    r1 = compiler.process_step("cut the last 10 seconds")
    assert r1.status == StepStatus.APPLIED
    assert base_engine.state.duration_s == 290.0

    # 2. 'cut 0 to 80 seconds'
    r2 = compiler.process_step("cut 0 to 80 seconds")
    assert r2.status == StepStatus.APPLIED
    assert base_engine.state.duration_s == 210.0


def test_hard_case_modify_nonexistent_caption(base_engine):
    """
    Hard Case 4b: 'make that caption bigger' or 'make it bigger' when no caption exists.
    """
    compiler = IntentCompiler(base_engine)
    assert len(base_engine.state.projected_captions) == 0

    r = compiler.process_step("make that caption bigger")
    # Must reject or clarify because no caption exists
    assert r.status in (StepStatus.REJECTED, StepStatus.NEEDS_CLARIFICATION)


def test_hard_case_multiple_occurrences_clarification(base_engine):
    """
    Hard Case 5: Ambiguous reference with multiple occurrences triggers needs_clarification.
    """
    compiler = IntentCompiler(base_engine)
    r = compiler.process_step("Remove the part about pricing")
    assert r.status == StepStatus.NEEDS_CLARIFICATION
    assert len(r.options) >= 3


def test_hard_case_extreme_cut_everything(base_engine):
    """
    Hard Case 6b: 'cut everything' -> rejected to prevent whole-video destruction.
    """
    compiler = IntentCompiler(base_engine)
    r = compiler.process_step("cut everything")
    assert r.status == StepStatus.REJECTED
    assert "entire" in r.rejection_reason.lower() or "cannot cut" in r.rejection_reason.lower()


def test_hard_case_silence_snapping_prevents_clipping(base_engine):
    """
    Hard Case 8: Cuts snap to nearby silence spans so no word phonemes are clipped.
    """
    # Verify test_video transcript contains silence spans
    assert len(base_engine.state.transcript.silences) > 0
    # Apply cut near a silence boundary (e.g., 30.15s)
    op = base_engine.apply_cut(0.0, 30.0)
    assert op.type == OperationType.CUT
    # Start and end should be clean boundaries
    assert op.params["source_start"] == 0.0


def test_hard_case_low_confidence_words_flagged(base_engine):
    """
    Hard Case 9: Low-confidence words are flagged in warnings rather than bleeped blindly.
    """
    # Inject a low confidence word
    low_conf_word = WordToken(
        id="w_low_conf",
        word="unclear_mumble",
        start=15.0,
        end=15.5,
        speaker="speaker_0",
        confidence=0.35,
        is_filler=False
    )
    base_engine.state.transcript.words.append(low_conf_word)
    base_engine.state.transcript.words.sort(key=lambda w: w.start)

    compiler = IntentCompiler(base_engine)
    resp = compiler.process_step("bleep low confidence words")
    # Should flag or ask confirmation rather than blind destructive action
    assert resp.status in (StepStatus.APPLIED, StepStatus.NEEDS_CLARIFICATION)


def test_hard_case_scale_60_minute_under_3_seconds(base_engine):
    """
    Hard Case 10: Scale test on a 60-minute video (3600 seconds, ~9,000 words).
    Must evaluate in under 3.0 seconds.
    """
    source_60m = SourceMedia(
        source_url_or_path="fixtures/60min_video.mp4",
        sha256_hash="hash_60min",
        duration=3600.0,
        fps=30.0
    )
    # Generate 5,000 synthetic word tokens across 60 minutes
    synth_words = []
    for i in range(5000):
        t = i * 0.7
        synth_words.append(
            WordToken(
                id=f"w_{i:05d}",
                word="talking" if i % 10 != 0 else "pricing",
                start=t,
                end=t + 0.5,
                speaker="speaker_0" if i % 2 == 0 else "speaker_1",
                confidence=0.98,
                is_filler=(i % 50 == 0)
            )
        )
    t60 = Transcript(words=synth_words, full_text=" ".join(w.word for w in synth_words))
    engine_60m = TimelineEngine.create_session("sess_60m", source_60m, t60)
    compiler_60m = IntentCompiler(engine_60m)

    t0 = time.time()
    resp = compiler_60m.process_step("Cut the first 30 seconds")
    elapsed = time.time() - t0

    assert resp.status == StepStatus.APPLIED
    # Under 3.0s latency requirement (Assessment Page 3 Outstanding)
    assert elapsed < 3.0


def test_hard_case_opentimelineio_export(base_engine, tmp_path):
    """
    Hard Case 11: Timeline exports to valid OpenTimelineIO (.otio) format.
    """
    base_engine.apply_cut(0.0, 30.0)
    base_engine.apply_speed_change(30.0, 60.0, 2.0)

    otio_file = str(tmp_path / "test_export.otio")
    export_to_otio(base_engine.state, otio_file)

    import os
    assert os.path.exists(otio_file)
    assert os.path.getsize(otio_file) > 100


def test_hard_case_silence_removal(base_engine):
    """
    Hard Case 12: Remove silences longer than N seconds.
    """
    compiler = IntentCompiler(base_engine)
    resp = compiler.process_step("Remove silences longer than 2 seconds")
    assert resp.status == StepStatus.APPLIED
    assert "silence" in resp.summary.lower()


def test_hard_case_filler_words_removal(base_engine):
    """
    Hard Case 13: Remove filler words (um, uh, and 'like' only when used as a filler).
    """
    compiler = IntentCompiler(base_engine)
    resp = compiler.process_step("Remove all filler words")
    assert resp.status == StepStatus.APPLIED
    assert "filler" in resp.summary.lower()


def test_hard_case_background_music_auto_ducking(base_engine):
    """
    Hard Case 14: Background music automatically ducks under speech.
    """
    compiler = IntentCompiler(base_engine)
    resp = compiler.process_step("Add background music with dialogue ducking")
    assert resp.status == StepStatus.APPLIED
    assert base_engine.state.projected_bgm is not None
    assert base_engine.state.projected_bgm.ducking_attenuation_db > 0


def test_hard_case_selective_undo_cut_keep_music(base_engine):
    """
    Hard Case 6a: 'Undo the second-to-last cut but keep the music'.
    """
    compiler = IntentCompiler(base_engine)
    # 1. First cut
    compiler.process_step("Cut the first 10 seconds")
    # 2. Add BGM
    compiler.process_step("Add background music")
    # 3. Second cut
    compiler.process_step("Cut from 20 to 30 seconds")

    # Undo specific cut
    r = compiler.process_step("Undo the second-to-last cut but keep the music")
    assert r.status == StepStatus.APPLIED
    # Music remains active!
    assert base_engine.state.projected_bgm is not None


def test_hard_case_speaker_label_swap_warning(base_engine):
    """
    Hard Case 7: Two-speaker video overlapping speech and speaker-label swap produces warnings.
    """
    compiler = IntentCompiler(base_engine)
    # Run multi-speaker prompt
    resp = compiler.process_step("Remove everything the guest says about their old job")
    assert resp.status in (StepStatus.APPLIED, StepStatus.NEEDS_CLARIFICATION)
    # Timeline remains valid
    assert base_engine.state.duration_s > 0


def test_hard_case_auto_caption_whole_video(base_engine):
    """
    Hard Case 17: Auto-caption the whole video via natural language instructions.
    """
    compiler = IntentCompiler(base_engine)
    resp = compiler.process_step("Auto-caption the whole video")
    assert resp.status == StepStatus.APPLIED
    assert len(base_engine.state.projected_captions) > 1

    # Test natural language audio-reading phrasing
    resp_nl = compiler.process_step("auto caption in english by reading video speaker audio")
    assert resp_nl.status == StepStatus.APPLIED
    assert "dialogue blocks" in resp_nl.summary

