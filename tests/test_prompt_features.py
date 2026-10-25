"""
Automated Test Suite for CapCut-Style Prompt-Based Video Editing Features in IncuBrix.

Tests:
1. Auto-Reframe (9:16 vertical canvas with blurred background) via prompt
2. Dynamic Punch-In / Zoom anchored to content phrases via prompt
3. Cinematic LUTs / Color grading presets via prompt
4. Kinetic & Karaoke animated captions via prompt
5. Clip transitions (crossfade, fade) via prompt
6. B-roll / Graphic overlays anchored to spoken words via prompt
7. Undo / Redo and selective restores for all new prompt features
8. End-to-end FFmpeg filtergraph generation with all features active
"""

import pytest
from src.core.models import SourceMedia, StepStatus, OperationType
from src.core.state_machine import TimelineEngine
from src.nlp.intent_compiler import IntentCompiler
from src.render.ffmpeg_renderer import FFmpegRenderer
from src.stt.mock_stt import MockSTTProvider


@pytest.fixture
def test_engine():
    source = SourceMedia(
        source_url_or_path="fixtures/test_video.mp4",
        sha256_hash="dummy_hash_12345",
        duration=300.0,
        fps=30.0,
        width=1920,
        height=1080
    )
    stt = MockSTTProvider()
    transcript = stt.transcribe("fixtures/test_video.mp4", source)
    engine = TimelineEngine.create_session("sess_capcut_test", source, transcript)
    return engine


def test_auto_reframe_via_prompt(test_engine):
    """Test re-framing 16:9 to 9:16 Shorts/TikTok via prompt."""
    compiler = IntentCompiler(test_engine)

    # 1. Apply auto-reframe prompt
    resp = compiler.process_step("Convert this video to 9:16 for TikTok with blurred background")
    assert resp.status == StepStatus.APPLIED
    assert "9:16" in resp.summary
    assert test_engine.state.canvas_aspect_ratio == "9:16"
    assert test_engine.state.canvas_background == "blur"

    # 2. Test undo
    undo_resp = compiler.process_step("Undo")
    assert undo_resp.status == StepStatus.APPLIED
    assert test_engine.state.canvas_aspect_ratio is None

    # 3. Test redo
    redo_resp = compiler.process_step("Redo")
    assert redo_resp.status == StepStatus.APPLIED
    assert test_engine.state.canvas_aspect_ratio == "9:16"


def test_punch_in_zoom_via_prompt(test_engine):
    """Test dynamic punch-in zoom anchored to spoken phrase."""
    compiler = IntentCompiler(test_engine)

    # Apply punch-in on "welcome to the channel"
    resp = compiler.process_step("Punch in 1.2x when I say welcome to the channel")
    assert resp.status == StepStatus.APPLIED
    assert "1.2x" in resp.summary
    assert len(resp.operations) == 1

    # Verify that a projected clip has punch_in_scale applied
    punched_clips = [c for c in test_engine.state.projected_clips if c.punch_in_scale >= 1.2]
    assert len(punched_clips) >= 1
    assert punched_clips[0].punch_in_scale == 1.2


def test_cinematic_lut_and_selective_restore_via_prompt(test_engine):
    """Test applying cinematic LUT and selectively reverting it."""
    compiler = IntentCompiler(test_engine)

    # 1. Apply warm cinematic LUT
    resp = compiler.process_step("Apply a warm cinematic LUT")
    assert resp.status == StepStatus.APPLIED
    assert test_engine.state.active_lut == "cinematic_warm"

    # 2. Add an unrelated edit (cut first 10 seconds)
    cut_resp = compiler.process_step("Cut the first 10 seconds")
    assert cut_resp.status == StepStatus.APPLIED

    # 3. Selectively restore the color/LUT without undoing the cut!
    restore_resp = compiler.process_step("Actually, put the color back")
    assert restore_resp.status == StepStatus.APPLIED
    assert test_engine.state.active_lut is None
    # Cut remains active!
    assert test_engine.state.duration_s < 300.0


def test_karaoke_captions_via_prompt(test_engine):
    """Test generating karaoke-animated captions via natural language prompt."""
    compiler = IntentCompiler(test_engine)

    # 1. Add caption in karaoke style
    resp = compiler.process_step("Add a karaoke caption 'Welcome!' when I say welcome to the channel")
    assert resp.status == StepStatus.APPLIED
    assert len(test_engine.state.projected_captions) == 1

    cap = test_engine.state.projected_captions[0]
    assert cap.style.animation == "karaoke"
    assert cap.style.highlight_color == "yellow"

    # 2. Modify color
    mod_resp = compiler.process_step("Make that caption red")
    assert mod_resp.status == StepStatus.APPLIED
    assert test_engine.state.projected_captions[0].style.color == "red"


def test_transitions_via_prompt(test_engine):
    """Test adding crossfade transitions between cuts."""
    compiler = IntentCompiler(test_engine)

    resp = compiler.process_step("Add a 0.5s crossfade transition between clips")
    assert resp.status == StepStatus.APPLIED
    assert len(test_engine.state.projected_transitions) == 1
    trans = test_engine.state.projected_transitions[0]
    assert trans.transition_type == "crossfade"
    assert trans.duration == 0.5


def test_broll_overlay_anchored_to_words_via_prompt(test_engine):
    """Test overlaying graphic/B-roll anchored to spoken words."""
    compiler = IntentCompiler(test_engine)

    resp = compiler.process_step("Overlay chart.png when I say welcome to the channel")
    assert resp.status == StepStatus.APPLIED
    assert len(test_engine.state.projected_overlays) == 1

    overlay = test_engine.state.projected_overlays[0]
    assert overlay.media_path == "chart.png"
    assert overlay.timeline_start > 0.0
    assert overlay.timeline_end > overlay.timeline_start


def test_ffmpeg_renderer_builds_complex_filtergraph(test_engine):
    """Test that FFmpeg renderer produces a valid filtergraph with all new features active."""
    compiler = IntentCompiler(test_engine)

    # Apply punch-in, karaoke caption, 9:16 reframe, LUT, and overlay
    compiler.process_step("Punch in 1.25x when I say welcome to the channel")
    compiler.process_step("Add a karaoke caption 'Welcome!' when I say welcome to the channel")
    compiler.process_step("Convert to 9:16 with blurred background")
    compiler.process_step("Apply teal and orange color grade")
    compiler.process_step("Overlay logo.png when I say welcome to the channel")

    renderer = FFmpegRenderer()
    filtergraph, v_out, a_out = renderer.build_filtergraph(test_engine.state)

    # Verify key FFmpeg filter blocks are present
    assert "boxblur=25:5" in filtergraph
    assert "scale=1080:1920" in filtergraph
    assert "crop=iw/1.25" in filtergraph
    assert "colorbalance=" in filtergraph
    assert "fontcolor=yellow" in filtergraph
    assert "logo.png" in filtergraph
    assert len(v_out) > 0
    assert len(a_out) > 0


def test_trim_video_via_prompt(test_engine):
    """Test that trim instructions accurately cut ranges and adjust timeline duration."""
    compiler = IntentCompiler(test_engine)
    orig_dur = test_engine.state.duration_s

    # 1. "trim video from 1:00 to 2:00" (cuts 60s from 60s to 120s)
    resp = compiler.process_step("trim video from 1:00 to 2:00")
    assert resp.status == StepStatus.APPLIED
    assert resp.operations
    assert "Trimmed" in resp.summary or "Cut" in resp.summary
    assert test_engine.state.duration_s == orig_dur - 60.0

    # 2. "trim the first 10 seconds"
    dur_after_step1 = test_engine.state.duration_s
    resp2 = compiler.process_step("trim the first 10 seconds")
    assert resp2.status == StepStatus.APPLIED
    assert test_engine.state.duration_s == dur_after_step1 - 10.0

    # 3. "trim last 5s"
    dur_after_step2 = test_engine.state.duration_s
    resp3 = compiler.process_step("trim last 5s")
    assert resp3.status == StepStatus.APPLIED
    assert test_engine.state.duration_s == dur_after_step2 - 5.0

