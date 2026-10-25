"""
Core Domain Models for the Prompt-Based Video Editing Engine.

KEY ARCHITECTURAL PRINCIPLE:
Edits belong to content (words/spans), not fixed clock positions.
This module defines the data structures that represent:
1. Source Media & Ingest Transcript (Immutable ground truth).
2. Content Anchors (Tokens, Silences, and Media Spans).
3. Operations (Non-destructive Event Log).
4. Projected Playhead Timeline (Computed output state).
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class OperationType(str, Enum):
    """Supported video editing operations."""
    CUT = "cut"
    KEEP_ONLY = "keep_only"
    REORDER = "reorder"
    SPEED_CHANGE = "speed_change"
    MUTE = "mute"
    BLEEP = "bleep"
    ADD_CAPTION = "add_caption"
    MODIFY_CAPTION = "modify_caption"
    AUTO_CAPTION = "auto_caption"
    REMOVE_FILLER = "remove_filler"
    REMOVE_SILENCE = "remove_silence"
    ADD_BGM = "add_bgm"
    RESTORE_EDIT = "restore_edit"
    UNDO = "undo"
    REDO = "redo"
    AUTO_REFRAME = "auto_reframe"
    PUNCH_IN = "punch_in"
    APPLY_LUT = "apply_lut"
    ADD_TRANSITION = "add_transition"
    ADD_OVERLAY = "add_overlay"


class StepStatus(str, Enum):
    """The 3 mandatory response statuses required by IncuBrix."""
    APPLIED = "applied"
    NEEDS_CLARIFICATION = "needs_clarification"
    REJECTED = "rejected"


class TimePolicy(str, Enum):
    """Policy used when user specifies a clock time (e.g., 'at 1:20')."""
    CURRENT_PLAYBACK_TIME = "current_playback_time"
    ORIGINAL_SOURCE_TIME = "original_source_time"


# ---------------------------------------------------------------------------
# Source Media & Transcript Models (Immutable Reference)
# ---------------------------------------------------------------------------

class SourceMedia(BaseModel):
    """Metadata describing the untouched original video file."""
    source_url_or_path: str = Field(..., description="Path or URL to the source media")
    sha256_hash: str = Field(..., description="Hash of the untouched file to guarantee non-destructive editing")
    duration: float = Field(..., description="Total duration in seconds")
    fps: float = Field(..., description="Frame rate (frames per second)")
    width: int = Field(default=1920, description="Video width in pixels")
    height: int = Field(default=1080, description="Video height in pixels")
    sample_rate: int = Field(default=48000, description="Audio sample rate in Hz")
    channels: int = Field(default=2, description="Audio channel count")
    is_vfr: bool = Field(default=False, description="Flag indicating variable frame rate")


class WordToken(BaseModel):
    """
    An immutable word token from speech-to-text.
    All content-based edits (captions, mutes, cuts) anchor to this ID.
    """
    id: str = Field(..., description="Unique word identifier, e.g., 'w_001'")
    word: str = Field(..., description="The spoken word text")
    start: float = Field(..., description="Start timestamp in original source media (seconds)")
    end: float = Field(..., description="End timestamp in original source media (seconds)")
    speaker: str = Field(default="speaker_0", description="Speaker label from diarization (e.g., host, guest)")
    confidence: float = Field(default=1.0, description="Speech-to-text confidence score (0.0 to 1.0)")
    is_filler: bool = Field(default=False, description="True if identified as a filler word (um, uh, like)")


class SilenceSpan(BaseModel):
    """Detected period of silence in the audio track."""
    id: str = Field(..., description="Unique silence identifier, e.g., 's_001'")
    start: float = Field(..., description="Start timestamp in source media (seconds)")
    end: float = Field(..., description="End timestamp in source media (seconds)")
    duration: float = Field(..., description="Duration of the silence in seconds")


class Transcript(BaseModel):
    """Complete word-level and silence transcript of the source video."""
    words: List[WordToken] = Field(default_factory=list)
    silences: List[SilenceSpan] = Field(default_factory=list)
    full_text: str = Field(default="", description="Full concatenated transcript text")


# ---------------------------------------------------------------------------
# Content Anchors & Operations
# ---------------------------------------------------------------------------

class ContentAnchor(BaseModel):
    """
    Defines how an operation binds to the video content.
    - If word_ids is provided, the operation sticks to those words regardless of cuts.
    - If source_range is provided, it anchors to immutable source video seconds.
    - If speaker is provided, it applies to speech uttered by that speaker.
    """
    word_ids: Optional[List[str]] = Field(None, description="List of WordToken IDs this edit attaches to")
    source_start: Optional[float] = Field(None, description="Start time in original source video")
    source_end: Optional[float] = Field(None, description="End time in original source video")
    speaker: Optional[str] = Field(None, description="Speaker filter if targeting specific speaker")
    target_caption_id: Optional[str] = Field(None, description="Referenced caption ID when modifying styling")


class TimelineOperation(BaseModel):
    """
    An atomic editing instruction stored in the event log.
    Edits are non-destructive steps stacked on the untouched original.
    """
    id: str = Field(..., description="Unique operation identifier, e.g., 'op_001'")
    step_number: int = Field(..., description="The conversational turn index (1-based)")
    type: OperationType = Field(..., description="Type of operation performed")
    anchor: ContentAnchor = Field(..., description="Content anchor determining placement")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters (e.g., speed factor, caption text)")
    is_active: bool = Field(default=True, description="True if active; False if undone or superseded")
    prompt: str = Field(default="", description="The natural language command that created this operation")
    description: str = Field(default="", description="Human-readable summary of what this operation did")


# ---------------------------------------------------------------------------
# Projected Timeline Models (The Output Coordinates)
# ---------------------------------------------------------------------------

class ProjectedClip(BaseModel):
    """
    A contiguous piece of source video/audio placed on the rendered timeline.
    Generated dynamically by applying active operations onto the source media.
    """
    clip_id: str
    source_start: float = Field(..., description="Where this clip starts in the untouched original file")
    source_end: float = Field(..., description="Where this clip ends in the untouched original file")
    timeline_start: float = Field(..., description="Start position in the rendered output video")
    timeline_end: float = Field(..., description="End position in the rendered output video")
    speed: float = Field(default=1.0, description="Playback speed multiplier (e.g., 2.0x, 0.5x)")
    is_muted: bool = Field(default=False, description="Whether audio is completely silenced")
    is_bleeped: bool = Field(default=False, description="Whether audio is replaced with a bleep tone")
    punch_in_scale: float = Field(default=1.0, description="Punch-in zoom scale factor (e.g., 1.25x)")
    speaker: Optional[str] = None


class CaptionStyle(BaseModel):
    """Visual style attributes for rendered captions."""
    color: str = Field(default="white", description="Caption font color (e.g., white, red, yellow)")
    font_size: int = Field(default=48, description="Font size in points")
    position: str = Field(default="bottom_center", description="Position on screen")
    background_box: bool = Field(default=True, description="Whether to draw a subtle background box for readability")
    animation: str = Field(default="default", description="Caption animation style: 'default', 'karaoke', 'bounce'")
    highlight_color: Optional[str] = Field(default=None, description="Active word highlight color for karaoke captions")


class ProjectedCaption(BaseModel):
    """
    A caption positioned on the output timeline.
    Its timeline_start and timeline_end are computed on-the-fly from its anchored words!
    """
    id: str
    operation_id: str
    text: str
    timeline_start: float = Field(..., description="Computed playhead start time in rendered video")
    timeline_end: float = Field(..., description="Computed playhead end time in rendered video")
    style: CaptionStyle = Field(default_factory=CaptionStyle)
    anchor_word_ids: List[str] = Field(default_factory=list)


class ProjectedTransition(BaseModel):
    """Transition between cuts (e.g. crossfade, dissolve)."""
    id: str
    transition_type: str = Field(default="fade", description="Transition style: fade, dissolve, wipeleft")
    duration: float = Field(default=0.5, description="Transition duration in seconds")


class ProjectedOverlay(BaseModel):
    """Image, graphic, or b-roll overlay placed on the output timeline."""
    id: str
    operation_id: str
    media_path: str
    timeline_start: float = Field(..., description="Playhead start time in rendered video")
    timeline_end: float = Field(..., description="Playhead end time in rendered video")
    position: str = Field(default="center", description="Screen position: center, top_right, etc.")
    scale: float = Field(default=1.0, description="Visual scale of overlay")
    anchor_word_ids: List[str] = Field(default_factory=list)


class BackgroundMusicTrack(BaseModel):
    """Background audio with automated ducking under spoken dialogue."""
    audio_path_or_url: str
    volume_db: float = Field(default=-18.0, description="Base music volume level in dB")
    ducking_attenuation_db: float = Field(default=14.0, description="Amount to lower music during dialogue")


# ---------------------------------------------------------------------------
# Complete Timeline State
# ---------------------------------------------------------------------------

class TimelineState(BaseModel):
    """
    The full state of the project.
    Can be serialized to JSON, hashed for determinism, or exported to OpenTimelineIO.
    """
    session_id: str
    source: SourceMedia
    transcript: Transcript
    operations: List[TimelineOperation] = Field(default_factory=list, description="Active event log")
    undone_operations: List[TimelineOperation] = Field(default_factory=list, description="Stack for redo")
    
    # Dynamically projected outputs (computed from operations + source)
    projected_clips: List[ProjectedClip] = Field(default_factory=list)
    projected_captions: List[ProjectedCaption] = Field(default_factory=list)
    projected_bgm: Optional[BackgroundMusicTrack] = None
    projected_transitions: List[ProjectedTransition] = Field(default_factory=list)
    projected_overlays: List[ProjectedOverlay] = Field(default_factory=list)
    canvas_aspect_ratio: Optional[str] = Field(default=None, description="Canvas aspect ratio: '9:16', '1:1', '16:9'")
    canvas_background: Optional[str] = Field(default=None, description="Canvas background mode: 'blur', 'black'")
    active_lut: Optional[str] = Field(default=None, description="Active color grade / 3D LUT preset")
    
    # State summary metrics
    duration_s: float = Field(default=0.0, description="Total duration of edited timeline")
    timeline_hash: str = Field(default="", description="SHA-256 hash of canonical state for determinism verification")


# ---------------------------------------------------------------------------
# Conversational Step Response (The 3-Way Answer Contract)
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    """
    Contract for every instruction processed by the engine.
    Must return one of: 'applied', 'needs_clarification', or 'rejected'.
    """
    step_number: int
    prompt: str
    status: StepStatus
    
    # Populated when status == APPLIED
    operations: List[str] = Field(default_factory=list, description="List of operation IDs or types performed")
    summary: str = Field(default="", description="Before/after summary, e.g., 'Video is 30s shorter.'")
    time_policy_used: Optional[TimePolicy] = None
    
    # Populated when status == NEEDS_CLARIFICATION
    question: Optional[str] = Field(None, description="Clarifying question when ambiguous")
    options: Optional[List[str]] = Field(None, description="Explicit choices given to the user")
    
    # Populated when status == REJECTED
    rejection_reason: Optional[str] = Field(None, description="Explanation why operation is invalid or impossible")
    
    # Determinism & tracking
    timeline_hash: str = Field(default="", description="Resulting timeline hash after this step")
    duration_s: float = Field(default=0.0, description="Current video duration in seconds")
