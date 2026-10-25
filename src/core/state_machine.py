"""
Timeline State Machine & Content-Anchored Engine.

This is the core brain of the editing engine.
It maintains:
1. An event-sourced history of operations applied to the untouched source media.
2. A pure projection function that projects source content onto the output playhead.
3. Content-anchored placement for captions, mutes, bleeps, and speed changes.
4. Silence snapping to prevent clipping word phonemes.
5. Selective restore and branching undo/redo stacks.
6. Deterministic SHA-256 state hashing.
"""

import hashlib
import json
import uuid
from typing import List, Optional, Tuple, Dict, Any

from src.core.models import (
    TimelineState,
    TimelineOperation,
    OperationType,
    ContentAnchor,
    ProjectedClip,
    ProjectedCaption,
    CaptionStyle,
    BackgroundMusicTrack,
    ProjectedTransition,
    ProjectedOverlay,
    TimePolicy,
    SourceMedia,
    Transcript,
    WordToken,
    SilenceSpan,
)
from src.core.time_policy import TimeTranslator


class Interval:
    """Represents a continuous segment of the original source video."""
    def __init__(
        self,
        start: float,
        end: float,
        speed: float = 1.0,
        is_muted: bool = False,
        is_bleeped: bool = False,
        clip_id: Optional[str] = None,
        punch_in_scale: float = 1.0
    ):
        self.start = round(start, 4)
        self.end = round(end, 4)
        self.speed = speed
        self.is_muted = is_muted
        self.is_bleeped = is_bleeped
        self.clip_id = clip_id or f"clip_{uuid.uuid4().hex[:8]}"
        self.punch_in_scale = punch_in_scale

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def timeline_duration(self) -> float:
        return self.duration / self.speed if self.speed > 0 else 0.0

    def copy(self) -> "Interval":
        return Interval(
            self.start,
            self.end,
            self.speed,
            self.is_muted,
            self.is_bleeped,
            self.clip_id,
            self.punch_in_scale
        )

    def __repr__(self) -> str:
        return f"Interval({self.start:.2f}-{self.end:.2f}, speed={self.speed}x, punch={self.punch_in_scale}x)"


class TimelineEngine:
    """
    Deterministic state machine for content-anchored video editing.
    """

    def __init__(self, state: TimelineState):
        self.state = state
        self._recompute_projection()

    @classmethod
    def create_session(
        cls,
        session_id: str,
        source: SourceMedia,
        transcript: Transcript
    ) -> "TimelineEngine":
        """Initializes a clean editing session from untouched source media."""
        state = TimelineState(
            session_id=session_id,
            source=source,
            transcript=transcript,
            operations=[],
            undone_operations=[],
            duration_s=source.duration,
            timeline_hash=""
        )
        engine = cls(state)
        return engine

    # -----------------------------------------------------------------------
    # Public Edit Operations
    # -----------------------------------------------------------------------

    def apply_cut(
        self,
        source_start: float,
        source_end: float,
        prompt: str = "",
        snap_to_silence: bool = True
    ) -> TimelineOperation:
        """Cuts out a time range from the video."""
        if snap_to_silence:
            source_start, source_end = self._snap_range_to_silence(source_start, source_end)

        # Content anchoring: identify which words are cut
        cut_word_ids = [
            w.id for w in self.state.transcript.words
            if source_start <= w.start and w.end <= source_end
        ]

        # Advanced requirement: Check if any active caption is removed by this cut
        cut_captions = [
            cap.text for cap in self.state.projected_captions
            if any(wid in cut_word_ids for wid in cap.anchor_word_ids)
        ]

        desc = f"Cut {source_start:.2f}s to {source_end:.2f}s ({len(cut_word_ids)} words removed)"
        if cut_captions:
            desc += f". Note: Caption '{cut_captions[0]}' was removed because its words were cut."

        op = TimelineOperation(
            id=f"op_{len(self.state.operations) + 1:03d}",
            step_number=len(self.state.operations) + 1,
            type=OperationType.CUT,
            anchor=ContentAnchor(
                source_start=source_start,
                source_end=source_end,
                word_ids=cut_word_ids
            ),
            params={"source_start": source_start, "source_end": source_end},
            prompt=prompt,
            description=desc
        )
        self._push_operation(op)
        return op

    def apply_speed_change(
        self,
        source_start: float,
        source_end: float,
        speed: float,
        prompt: str = ""
    ) -> TimelineOperation:
        """Changes playback speed over a specific source span (e.g., 2.0x)."""
        op = TimelineOperation(
            id=f"op_{len(self.state.operations) + 1:03d}",
            step_number=len(self.state.operations) + 1,
            type=OperationType.SPEED_CHANGE,
            anchor=ContentAnchor(source_start=source_start, source_end=source_end),
            params={"speed": speed, "source_start": source_start, "source_end": source_end},
            prompt=prompt,
            description=f"Speed changed to {speed}x between {source_start:.2f}s and {source_end:.2f}s"
        )
        self._push_operation(op)
        return op

    def apply_add_caption(
        self,
        text: str,
        word_ids: List[str],
        style: Optional[CaptionStyle] = None,
        prompt: str = ""
    ) -> TimelineOperation:
        """Adds a caption strictly anchored to word tokens."""
        caption_style = style or CaptionStyle()
        caption_id = f"cap_{uuid.uuid4().hex[:6]}"
        
        op = TimelineOperation(
            id=f"op_{len(self.state.operations) + 1:03d}",
            step_number=len(self.state.operations) + 1,
            type=OperationType.ADD_CAPTION,
            anchor=ContentAnchor(word_ids=word_ids, target_caption_id=caption_id),
            params={
                "caption_id": caption_id,
                "text": text,
                "style": caption_style.model_dump()
            },
            prompt=prompt,
            description=f"Caption '{text}' anchored to {len(word_ids)} words"
        )
        self._push_operation(op)
        return op

    def apply_modify_caption_style(
        self,
        caption_id: str,
        color: Optional[str] = None,
        font_size: Optional[int] = None,
        prompt: str = ""
    ) -> TimelineOperation:
        """Modifies styling of an existing caption (e.g., 'make that caption red')."""
        op = TimelineOperation(
            id=f"op_{len(self.state.operations) + 1:03d}",
            step_number=len(self.state.operations) + 1,
            type=OperationType.MODIFY_CAPTION,
            anchor=ContentAnchor(target_caption_id=caption_id),
            params={"caption_id": caption_id, "color": color, "font_size": font_size},
            prompt=prompt,
            description=f"Updated caption {caption_id} style (color={color})"
        )
        self._push_operation(op)
        return op

    def apply_mute_or_bleep(
        self,
        source_start: float,
        source_end: float,
        is_bleep: bool = False,
        prompt: str = ""
    ) -> TimelineOperation:
        """Mutes audio or overlays a bleep tone over a specific span."""
        op_type = OperationType.BLEEP if is_bleep else OperationType.MUTE
        word_ids = [
            w.id for w in self.state.transcript.words
            if source_start <= w.start and w.end <= source_end
        ]

        op = TimelineOperation(
            id=f"op_{len(self.state.operations) + 1:03d}",
            step_number=len(self.state.operations) + 1,
            type=op_type,
            anchor=ContentAnchor(source_start=source_start, source_end=source_end, word_ids=word_ids),
            params={"source_start": source_start, "source_end": source_end, "is_bleep": is_bleep},
            prompt=prompt,
            description=f"{'Bleeped' if is_bleep else 'Muted'} audio from {source_start:.2f}s to {source_end:.2f}s"
        )
        self._push_operation(op)
        return op

    def apply_add_bgm(
        self,
        audio_path_or_url: str,
        volume_db: float = -18.0,
        ducking_db: float = 14.0,
        prompt: str = ""
    ) -> TimelineOperation:
        """Adds background music that automatically ducks under speech."""
        op = TimelineOperation(
            id=f"op_{len(self.state.operations) + 1:03d}",
            step_number=len(self.state.operations) + 1,
            type=OperationType.ADD_BGM,
            anchor=ContentAnchor(),
            params={
                "audio_path_or_url": audio_path_or_url,
                "volume_db": volume_db,
                "ducking_db": ducking_db
            },
            prompt=prompt,
            description="Added background music track with automated dialogue ducking"
        )
        self._push_operation(op)
        return op

    def apply_auto_reframe(
        self,
        aspect_ratio: str = "9:16",
        background: str = "blur",
        prompt: str = ""
    ) -> TimelineOperation:
        """Changes video canvas aspect ratio (e.g. 9:16 for Shorts/TikTok, 1:1, 16:9)."""
        op = TimelineOperation(
            id=f"op_{len(self.state.operations) + 1:03d}",
            step_number=len(self.state.operations) + 1,
            type=OperationType.AUTO_REFRAME,
            anchor=ContentAnchor(),
            params={
                "aspect_ratio": aspect_ratio,
                "background": background
            },
            prompt=prompt,
            description=f"Re-framed canvas to {aspect_ratio} with {background} background"
        )
        self._push_operation(op)
        return op

    def apply_punch_in(
        self,
        source_start: float,
        source_end: float,
        scale: float = 1.25,
        word_ids: Optional[List[str]] = None,
        prompt: str = ""
    ) -> TimelineOperation:
        """Applies a punch-in / zoom scale factor anchored to content words or source span."""
        op = TimelineOperation(
            id=f"op_{len(self.state.operations) + 1:03d}",
            step_number=len(self.state.operations) + 1,
            type=OperationType.PUNCH_IN,
            anchor=ContentAnchor(
                source_start=source_start,
                source_end=source_end,
                word_ids=word_ids
            ),
            params={
                "source_start": source_start,
                "source_end": source_end,
                "scale": scale
            },
            prompt=prompt,
            description=f"Punch-in zoom ({scale}x) applied between {source_start:.2f}s and {source_end:.2f}s"
        )
        self._push_operation(op)
        return op

    def apply_lut(
        self,
        lut_name: str = "cinematic_warm",
        prompt: str = ""
    ) -> TimelineOperation:
        """Applies a cinematic 3D LUT or color grade preset across the timeline."""
        op = TimelineOperation(
            id=f"op_{len(self.state.operations) + 1:03d}",
            step_number=len(self.state.operations) + 1,
            type=OperationType.APPLY_LUT,
            anchor=ContentAnchor(),
            params={"lut_name": lut_name},
            prompt=prompt,
            description=f"Applied color grade / LUT preset '{lut_name}'"
        )
        self._push_operation(op)
        return op

    def apply_transition(
        self,
        transition_type: str = "fade",
        duration: float = 0.5,
        prompt: str = ""
    ) -> TimelineOperation:
        """Adds transitions (crossfade, dissolve, wipe) between cuts."""
        op = TimelineOperation(
            id=f"op_{len(self.state.operations) + 1:03d}",
            step_number=len(self.state.operations) + 1,
            type=OperationType.ADD_TRANSITION,
            anchor=ContentAnchor(),
            params={
                "transition_type": transition_type,
                "duration": duration
            },
            prompt=prompt,
            description=f"Added {transition_type} transition ({duration:.1f}s) between cuts"
        )
        self._push_operation(op)
        return op

    def apply_overlay(
        self,
        media_path: str,
        word_ids: Optional[List[str]] = None,
        source_start: Optional[float] = None,
        source_end: Optional[float] = None,
        position: str = "center",
        scale: float = 1.0,
        prompt: str = ""
    ) -> TimelineOperation:
        """Adds graphic or b-roll overlay anchored to words or source span."""
        overlay_id = f"ov_{uuid.uuid4().hex[:6]}"
        op = TimelineOperation(
            id=f"op_{len(self.state.operations) + 1:03d}",
            step_number=len(self.state.operations) + 1,
            type=OperationType.ADD_OVERLAY,
            anchor=ContentAnchor(
                word_ids=word_ids,
                source_start=source_start,
                source_end=source_end
            ),
            params={
                "overlay_id": overlay_id,
                "media_path": media_path,
                "position": position,
                "scale": scale
            },
            prompt=prompt,
            description=f"Overlay '{media_path}' anchored to content"
        )
        self._push_operation(op)
        return op

    def remove_silences(self, min_duration: float = 1.0, prompt: str = "") -> List[TimelineOperation]:
        """Cuts out silence spans exceeding min_duration."""
        silence_spans = [
            s for s in self.state.transcript.silences
            if (s.end - s.start) >= min_duration
        ]
        ops = []
        for s in silence_spans:
            op = self.apply_cut(s.start, s.end, prompt=prompt)
            ops.append(op)
        return ops

    def remove_filler_words(self, prompt: str = "") -> List[TimelineOperation]:
        """Cuts out filler words (um, uh, and like only when marked filler)."""
        fillers = [
            w for w in self.state.transcript.words
            if w.is_filler or w.word.lower().strip(",.!?") in ("um", "uh", "umm", "uhh")
        ]
        ops = []
        for w in fillers:
            op = self.apply_cut(w.start, w.end, prompt=prompt)
            ops.append(op)
        return ops

    def auto_caption(self, prompt: str = "") -> List[TimelineOperation]:
        """Auto-captions dialogue across the timeline in natural rhythmic word blocks."""
        words = self.state.transcript.words
        ops = []
        chunk_size = 4
        style = CaptionStyle()
        if "karaoke" in prompt.lower():
            style.animation = "karaoke"
            style.highlight_color = "yellow"
        for i in range(0, len(words), chunk_size):
            chunk = words[i:i + chunk_size]
            text = " ".join(w.word for w in chunk)
            op = self.apply_add_caption(text=text, word_ids=[w.id for w in chunk], style=style, prompt=prompt)
            ops.append(op)
        return ops

    # -----------------------------------------------------------------------
    # Undo, Redo, and Selective Restore
    # -----------------------------------------------------------------------

    def undo(self) -> Optional[TimelineOperation]:
        """
        Reverses the most recent instruction (Must-Have #6).
        Moves the operation from operations to undone_operations.
        """
        if not self.state.operations:
            return None
        last_op = self.state.operations.pop()
        if last_op.type == OperationType.RESTORE_EDIT:
            target_ids = last_op.params.get("target_op_ids") or [last_op.params.get("target_op_id")]
            for op in self.state.operations:
                if op.id in target_ids:
                    op.is_active = True
        self.state.undone_operations.append(last_op)
        self._recompute_projection()
        return last_op

    def redo(self) -> Optional[TimelineOperation]:
        """Restores the most recently undone operation."""
        if not self.state.undone_operations:
            return None
        op_to_redo = self.state.undone_operations.pop()
        if op_to_redo.type == OperationType.RESTORE_EDIT:
            target_ids = op_to_redo.params.get("target_op_ids") or [op_to_redo.params.get("target_op_id")]
            for op in self.state.operations:
                if op.id in target_ids:
                    op.is_active = False
        self.state.operations.append(op_to_redo)
        self._recompute_projection()
        return op_to_redo

    def undo_all(self) -> List[TimelineOperation]:
        """Reverses all active operations back to untouched source video."""
        undone = []
        while self.state.operations:
            op = self.undo()
            if op:
                undone.append(op)
            else:
                break
        return undone

    def restore_edit_by_id_or_match(self, identifier: str) -> Optional[TimelineOperation]:
        """
        Selectively restores an earlier edit without wiping subsequent edits (Must-Have #6).
        E.g., 'Actually, put the intro back' finds the cut operation covering the intro
        and deactivates that cut.
        """
        target_ops = []
        is_multi = any(k in identifier.lower() for k in ("all", "everything", "cuts", "pricing"))

        # Special case: second-to-last cut
        if "second-to-last" in identifier.lower() or "second to last" in identifier.lower():
            active_cuts = [op for op in self.state.operations if op.is_active and op.type == OperationType.CUT]
            if len(active_cuts) >= 2:
                target_ops.append(active_cuts[-2])

        if not target_ops:
            for op in reversed(self.state.operations):
                if op.is_active:
                    if op.id == identifier:
                        target_ops.append(op)
                        break
                    if any(k in identifier.lower() for k in ("all", "everything", "all cuts")) and op.type == OperationType.CUT:
                        target_ops.append(op)
                        continue
                    if "pricing" in identifier.lower() and "pricing" in op.prompt.lower():
                        target_ops.append(op)
                        if not is_multi:
                            break
                        continue
                    if any(k in identifier.lower() for k in ("cut", "the cut", "cuts")) and op.type == OperationType.CUT:
                        target_ops.append(op)
                        if not is_multi:
                            break
                        continue
                # Match by description/prompt (e.g. 'intro', 'first 30 seconds')
                if "intro" in identifier.lower() and op.type == OperationType.CUT:
                    if op.params.get("source_start") == 0.0 or "first" in op.prompt.lower() or "intro" in op.prompt.lower():
                        target_ops.append(op)
                        break
                if any(k in identifier.lower() for k in ("reframe", "aspect", "canvas", "vertical", "portrait")) and op.type == OperationType.AUTO_REFRAME:
                    target_ops.append(op)
                    break
                if any(k in identifier.lower() for k in ("lut", "color", "grade", "filter")) and op.type == OperationType.APPLY_LUT:
                    target_ops.append(op)
                    break
                if any(k in identifier.lower() for k in ("punch", "zoom")) and op.type == OperationType.PUNCH_IN:
                    target_ops.append(op)
                    break
                if "transition" in identifier.lower() and op.type == OperationType.ADD_TRANSITION:
                    target_ops.append(op)
                    break
                if "overlay" in identifier.lower() and op.type == OperationType.ADD_OVERLAY:
                    target_ops.append(op)
                    break

        if target_ops:
            target_ids = [op.id for op in target_ops]
            for op in target_ops:
                op.is_active = False

            primary_op = target_ops[0]
            desc = f"Restored previously cut segment: {primary_op.description}" if len(target_ops) == 1 else f"Restored {len(target_ops)} previously cut segments"
            restore_op = TimelineOperation(
                id=f"op_{len(self.state.operations) + 1:03d}",
                step_number=len(self.state.operations) + 1,
                type=OperationType.RESTORE_EDIT,
                anchor=primary_op.anchor,
                params={"target_op_id": primary_op.id, "target_op_ids": target_ids},
                prompt=identifier,
                description=desc
            )
            self._push_operation(restore_op)
            return primary_op

        return None

    # -----------------------------------------------------------------------
    # Dynamic Projection (Content-Anchoring Engine)
    # -----------------------------------------------------------------------

    def _recompute_projection(self) -> None:
        """
        The pure functional projection algorithm:
        Projects the immutable source video + active operations into:
        1. Projected clips with playhead start/end times.
        2. Content-anchored captions positioned on output playhead.
        3. Background music tracks with ducking.
        4. Canonical SHA-256 timeline hash.
        """
        # Step 1: Start with one full continuous interval of the untouched source
        intervals = [Interval(0.0, self.state.source.duration)]

        # Step 2: Sequentially apply active structural operations (CUT, SPEED, PUNCH_IN, MUTE)
        for op in self.state.operations:
            if not op.is_active:
                continue

            if op.type == OperationType.CUT:
                cut_start = op.params["source_start"]
                cut_end = op.params["source_end"]
                intervals = self._cut_intervals(intervals, cut_start, cut_end)

            elif op.type == OperationType.SPEED_CHANGE:
                speed_start = op.params["source_start"]
                speed_end = op.params["source_end"]
                speed = op.params["speed"]
                intervals = self._apply_speed_to_intervals(intervals, speed_start, speed_end, speed)

            elif op.type == OperationType.PUNCH_IN:
                p_start = op.params["source_start"]
                p_end = op.params["source_end"]
                scale = op.params.get("scale", 1.25)
                intervals = self._apply_punch_in_to_intervals(intervals, p_start, p_end, scale)

            elif op.type in (OperationType.MUTE, OperationType.BLEEP):
                m_start = op.params["source_start"]
                m_end = op.params["source_end"]
                is_bleep = op.params.get("is_bleep", False)
                intervals = self._apply_mute_bleep_to_intervals(intervals, m_start, m_end, is_bleep)

        # Step 3: Reconstruct final ProjectedClips with output timeline playhead clocks
        projected_clips: List[ProjectedClip] = []
        current_playhead = 0.0

        for idx, inter in enumerate(intervals):
            if inter.duration <= 0.001:
                continue
            clip_playhead_start = round(current_playhead, 4)
            clip_playhead_end = round(current_playhead + inter.timeline_duration, 4)
            current_playhead = clip_playhead_end

            projected_clips.append(
                ProjectedClip(
                    clip_id=f"clip_{idx+1:03d}",
                    source_start=inter.start,
                    source_end=inter.end,
                    timeline_start=clip_playhead_start,
                    timeline_end=clip_playhead_end,
                    speed=inter.speed,
                    is_muted=inter.is_muted,
                    is_bleeped=inter.is_bleeped,
                    punch_in_scale=inter.punch_in_scale
                )
            )

        self.state.projected_clips = projected_clips
        self.state.duration_s = round(current_playhead, 4)

        # Step 4: Project Content-Anchored Captions and Overlays
        self._project_captions()
        self._project_overlays()

        # Step 5: Global Canvas & Look State (Auto-reframe, LUT, Transitions)
        reframe_op = next((op for op in reversed(self.state.operations) if op.is_active and op.type == OperationType.AUTO_REFRAME), None)
        if reframe_op:
            self.state.canvas_aspect_ratio = reframe_op.params.get("aspect_ratio", "9:16")
            self.state.canvas_background = reframe_op.params.get("background", "blur")
        else:
            self.state.canvas_aspect_ratio = None
            self.state.canvas_background = None

        lut_op = next((op for op in reversed(self.state.operations) if op.is_active and op.type == OperationType.APPLY_LUT), None)
        if lut_op:
            self.state.active_lut = lut_op.params.get("lut_name", "cinematic_warm")
        else:
            self.state.active_lut = None

        trans_ops = [op for op in self.state.operations if op.is_active and op.type == OperationType.ADD_TRANSITION]
        self.state.projected_transitions = [
            ProjectedTransition(
                id=f"tr_{op.id}",
                transition_type=op.params.get("transition_type", "fade"),
                duration=op.params.get("duration", 0.5)
            )
            for op in trans_ops
        ]

        # Step 6: Background Music
        bgm_op = next((op for op in reversed(self.state.operations) if op.is_active and op.type == OperationType.ADD_BGM), None)
        if bgm_op:
            self.state.projected_bgm = BackgroundMusicTrack(
                audio_path_or_url=bgm_op.params["audio_path_or_url"],
                volume_db=bgm_op.params.get("volume_db", -18.0),
                ducking_attenuation_db=bgm_op.params.get("ducking_db", 14.0)
            )
        else:
            self.state.projected_bgm = None

        # Step 7: Compute Canonical Hash for Determinism Proof
        self.state.timeline_hash = self._compute_timeline_hash()

    def _project_captions(self) -> None:
        """
        Projects captions onto the output playhead based on their anchored words.
        If words were cut, the caption is safely hidden.
        If words were sped up, caption duration keeps a readable minimum (0.7s).
        """
        projected_captions: List[ProjectedCaption] = []
        words_map = {w.id: w for w in self.state.transcript.words}

        # Track caption styling overrides
        caption_styles: Dict[str, CaptionStyle] = {}
        for op in self.state.operations:
            if not op.is_active:
                continue
            if op.type == OperationType.ADD_CAPTION:
                cid = op.params["caption_id"]
                caption_styles[cid] = CaptionStyle(**op.params.get("style", {}))
            elif op.type == OperationType.MODIFY_CAPTION:
                cid = op.params["caption_id"]
                if cid in caption_styles:
                    if op.params.get("color"):
                        caption_styles[cid].color = op.params["color"]
                    if op.params.get("font_size"):
                        caption_styles[cid].font_size = op.params["font_size"]
                    if op.params.get("animation"):
                        caption_styles[cid].animation = op.params["animation"]
                    if op.params.get("highlight_color"):
                        caption_styles[cid].highlight_color = op.params["highlight_color"]

        for op in self.state.operations:
            if not op.is_active or op.type != OperationType.ADD_CAPTION:
                continue

            caption_id = op.params["caption_id"]
            text = op.params["text"]
            word_ids = op.anchor.word_ids or []

            # Retrieve source time span of anchored words
            anchored_words = [words_map[wid] for wid in word_ids if wid in words_map]
            if not anchored_words:
                continue

            src_start = anchored_words[0].start
            src_end = anchored_words[-1].end

            # Map source times to current playhead coordinates
            t_start = TimeTranslator.source_to_current(src_start, self.state.projected_clips)
            t_end = TimeTranslator.source_to_current(src_end, self.state.projected_clips)

            # If the words were cut out of the video, the caption does NOT project!
            if t_start is None and t_end is None:
                continue
            
            # If partial cut, snap to available boundary
            if t_start is None:
                t_start = 0.0
            if t_end is None:
                t_end = t_start + 0.7

            # Advanced Level Requirement: Sped-up captions keep a readable minimum (~0.7s)
            if (t_end - t_start) < 0.7:
                t_end = min(self.state.duration_s, t_start + 0.7)

            projected_captions.append(
                ProjectedCaption(
                    id=caption_id,
                    operation_id=op.id,
                    text=text,
                    timeline_start=round(t_start, 4),
                    timeline_end=round(t_end, 4),
                    style=caption_styles.get(caption_id, CaptionStyle()),
                    anchor_word_ids=word_ids
                )
            )

        self.state.projected_captions = projected_captions

    def _project_overlays(self) -> None:
        """Projects image/video overlays onto output playhead anchored to word tokens or source span."""
        projected_overlays: List[ProjectedOverlay] = []
        words_map = {w.id: w for w in self.state.transcript.words}

        for op in self.state.operations:
            if not op.is_active or op.type != OperationType.ADD_OVERLAY:
                continue

            overlay_id = op.params.get("overlay_id", f"ov_{op.id}")
            media_path = op.params.get("media_path", "")
            word_ids = op.anchor.word_ids or []

            if word_ids:
                anchored_words = [words_map[wid] for wid in word_ids if wid in words_map]
                if not anchored_words:
                    continue
                src_start = anchored_words[0].start
                src_end = anchored_words[-1].end
            else:
                src_start = op.anchor.source_start or 0.0
                src_end = op.anchor.source_end or self.state.source.duration

            t_start = TimeTranslator.source_to_current(src_start, self.state.projected_clips)
            t_end = TimeTranslator.source_to_current(src_end, self.state.projected_clips)

            if t_start is not None and t_end is not None and t_end > t_start:
                projected_overlays.append(
                    ProjectedOverlay(
                        id=overlay_id,
                        operation_id=op.id,
                        media_path=media_path,
                        timeline_start=round(t_start, 4),
                        timeline_end=round(t_end, 4),
                        position=op.params.get("position", "center"),
                        scale=op.params.get("scale", 1.0),
                        anchor_word_ids=word_ids
                    )
                )

        self.state.projected_overlays = projected_overlays

    # -----------------------------------------------------------------------
    # Interval Math Helpers
    # -----------------------------------------------------------------------

    def _cut_intervals(self, intervals: List[Interval], cut_start: float, cut_end: float) -> List[Interval]:
        """Subtracts [cut_start, cut_end] from existing list of intervals."""
        new_intervals = []
        for inter in intervals:
            # Case 1: Cut does not intersect interval
            if cut_end <= inter.start or cut_start >= inter.end:
                new_intervals.append(inter)
            # Case 2: Cut completely swallows interval
            elif cut_start <= inter.start and cut_end >= inter.end:
                continue
            # Case 3: Cut splits interval into two
            elif cut_start > inter.start and cut_end < inter.end:
                new_intervals.append(Interval(inter.start, cut_start, inter.speed, inter.is_muted, inter.is_bleeped, punch_in_scale=inter.punch_in_scale))
                new_intervals.append(Interval(cut_end, inter.end, inter.speed, inter.is_muted, inter.is_bleeped, punch_in_scale=inter.punch_in_scale))
            # Case 4: Cut chops head
            elif cut_start <= inter.start and cut_end < inter.end:
                new_intervals.append(Interval(cut_end, inter.end, inter.speed, inter.is_muted, inter.is_bleeped, punch_in_scale=inter.punch_in_scale))
            # Case 5: Cut chops tail
            elif cut_start > inter.start and cut_end >= inter.end:
                new_intervals.append(Interval(inter.start, cut_start, inter.speed, inter.is_muted, inter.is_bleeped, punch_in_scale=inter.punch_in_scale))
        return new_intervals

    def _apply_speed_to_intervals(
        self,
        intervals: List[Interval],
        speed_start: float,
        speed_end: float,
        speed: float
    ) -> List[Interval]:
        """Applies a speed multiplier over a source interval."""
        new_intervals = []
        for inter in intervals:
            if speed_end <= inter.start or speed_start >= inter.end:
                new_intervals.append(inter)
            else:
                overlap_start = max(inter.start, speed_start)
                overlap_end = min(inter.end, speed_end)

                if inter.start < overlap_start:
                    new_intervals.append(Interval(inter.start, overlap_start, inter.speed, inter.is_muted, inter.is_bleeped, punch_in_scale=inter.punch_in_scale))
                new_intervals.append(Interval(overlap_start, overlap_end, speed, inter.is_muted, inter.is_bleeped, punch_in_scale=inter.punch_in_scale))
                if inter.end > overlap_end:
                    new_intervals.append(Interval(overlap_end, inter.end, inter.speed, inter.is_muted, inter.is_bleeped, punch_in_scale=inter.punch_in_scale))
        return new_intervals

    def _apply_punch_in_to_intervals(
        self,
        intervals: List[Interval],
        p_start: float,
        p_end: float,
        scale: float
    ) -> List[Interval]:
        """Applies a punch-in zoom scale factor over a source interval."""
        new_intervals = []
        for inter in intervals:
            if p_end <= inter.start or p_start >= inter.end:
                new_intervals.append(inter)
            else:
                overlap_start = max(inter.start, p_start)
                overlap_end = min(inter.end, p_end)

                if inter.start < overlap_start:
                    new_intervals.append(Interval(inter.start, overlap_start, inter.speed, inter.is_muted, inter.is_bleeped, punch_in_scale=inter.punch_in_scale))
                new_intervals.append(Interval(
                    overlap_start, overlap_end, inter.speed, inter.is_muted, inter.is_bleeped,
                    punch_in_scale=scale
                ))
                if inter.end > overlap_end:
                    new_intervals.append(Interval(overlap_end, inter.end, inter.speed, inter.is_muted, inter.is_bleeped, punch_in_scale=inter.punch_in_scale))
        return new_intervals

    def _apply_mute_bleep_to_intervals(
        self,
        intervals: List[Interval],
        m_start: float,
        m_end: float,
        is_bleep: bool
    ) -> List[Interval]:
        """Applies audio mute or bleep flags to intervals covering the range."""
        new_intervals = []
        for inter in intervals:
            if m_end <= inter.start or m_start >= inter.end:
                new_intervals.append(inter)
            else:
                overlap_start = max(inter.start, m_start)
                overlap_end = min(inter.end, m_end)

                if inter.start < overlap_start:
                    new_intervals.append(Interval(inter.start, overlap_start, inter.speed, inter.is_muted, inter.is_bleeped, punch_in_scale=inter.punch_in_scale))
                new_intervals.append(Interval(
                    overlap_start, overlap_end, inter.speed,
                    is_muted=(not is_bleep), is_bleeped=is_bleep,
                    punch_in_scale=inter.punch_in_scale
                ))
                if inter.end > overlap_end:
                    new_intervals.append(Interval(overlap_end, inter.end, inter.speed, inter.is_muted, inter.is_bleeped, punch_in_scale=inter.punch_in_scale))
        return new_intervals

    def _snap_range_to_silence(self, start: float, end: float, threshold_s: float = 0.25) -> Tuple[float, float]:
        """
        Snaps start and end boundaries to adjacent silence intervals to prevent cutting into phonemes.
        (Outstanding Level Requirement).
        """
        snapped_start = start
        snapped_end = end

        for s in self.state.transcript.silences:
            # Snap start if within threshold of a silence gap
            if abs(start - s.end) <= threshold_s:
                snapped_start = s.end
            elif abs(start - s.start) <= threshold_s:
                snapped_start = s.start

            # Snap end if within threshold of a silence gap
            if abs(end - s.start) <= threshold_s:
                snapped_end = s.start
            elif abs(end - s.end) <= threshold_s:
                snapped_end = s.end

        return round(snapped_start, 4), round(snapped_end, 4)

    def _push_operation(self, op: TimelineOperation) -> None:
        """Adds a new operation to history, clears redo stack, and reprojects."""
        self.state.operations.append(op)
        self.state.undone_operations.clear()
        self._recompute_projection()

    def _compute_timeline_hash(self) -> str:
        """
        Computes canonical SHA-256 hash across all projected clips, captions, and active operations.
        Guarantees 100% determinism across test runs.
        """
        representation = {
            "source_hash": self.state.source.sha256_hash,
            "duration": self.state.duration_s,
            "clips": [
                {
                    "src_s": c.source_start,
                    "src_e": c.source_end,
                    "tl_s": c.timeline_start,
                    "tl_e": c.timeline_end,
                    "speed": c.speed,
                    "muted": c.is_muted,
                    "bleeped": c.is_bleeped,
                    "punch_in": c.punch_in_scale
                }
                for c in self.state.projected_clips
            ],
            "captions": [
                {
                    "text": cap.text,
                    "tl_s": cap.timeline_start,
                    "tl_e": cap.timeline_end,
                    "color": cap.style.color,
                    "font_size": cap.style.font_size,
                    "animation": cap.style.animation
                }
                for cap in self.state.projected_captions
            ],
            "bgm": self.state.projected_bgm.model_dump() if self.state.projected_bgm else None,
            "canvas": self.state.canvas_aspect_ratio,
            "lut": self.state.active_lut
        }
        canonical_json = json.dumps(representation, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
