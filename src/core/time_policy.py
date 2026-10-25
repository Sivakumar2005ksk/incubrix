"""
Time Policy & Coordinate Translation Module.

Resolves timestamps between:
1. Original Source Time (the immutable clock of the untouched raw footage).
2. Current Playback Time (the dynamic clock of the edited output video).

As required by IncuBrix Must-Have #7:
"When a user says 'at 1:20' after earlier cuts, your system uses one stated policy
(original time or current time), everywhere, and says which one in its answer."
"""

from typing import List, Tuple, Optional
from src.core.models import ProjectedClip, TimePolicy


class TimeTranslator:
    """
    Handles bidirectional mapping between source timestamps and projected playhead timestamps.
    """

    @staticmethod
    def current_to_source(
        current_time: float,
        projected_clips: List[ProjectedClip]
    ) -> Optional[float]:
        """
        Translates a timestamp on the current edited playhead back to the original source video timestamp.
        Returns None if current_time exceeds the total edited duration.
        """
        for clip in projected_clips:
            if clip.timeline_start <= current_time <= clip.timeline_end:
                # Offset within the clip on the current timeline
                offset_timeline = current_time - clip.timeline_start
                # Scale by speed (e.g., if speed is 2x, 1 sec on timeline = 2 sec in source)
                offset_source = offset_timeline * clip.speed
                return round(clip.source_start + offset_source, 4)
        
        # If current_time is at the exact boundary of the last clip
        if projected_clips and abs(current_time - projected_clips[-1].timeline_end) < 0.01:
            return projected_clips[-1].source_end

        return None

    @staticmethod
    def source_to_current(
        source_time: float,
        projected_clips: List[ProjectedClip]
    ) -> Optional[float]:
        """
        Translates a timestamp from the original source video to the current edited playhead.
        Returns None if the source timestamp is inside a cut section.
        """
        for clip in projected_clips:
            if clip.source_start <= source_time <= clip.source_end:
                offset_source = source_time - clip.source_start
                offset_timeline = offset_source / clip.speed
                return round(clip.timeline_start + offset_timeline, 4)
        
        return None  # The requested source time was cut out of the video!

    @staticmethod
    def resolve_span(
        start_time: float,
        end_time: float,
        policy: TimePolicy,
        projected_clips: List[ProjectedClip],
        source_duration: float
    ) -> Tuple[float, float, str]:
        """
        Converts given start and end times into canonical source coordinates
        based on the active time policy, and returns a human-readable explanation.
        """
        if policy == TimePolicy.CURRENT_PLAYBACK_TIME:
            src_start = TimeTranslator.current_to_source(start_time, projected_clips)
            src_end = TimeTranslator.current_to_source(end_time, projected_clips)
            
            if src_start is None or src_end is None:
                raise ValueError(
                    f"Requested range [{start_time:.2f}s - {end_time:.2f}s] is out of bounds for the current video."
                )
            
            explanation = (
                f"Resolved using {TimePolicy.CURRENT_PLAYBACK_TIME.value}: "
                f"[{start_time:.2f}s - {end_time:.2f}s] maps to source [{src_start:.2f}s - {src_end:.2f}s]."
            )
            return src_start, src_end, explanation
        
        else:  # ORIGINAL_SOURCE_TIME
            if start_time < 0 or end_time > source_duration:
                raise ValueError(
                    f"Requested range [{start_time:.2f}s - {end_time:.2f}s] exceeds source video duration ({source_duration:.2f}s)."
                )
            explanation = f"Resolved using {TimePolicy.ORIGINAL_SOURCE_TIME.value} [{start_time:.2f}s - {end_time:.2f}s]."
            return start_time, end_time, explanation
