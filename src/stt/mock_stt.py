"""
Deterministic Mock Speech-to-Text Provider.

Provides instant, reproducible, zero-key transcription fixtures.
Specifically models:
1. The 8-step worked example from the IncuBrix assessment brief.
2. Word-level timestamps, confidence scores, and speaker diarization.
3. Multi-occurrence keywords (e.g. 'pricing' at 1:05, 2:40, 3:15).
4. Filler words (um, uh, like) and detected silence gaps.
"""

from typing import Optional, List
from src.core.models import Transcript, WordToken, SilenceSpan, SourceMedia
from src.stt.base import BaseSTTProvider


class MockSTTProvider(BaseSTTProvider):
    """
    Offline deterministic STT provider for unit tests, CI pipelines, and screening tests.
    """

    def __init__(self, custom_transcript: Optional[Transcript] = None):
        self._custom = custom_transcript

    def transcribe(self, media_path: str, source_media: Optional[SourceMedia] = None) -> Transcript:
        if self._custom:
            return self._custom

        # Generate realistic default transcript matching the IncuBrix 8-step brief
        words: List[WordToken] = []
        silences: List[SilenceSpan] = []

        # 0:00 - 0:30 Intro monologue
        words.extend([
            WordToken(id="w_001", word="Hey", start=1.0, end=1.4, speaker="speaker_0"),
            WordToken(id="w_002", word="everyone", start=1.4, end=1.9, speaker="speaker_0"),
            WordToken(id="w_003", word="in", start=2.0, end=2.2, speaker="speaker_0"),
            WordToken(id="w_004", word="this", start=2.2, end=2.5, speaker="speaker_0"),
            WordToken(id="w_005", word="video", start=2.5, end=2.9, speaker="speaker_0"),
            WordToken(id="w_006", word="we", start=3.0, end=3.2, speaker="speaker_0"),
            WordToken(id="w_007", word="will", start=3.2, end=3.4, speaker="speaker_0"),
            WordToken(id="w_008", word="discuss", start=3.4, end=3.9, speaker="speaker_0"),
            WordToken(id="w_009", word="our", start=3.9, end=4.2, speaker="speaker_0"),
            WordToken(id="w_010", word="new", start=4.2, end=4.5, speaker="speaker_0"),
            WordToken(id="w_011", word="product", start=4.5, end=5.0, speaker="speaker_0"),
        ])
        silences.append(SilenceSpan(id="s_001", start=5.0, end=6.5, duration=1.5))

        # 0:42 - 0:44 The famous phrase: "welcome to the channel"
        words.extend([
            WordToken(id="w_040", word="So", start=41.5, end=41.9, speaker="speaker_0"),
            WordToken(id="w_041", word="again", start=41.9, end=42.1, speaker="speaker_0"),
            WordToken(id="w_042", word="welcome", start=42.12, end=42.50, speaker="speaker_0"),
            WordToken(id="w_043", word="to", start=42.50, end=42.80, speaker="speaker_0"),
            WordToken(id="w_044", word="the", start=42.80, end=43.10, speaker="speaker_0"),
            WordToken(id="w_045", word="channel", start=43.10, end=43.90, speaker="speaker_0"),
        ])
        silences.append(SilenceSpan(id="s_002", start=44.0, end=45.5, duration=1.5))

        # Pricing Occurrence 1: At ~1:05 (65.0s)
        words.extend([
            WordToken(id="w_060", word="First", start=63.0, end=63.5, speaker="speaker_0"),
            WordToken(id="w_061", word="let's", start=63.5, end=63.9, speaker="speaker_0"),
            WordToken(id="w_062", word="look", start=63.9, end=64.2, speaker="speaker_0"),
            WordToken(id="w_063", word="at", start=64.2, end=64.4, speaker="speaker_0"),
            WordToken(id="w_064", word="initial", start=64.4, end=64.9, speaker="speaker_0"),
            WordToken(id="w_065", word="pricing", start=65.0, end=65.6, speaker="speaker_0"),
            WordToken(id="w_066", word="plans", start=65.6, end=66.1, speaker="speaker_0"),
        ])
        silences.append(SilenceSpan(id="s_003", start=66.2, end=67.5, duration=1.3))

        # Pricing Occurrence 2: At ~2:40 (160.0s)
        words.extend([
            WordToken(id="w_160", word="Now", start=158.0, end=158.4, speaker="speaker_1"),
            WordToken(id="w_161", word="regarding", start=158.4, end=159.0, speaker="speaker_1"),
            WordToken(id="w_162", word="our", start=159.0, end=159.3, speaker="speaker_1"),
            WordToken(id="w_163", word="enterprise", start=159.3, end=159.9, speaker="speaker_1"),
            WordToken(id="w_164", word="pricing", start=160.0, end=160.7, speaker="speaker_1"),
            WordToken(id="w_165", word="model", start=160.7, end=161.2, speaker="speaker_1"),
        ])
        silences.append(SilenceSpan(id="s_004", start=161.5, end=162.8, duration=1.3))

        # Pricing Occurrence 3: At ~3:15 (195.0s)
        words.extend([
            WordToken(id="w_195", word="Finally", start=193.0, end=193.5, speaker="speaker_0"),
            WordToken(id="w_196", word="the", start=193.5, end=193.7, speaker="speaker_0"),
            WordToken(id="w_197", word="discounted", start=193.7, end=194.4, speaker="speaker_0"),
            WordToken(id="w_198", word="pricing", start=195.0, end=195.8, speaker="speaker_0"),
            WordToken(id="w_199", word="summary", start=195.8, end=196.4, speaker="speaker_0"),
        ])

        # Fillers & low confidence words for testing edge cases
        words.extend([
            WordToken(id="w_250", word="um", start=250.0, end=250.4, is_filler=True),
            WordToken(id="w_251", word="uh", start=251.0, end=251.3, is_filler=True),
            WordToken(id="w_252", word="like", start=252.0, end=252.3, is_filler=True),
            WordToken(id="w_300", word="thanks", start=300.0, end=300.5, speaker="speaker_0"),
        ])

        full_text = " ".join(w.word for w in words)
        return Transcript(words=words, silences=silences, full_text=full_text)
