"""
Speech-to-Text (STT) Abstraction Layer.

Defines the contract for transcribing audio/video streams into word tokens and silences.
Enables swapping between:
1. Deterministic Mock STT (for offline zero-key CI tests).
2. Local faster-whisper (free, commercial-friendly MIT model).
3. Cloud hosted STT APIs (OpenAI Whisper, Deepgram).
"""

from abc import ABC, abstractmethod
from typing import Optional
from src.core.models import Transcript, SourceMedia


class BaseSTTProvider(ABC):
    """Abstract base class for speech-to-text providers."""

    @abstractmethod
    def transcribe(self, media_path: str, source_media: Optional[SourceMedia] = None) -> Transcript:
        """Transcribes media into word tokens with timestamps, speaker labels, and silences."""
        pass
