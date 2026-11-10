"""Bound PCM delivery lead using sample duration across utterance boundaries."""
from __future__ import annotations

import math


class AudioLeadPacer:
    def __init__(self, lead_seconds: float, bytes_per_second: int = 32000):
        self.lead_seconds = float(lead_seconds)
        if not math.isfinite(self.lead_seconds) or self.lead_seconds <= 0:
            raise ValueError("audio delivery lead must be finite and positive")
        if bytes_per_second <= 0:
            raise ValueError("PCM byte rate must be positive")
        self.bytes_per_second = bytes_per_second
        self._audio_end = None

    def delay(self, byte_count: int, now: float) -> float:
        """Time until this complete frame fits in the channel's lead budget."""
        if byte_count <= 0 or byte_count / self.bytes_per_second > self.lead_seconds:
            raise ValueError("PCM frame must fit within the audio delivery lead")
        end = now if self._audio_end is None else max(now, self._audio_end)
        return max(0.0, end + byte_count / self.bytes_per_second
                   - self.lead_seconds - now)

    def sent(self, byte_count: int, now: float) -> None:
        # Never reset at EOF: a consumer may still be playing the previous
        # utterance. Reanchor only after the already delivered samples expire.
        end = now if self._audio_end is None else max(now, self._audio_end)
        self._audio_end = end + byte_count / self.bytes_per_second
