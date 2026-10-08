"""Transcription interface, a deterministic mock, and pure transcript arithmetic.

Everything that involves timestamps (chunking, dead air) is computed here in code, so the QA
agent never has to do arithmetic on the timeline.
"""

from __future__ import annotations

import asyncio
import itertools
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Protocol

from video_qa.schemas import DeadAirSegment, TranscriptChunk, WordTimestamp


class Transcriber(Protocol):
    """A transcriber may also set ``simulated = True`` when its words are placeholders rather
    than real speech recognition; the dashboard then warns that speech is simulated."""

    async def transcribe(self, audio_path: Path) -> list[WordTimestamp]:
        """Return word-level timestamps for a 16 kHz mono WAV file, in seconds."""
        ...


def is_simulated(transcriber: Transcriber) -> bool:
    return bool(getattr(transcriber, "simulated", False))


DEFAULT_MOCK_SCRIPT = (
    "Stop scrolling if your kitchen never feels big enough. "
    "Here is the one layout trick every renovation in this series uses. "
    "Move the fridge to the end of the run and watch the counter space double. "
    "It costs less than a new tap and takes one weekend. "
    "Follow for part two where we tackle the pantry."
)


class MockTranscriber:
    """Deterministic stand-in for a real speech-to-text service.

    It reads the WAV's real duration and lays a fixed script over it at a natural speaking
    rate: short gaps between words and a longer pause every ``pause_every`` words, so the
    chunking logic has sentence-like breaks to find. The same duration always gives the same
    words and timestamps.
    """

    simulated: ClassVar[bool] = True

    def __init__(
        self,
        script: str = DEFAULT_MOCK_SCRIPT,
        *,
        lead_in_seconds: float = 0.4,
        word_seconds: float = 0.3,
        gap_seconds: float = 0.1,
        pause_every: int = 9,
        pause_seconds: float = 0.7,
        tail_seconds: float = 0.3,
    ) -> None:
        self._words = script.split()
        if not self._words:
            raise ValueError("script must contain at least one word")
        self._lead_in = lead_in_seconds
        self._word = word_seconds
        self._gap = gap_seconds
        self._pause_every = pause_every
        self._pause = pause_seconds
        self._tail = tail_seconds

    async def transcribe(self, audio_path: Path) -> list[WordTimestamp]:
        duration = await asyncio.to_thread(wav_duration_seconds, audio_path)
        return self.words_for_duration(duration)

    def words_for_duration(self, duration_seconds: float) -> list[WordTimestamp]:
        words: list[WordTimestamp] = []
        cursor = self._lead_in
        last_start = duration_seconds - self._tail - self._word
        for count, text in enumerate(itertools.cycle(self._words), start=1):
            if cursor > last_start:
                break
            end = cursor + self._word
            words.append(WordTimestamp(word=text, start=round(cursor, 3), end=round(end, 3)))
            cursor = end + (self._pause if count % self._pause_every == 0 else self._gap)
            if cursor > last_start and end + self._gap <= last_start:
                # A sentence pause here would push the next word past the end and leave a
                # long silent tail, which the dead-air check would report as a real problem.
                cursor = end + self._gap
        return words


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        return wav.getnframes() / rate if rate else 0.0


def group_words_into_chunks(
    words: Sequence[WordTimestamp],
    *,
    pause_seconds: float = 0.6,
    max_chunk_seconds: float = 6.0,
) -> list[TranscriptChunk]:
    """Group words into phrases.

    A new chunk starts after a pause of ``pause_seconds`` or more, or when the next word would
    stretch the current chunk past ``max_chunk_seconds``.
    """
    chunks: list[TranscriptChunk] = []
    current: list[WordTimestamp] = []
    for word in sorted(words, key=lambda w: w.start):
        if current:
            paused = word.start - current[-1].end >= pause_seconds
            too_long = word.end - current[0].start > max_chunk_seconds
            if paused or too_long:
                chunks.append(_chunk(current))
                current = []
        current.append(word)
    if current:
        chunks.append(_chunk(current))
    return chunks


def _chunk(words: Sequence[WordTimestamp]) -> TranscriptChunk:
    return TranscriptChunk(
        start_seconds=words[0].start,
        end_seconds=max(w.end for w in words),
        text=" ".join(w.word for w in words),
    )


def compute_dead_air(
    words: Sequence[WordTimestamp],
    duration_seconds: float,
    *,
    threshold_seconds: float = 1.0,
) -> list[DeadAirSegment]:
    """Silences longer than ``threshold_seconds``: before the first word, between words, and
    after the last word. With no words at all, the whole video is one silence."""
    segments: list[DeadAirSegment] = []
    cursor = 0.0
    for word in sorted(words, key=lambda w: w.start):
        if word.start - cursor > threshold_seconds:
            segments.append(_segment(cursor, word.start))
        cursor = max(cursor, word.end)
    if duration_seconds - cursor > threshold_seconds:
        segments.append(_segment(cursor, duration_seconds))
    return segments


def _segment(start: float, end: float) -> DeadAirSegment:
    return DeadAirSegment(start_seconds=round(start, 3), end_seconds=round(end, 3))
