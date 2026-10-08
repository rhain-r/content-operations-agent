from __future__ import annotations

import wave
from itertools import pairwise
from pathlib import Path

from video_qa.transcription import (
    MockTranscriber,
    compute_dead_air,
    group_words_into_chunks,
)

from helpers import segment, words


def write_silent_wav(path: Path, seconds: float, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))


async def test_mock_transcriber_spreads_words_over_real_duration(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    write_silent_wav(audio, 12.0)

    first = await MockTranscriber().transcribe(audio)
    second = await MockTranscriber().transcribe(audio)

    assert first == second  # deterministic
    assert first[0].word == "Stop"
    assert first[0].start == 0.4
    assert first[-1].end <= 12.0 - 0.3
    assert first[-1].end > 12.0 - 1.0  # fills the clip rather than bunching at the start
    assert all(a.end <= b.start for a, b in pairwise(first))
    assert compute_dead_air(first, 12.0) == []


def test_mock_transcriber_never_invents_dead_air() -> None:
    # Placeholder speech must not trip the silence check on its own, whatever the length.
    mock = MockTranscriber()
    for tenths in range(20, 601):
        duration = tenths / 10
        assert compute_dead_air(mock.words_for_duration(duration), duration) == [], duration


def test_mock_transcriber_handles_clips_too_short_for_a_word() -> None:
    assert MockTranscriber().words_for_duration(0.5) == []


def test_chunks_break_on_pauses_of_at_least_0_6_seconds() -> None:
    spoken = words(
        ("a", 0.0, 0.3),
        ("b", 0.89, 1.2),  # 0.59 s gap: same chunk
        ("c", 1.8, 2.0),  # 0.6 s gap: new chunk
    )
    chunks = group_words_into_chunks(spoken)
    assert [(c.text, c.start_seconds, c.end_seconds) for c in chunks] == [
        ("a b", 0.0, 1.2),
        ("c", 1.8, 2.0),
    ]


def test_chunks_never_exceed_six_seconds() -> None:
    spoken = [w for i in range(20) for w in words((f"w{i}", i * 0.5, i * 0.5 + 0.4))]
    chunks = group_words_into_chunks(spoken)
    assert all(c.end_seconds - c.start_seconds <= 6.0 for c in chunks)
    assert [c.text.split()[0] for c in chunks] == ["w0", "w12"]
    assert " ".join(c.text for c in chunks) == " ".join(w.word for w in spoken)


def test_dead_air_before_between_and_after_words() -> None:
    spoken = words(("hi", 1.5, 1.8), ("there", 1.9, 2.2), ("again", 4.0, 4.3))
    assert compute_dead_air(spoken, 6.0) == [
        segment(0.0, 1.5),
        segment(2.2, 4.0),
        segment(4.3, 6.0),
    ]


def test_dead_air_threshold_is_strictly_longer_than_one_second() -> None:
    spoken = words(("a", 1.0, 1.5), ("b", 2.5, 3.0))
    assert compute_dead_air(spoken, 4.0) == []


def test_dead_air_handles_overlapping_words() -> None:
    spoken = words(("long", 0.2, 3.0), ("short", 0.5, 0.8), ("next", 3.5, 3.9))
    assert compute_dead_air(spoken, 4.0) == []


def test_no_words_means_the_whole_video_is_dead_air() -> None:
    assert compute_dead_air([], 7.25) == [segment(0.0, 7.25)]
    assert compute_dead_air([], 0.8) == []
