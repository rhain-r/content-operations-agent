from __future__ import annotations

import json
import subprocess
import wave
from itertools import pairwise
from pathlib import Path

import pytest

from video_qa.deconstruction import (
    VideoRejectedError,
    deconstruct_video,
    expected_frame_count,
    map_words_to_frames,
    parse_showinfo_timestamps,
    probe_video,
    render_reference_image,
    resolve_frame_timestamps,
)
from video_qa.transcription import MockTranscriber

from helpers import make_settings, segment, words


def image_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return stream["width"], stream["height"]


async def test_frames_carry_exact_timestamps_and_words(
    synthetic_video: Path, tmp_path: Path
) -> None:
    video = await deconstruct_video(synthetic_video, tmp_path, MockTranscriber(), make_settings())

    # 29.97 fps: the first frame at or after each whole second is 1.001 s, 2.002 s, ... in.
    # An fps=1 filter would have labelled these 1.0, 2.0, ...
    assert len(video.frames) == 6
    assert [f.timestamp_seconds for f in video.frames] == pytest.approx(
        [0.0, 1.001, 2.002, 3.003, 4.004, 5.005], abs=1e-3
    )
    assert [f.index for f in video.frames] == list(range(6))
    assert all(f.path.is_file() for f in video.frames)
    assert image_size(video.frames[0].path) == (1024, 576)  # long edge scaled to 1024

    assert video.audio_path is not None
    with wave.open(str(video.audio_path), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16_000, 1, 2)

    # Every word sits on exactly one frame: the last frame at or before the word's start.
    assert video.words
    assert [w for frame in video.frames for w in frame.words] == video.words
    for frame, following in pairwise(video.frames):
        assert all(
            frame.timestamp_seconds <= w.start < following.timestamp_seconds for w in frame.words
        )

    assert video.metadata.duration_seconds == pytest.approx(5.5, abs=0.05)
    assert video.metadata.has_audio
    assert [c.text for c in video.transcript] == [
        "Stop scrolling if your kitchen never feels big enough.",
        "Here",
    ]
    assert video.dead_air == []


async def test_video_without_audio_is_all_dead_air(silent_video: Path, tmp_path: Path) -> None:
    video = await deconstruct_video(silent_video, tmp_path, MockTranscriber(), make_settings())

    assert video.audio_path is None
    assert video.words == []
    assert video.transcript == []
    assert video.dead_air == [segment(0.0, video.metadata.duration_seconds)]
    assert [f.timestamp_seconds for f in video.frames] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert all(f.words == [] for f in video.frames)
    assert image_size(video.frames[0].path) == (320, 240)  # never upscaled
    assert not (tmp_path / "audio.wav").exists()


async def test_rejects_video_over_frame_budget_before_extracting(
    synthetic_video: Path, tmp_path: Path
) -> None:
    with pytest.raises(VideoRejectedError, match=r"would produce 6 frames.*limit is 5"):
        await deconstruct_video(
            synthetic_video, tmp_path, MockTranscriber(), make_settings(max_frames=5)
        )
    assert list(tmp_path.iterdir()) == []


def test_logo_reference_takes_one_image_from_the_frame_budget(logo_png: Path) -> None:
    assert make_settings(max_frames=100).frame_budget == 100
    assert make_settings(max_frames=100, agency_logo_path=logo_png).frame_budget == 99
    assert make_settings(max_frames=40, agency_logo_path=logo_png).frame_budget == 40


async def test_rejects_file_without_video_stream(audio_only: Path) -> None:
    with pytest.raises(VideoRejectedError, match="no video stream"):
        await probe_video(audio_only, make_settings())


async def test_rejects_unreadable_file(media_dir: Path) -> None:
    junk = media_dir / "junk.mp4"
    junk.write_bytes(b"this is not a video")
    with pytest.raises(VideoRejectedError, match="Not a readable video file"):
        await probe_video(junk, make_settings())


async def test_rejects_still_image(logo_png: Path) -> None:
    with pytest.raises(VideoRejectedError, match="still image"):
        await probe_video(logo_png, make_settings())


async def test_render_reference_image_returns_png(logo_png: Path) -> None:
    png = await render_reference_image(logo_png, make_settings())
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_parse_showinfo_timestamps() -> None:
    stderr = "\n".join(
        [
            "[Parsed_showinfo_1 @ 000001e9] config in time_base: 1/30000, frame_rate: 30000/1001",
            "[Parsed_showinfo_1 @ 000001e9] n:   0 pts:      0 pts_time:0       duration: 1001",
            "[Parsed_showinfo_1 @ 000001e9] n:   1 pts:  30030 pts_time:1.001   duration: 1001",
            "[mjpeg @ 000001ea] deprecated pixel format used",
            "[Parsed_showinfo_1 @ 000001e9] n:   2 pts:  60060 pts_time:2.002   duration: 1001",
        ]
    )
    assert parse_showinfo_timestamps(stderr) == [0.0, 1.001, 2.002]


def test_resolve_frame_timestamps_falls_back_to_interval_on_mismatch() -> None:
    assert resolve_frame_timestamps([0.0, 1.001, 2.002], 3, 1.0) == [0.0, 1.001, 2.002]
    assert resolve_frame_timestamps([0.0, 1.001], 3, 1.0) == [0.0, 1.0, 2.0]
    assert resolve_frame_timestamps([], 2, 0.5) == [0.0, 0.5]


def test_map_words_to_frames() -> None:
    spoken = words(("early", 0.0, 0.2), ("one", 0.9, 1.2), ("two", 1.001, 1.3), ("end", 2.5, 2.8))
    buckets = map_words_to_frames([0.033, 1.001, 2.002], spoken)
    assert [[w.word for w in bucket] for bucket in buckets] == [
        ["early", "one"],  # "early" starts before the first frame, so it goes to frame 0
        ["two"],
        ["end"],
    ]


@pytest.mark.parametrize(
    ("duration", "fps", "interval", "expected"),
    [
        (5.472, 30000 / 1001, 1.0, 6),
        (6.0, 30.0, 1.0, 6),
        (100.0, 30.0, 1.0, 100),
        (100.5, 30.0, 1.0, 101),
        (3.2, 25.0, 1.0, 4),
        (0.5, 30.0, 1.0, 1),
        (6.0, None, 1.0, 7),
    ],
)
def test_expected_frame_count(
    duration: float, fps: float | None, interval: float, expected: int
) -> None:
    assert expected_frame_count(duration, fps, interval) == expected
