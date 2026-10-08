"""Shared fixtures. Media is generated with ffmpeg's lavfi sources, so no binary test assets are
checked in. Nothing here needs an API key or network access."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_qa.config import Settings

from helpers import HAVE_FFMPEG, run_ffmpeg


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real env (e.g. SLACK_WEBHOOK_URL) out of every test."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg and ffprobe must be on PATH")
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def synthetic_video(media_dir: Path) -> Path:
    """5.5 s, 1280x720 at 29.97 fps (NTSC timing), with a 440 Hz tone."""
    path = media_dir / "synthetic.mp4"
    run_ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30000/1001:duration=5.5",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=5.5",
        "-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac", "-shortest",
        str(path),
    )  # fmt: skip
    return path


@pytest.fixture(scope="session")
def silent_video(media_dir: Path) -> Path:
    """3.2 s, 320x240 at 25 fps, no audio stream."""
    path = media_dir / "silent.mp4"
    run_ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=3.2",
        "-c:v", "mpeg4", "-q:v", "5",
        str(path),
    )  # fmt: skip
    return path


@pytest.fixture(scope="session")
def audio_only(media_dir: Path) -> Path:
    path = media_dir / "tone.m4a"
    run_ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(path))
    return path


@pytest.fixture(scope="session")
def logo_png(media_dir: Path) -> Path:
    path = media_dir / "logo.png"
    run_ffmpeg("-f", "lavfi", "-i", "color=c=red:size=96x48", "-frames:v", "1", str(path))
    return path
