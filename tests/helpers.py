"""Plain helpers shared by the tests (fixtures live in conftest.py)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from video_qa.config import Settings
from video_qa.jobs import FrameInfo, Job, JobStage, new_job_id, utc_now
from video_qa.schemas import (
    DeadAirSegment,
    DeconstructedVideo,
    Frame,
    VideoMetadata,
    WordTimestamp,
)
from video_qa.transcription import compute_dead_air, group_words_into_chunks

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_settings(**overrides: Any) -> Settings:
    """Settings that ignore any local .env file."""
    return Settings(_env_file=None, **overrides)


def run_ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args],
        check=True,
        capture_output=True,
    )


def make_job(**changes: Any) -> Job:
    """A job record built by hand, for rendering pages without running the pipeline."""
    now = utc_now()
    job = Job(
        job_id=new_job_id(),
        video_name="kitchen-ad-v3.mp4",
        editor_name="Sam",
        duration_seconds=6.0,
        frames=[FrameInfo(index=i, timestamp_seconds=i * 1.001) for i in range(6)],
        created_at=now,
        updated_at=now,
        stage_started_at={JobStage.RECEIVED: now},
    )
    return job.model_copy(update=changes)


def words(*items: tuple[str, float, float]) -> list[WordTimestamp]:
    return [WordTimestamp(word=w, start=s, end=e) for w, s, e in items]


def segment(start: float, end: float) -> DeadAirSegment:
    return DeadAirSegment(start_seconds=start, end_seconds=end)


def fake_video(
    tmp_path: Path,
    *,
    timestamps: list[float],
    spoken: list[WordTimestamp],
    duration: float,
) -> DeconstructedVideo:
    """A DeconstructedVideo built by hand, with placeholder bytes for the frame images."""
    frames: list[Frame] = []
    for i, t in enumerate(timestamps):
        path = tmp_path / f"frame_{i + 1:05d}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0 frame %d \xff\xd9" % i)
        next_t = timestamps[i + 1] if i + 1 < len(timestamps) else float("inf")
        frame_words = [w for w in spoken if t <= w.start < next_t]
        frames.append(Frame(index=i, timestamp_seconds=t, path=path, words=frame_words))
    metadata = VideoMetadata(
        filename="ad.mp4",
        container="mov,mp4,m4a,3gp,3g2,mj2",
        duration_seconds=duration,
        width=1080,
        height=1920,
        fps=29.97,
        video_codec="h264",
        has_audio=True,
        audio_codec="aac",
        size_bytes=1234,
        expected_frame_count=len(timestamps),
        video_stream_index=0,
        audio_stream_index=1,
    )
    return DeconstructedVideo(
        metadata=metadata,
        frames=frames,
        audio_path=tmp_path / "audio.wav",
        words=spoken,
        transcript=group_words_into_chunks(spoken),
        dead_air=compute_dead_air(spoken, duration),
    )
