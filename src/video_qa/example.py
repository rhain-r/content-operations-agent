"""A hand-written example, so every dashboard screen can be shown without an API key.

The video is a real clip generated with ffmpeg's lavfi test source and split into frames the
same way an upload is. The words and the report are written by hand, and the report fails all
four checks. No AI is involved, and the dashboard labels the entry "Example".
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

import ffmpeg

from video_qa.config import Settings
from video_qa.deconstruction import extract_frames, map_words_to_frames, probe_video
from video_qa.jobs import FrameInfo, Job, JobStage, JobStatus, JobStore, job_media_dir, new_job_id
from video_qa.jobs import utc_now as now
from video_qa.schemas import AudioVisualMismatch, VideoQAReport, WordTimestamp
from video_qa.transcription import compute_dead_air, group_words_into_chunks

logger = logging.getLogger(__name__)

EXAMPLE_SECONDS = 10
EXAMPLE_NAME = "kitchen-tour-example.mp4"
EXAMPLE_EDITOR = "Sam (example)"

# Nothing is said until 0:04 (late opening line and a silence), then a second pause at 0:06.
EXAMPLE_WORDS = [
    WordTimestamp(word=word, start=start, end=end)
    for word, start, end in [
        ("So", 4.2, 4.4),
        ("here's", 4.5, 4.75),
        ("why", 4.8, 5.0),
        ("this", 5.1, 5.3),
        ("kitchen", 5.4, 5.8),
        ("actually", 5.9, 6.3),
        ("works.", 6.4, 6.8),
        ("Follow", 8.2, 8.5),
        ("for", 8.6, 8.75),
        ("part", 8.8, 9.0),
        ("two.", 9.1, 9.4),
    ]
]

EXAMPLE_FEEDBACK = (
    "Get the voice-over going sooner: nothing is said until 0:04, so most viewers will have "
    "scrolled on. The logo watermark drops out from 0:02 to 0:04 and again at 0:07, so put it "
    "back on those shots. At 0:05 the line about the kitchen plays over a test pattern; cut to "
    "a kitchen shot there. Finally, close the pause between 0:06 and 0:08."
)


async def create_example(settings: Settings, store: JobStore) -> Job | None:
    """Add the example to ``store``. Returns None (and logs why) if ffmpeg can't make it."""
    job_id = new_job_id()
    workdir = job_media_dir(settings.media_dir, job_id)
    video_path = workdir / "video.mp4"
    try:
        await asyncio.to_thread(workdir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(_render_clip, video_path, settings)
        metadata = await probe_video(video_path, settings)
        extracted = await extract_frames(video_path, workdir / "frames", metadata, settings)
    except Exception:
        logger.exception("Could not build the example video; starting without it")
        await asyncio.to_thread(shutil.rmtree, workdir, ignore_errors=True)
        return None

    timestamps = [timestamp for _, timestamp in extracted]
    frames = [
        FrameInfo(
            index=i, timestamp_seconds=timestamp, spoken=" ".join(w.word for w in frame_words)
        )
        for i, (timestamp, frame_words) in enumerate(
            zip(timestamps, map_words_to_frames(timestamps, EXAMPLE_WORDS), strict=True)
        )
    ]

    def at(index: int) -> float:
        return timestamps[min(index, len(timestamps) - 1)]

    report = VideoQAReport(
        passed_qa=False,
        logo_present=True,
        logo_missing_timestamps=sorted({at(2), at(3), at(4), at(7)}),
        hook_time_seconds=EXAMPLE_WORDS[0].start,
        audio_visual_mismatch=[
            AudioVisualMismatch(
                timestamp_seconds=at(5),
                spoken="“…why this kitchen actually works”",
                shown="A colourful test pattern, not a kitchen",
            )
        ],
        dead_air=compute_dead_air(
            EXAMPLE_WORDS,
            metadata.duration_seconds,
            threshold_seconds=settings.dead_air_threshold_seconds,
        ),
        feedback_for_editor=EXAMPLE_FEEDBACK,
    )

    logo_reference, logo_file = None, None
    if settings.agency_logo_path is not None:
        logo_file = f"logo{settings.agency_logo_path.suffix.lower()}"
        await asyncio.to_thread(shutil.copyfile, settings.agency_logo_path, workdir / logo_file)
        logo_reference = settings.agency_logo_path.name

    created = now()
    job = Job(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        stage=JobStage.DONE,
        stage_started_at=dict.fromkeys(JobStage, created),
        video_name=EXAMPLE_NAME,
        video_file=video_path.name,
        editor_name=EXAMPLE_EDITOR,
        is_example=True,
        created_at=created,
        updated_at=created,
        finished_at=created,
        duration_seconds=metadata.duration_seconds,
        frames=frames,
        transcript=group_words_into_chunks(
            EXAMPLE_WORDS,
            pause_seconds=settings.chunk_pause_seconds,
            max_chunk_seconds=settings.chunk_max_seconds,
        ),
        transcript_simulated=True,
        logo_reference=logo_reference,
        logo_file=logo_file,
        report=report,
    )
    await store.save(job)
    logger.info("Added the example report (%s)", job_id)
    return job


def _render_clip(path: Path, settings: Settings) -> None:
    duration = EXAMPLE_SECONDS
    picture = ffmpeg.input(f"testsrc2=size=720x1280:rate=30:duration={duration}", f="lavfi")
    tone = ffmpeg.input(f"sine=frequency=330:sample_rate=48000:duration={duration}", f="lavfi")
    options: dict[str, Any] = {
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
        "acodec": "aac",
        "movflags": "+faststart",
        "shortest": None,
    }
    try:
        _write(picture, tone, path, options, settings)
    except ffmpeg.Error:
        # ffmpeg builds without libx264: mpeg4 still yields frames, but browsers won't play it.
        logger.warning("libx264 unavailable; the example video won't play in a browser")
        _write(picture, tone, path, {**options, "vcodec": "mpeg4"}, settings)


def _write(
    picture: Any, tone: Any, path: Path, options: dict[str, Any], settings: Settings
) -> None:
    (
        ffmpeg.output(picture, tone, str(path), **options)
        .global_args("-hide_banner", "-nostdin")
        .overwrite_output()
        .run(cmd=settings.ffmpeg_cmd, capture_stdout=True, capture_stderr=True)
    )
