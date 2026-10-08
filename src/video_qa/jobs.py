"""QA job records, where they are kept, and the media folder that goes with each one."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, computed_field

from video_qa.schemas import TranscriptChunk, VideoQAReport

logger = logging.getLogger(__name__)

_JOB_ID = re.compile(r"[0-9a-f]{32}")


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_job_id() -> str:
    return uuid.uuid4().hex


def parse_job_id(raw: str) -> str | None:
    """The canonical id for ``raw`` if it is a UUID, else None. File paths are only ever built
    from what this returns, never from request text."""
    try:
        return uuid.UUID(raw).hex
    except ValueError:
        return None


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"  # the report is in; check report.passed_qa for the verdict
    ERROR = "error"  # the pipeline couldn't produce a report; see error / error_message


class JobStage(StrEnum):
    """Where a job is, in the order the dashboard's progress stepper shows."""

    RECEIVED = "received"
    READING = "reading"
    SPLITTING = "splitting"
    LISTENING = "listening"
    REVIEWING = "reviewing"
    DONE = "done"


class FrameInfo(BaseModel):
    index: int
    timestamp_seconds: float
    spoken: str = Field(default="", description="Words spoken from this frame to the next.")


class Job(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    stage: JobStage = JobStage.RECEIVED
    stage_started_at: dict[JobStage, datetime] = Field(default_factory=dict)
    video_name: str
    video_file: str = Field(
        default="video.mp4", description="The stored copy's name inside the job's media folder."
    )
    editor_name: str | None = None
    editor_slack_id: str | None = None
    is_example: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    frames: list[FrameInfo] = Field(default_factory=list)
    transcript: list[TranscriptChunk] = Field(default_factory=list)
    transcript_simulated: bool = False
    logo_reference: str | None = Field(
        default=None, description="File name of the logo image the reviewer was shown, if any."
    )
    logo_file: str | None = None
    model: str | None = None
    report: VideoQAReport | None = Field(default=None, description="Verdict enforced by code.")
    model_report: VideoQAReport | None = Field(
        default=None, description="The report exactly as the model returned it."
    )
    error: str | None = Field(default=None, description="Raw error, for developers.")
    error_message: str | None = Field(default=None, description="The same, in plain words.")
    editor_notified: bool = False
    notification_error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_passed_qa(self) -> bool | None:
        """The model's own verdict, before the code re-checked it."""
        return self.model_report.passed_qa if self.model_report else None

    @property
    def finished(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.ERROR)


class JobStore(Protocol):
    """Kept async and copy-in/copy-out so a Redis or database store can drop in."""

    async def save(self, job: Job) -> None: ...

    async def get(self, job_id: str) -> Job | None: ...

    async def list(self) -> list[Job]:
        """Every job, newest first."""
        ...

    async def delete(self, job_id: str) -> None: ...


class InMemoryJobStore:
    """Process-local store: jobs vanish on restart and aren't shared between workers."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def save(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.job_id] = job.model_copy(deep=True)

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    async def list(self) -> list[Job]:
        async with self._lock:
            jobs = [job.model_copy(deep=True) for job in self._jobs.values()]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    async def delete(self, job_id: str) -> None:
        async with self._lock:
            self._jobs.pop(job_id, None)


# --- Media folders ----------------------------------------------------------------------------


def job_media_dir(media_dir: Path, job_id: str) -> Path:
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError(f"not a job id: {job_id!r}")
    return media_dir / job_id


async def clear_media_dir(media_dir: Path) -> int:
    """Delete job folders left over from an earlier run. Only folders named like a job id are
    touched, so pointing MEDIA_DIR at a folder with other things in it is safe."""

    def clear() -> int:
        media_dir.mkdir(parents=True, exist_ok=True)
        removed = 0
        for child in media_dir.iterdir():
            if child.is_dir() and _JOB_ID.fullmatch(child.name):
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        return removed

    removed = await asyncio.to_thread(clear)
    if removed:
        logger.info("Removed %d stored job folder(s) from a previous run in %s", removed, media_dir)
    return removed


async def prune_jobs(store: JobStore, media_dir: Path, *, keep: int) -> list[str]:
    """Keep the ``keep`` newest jobs and delete the rest, records and media both. Jobs still
    being checked and the seeded example are never pruned."""
    jobs = [job for job in await store.list() if not job.is_example]
    removed: list[str] = []
    for job in jobs[keep:]:
        if not job.finished:
            continue
        await store.delete(job.job_id)
        await asyncio.to_thread(
            shutil.rmtree, job_media_dir(media_dir, job.job_id), ignore_errors=True
        )
        removed.append(job.job_id)
    return removed
