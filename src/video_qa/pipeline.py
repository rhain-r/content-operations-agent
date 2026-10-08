"""Runs one upload through deconstruction, QA, and (optionally) notification, as a bounded
background job, recording progress on the job as it goes."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from video_qa.agent import QAAgent
from video_qa.config import Settings
from video_qa.deconstruction import DeconstructionStage, deconstruct_video
from video_qa.findings import describe_failure
from video_qa.jobs import FrameInfo, Job, JobStage, JobStatus, JobStore, utc_now
from video_qa.notifier import Notifier
from video_qa.transcription import Transcriber, is_simulated

logger = logging.getLogger(__name__)

AUDIO_FILENAME = "audio.wav"


class QAPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        transcriber: Transcriber,
        agent: QAAgent,
        notifier: Notifier | None,
        store: JobStore,
    ) -> None:
        self._settings = settings
        self._transcriber = transcriber
        self._agent = agent
        # Only consulted when NOTIFIER=slack; with "none" the dashboard is the only output.
        self._notifier = notifier if settings.notifier == "slack" else None
        self._store = store

    async def run(self, job_id: str, video_path: Path, workdir: Path) -> Job:
        stored = await self._store.get(job_id)
        if stored is None:
            raise KeyError(f"unknown job {job_id}")
        job = await self._update(stored, status=JobStatus.PROCESSING)

        async def on_stage(stage: DeconstructionStage) -> None:
            nonlocal job
            job = await self._advance(job, JobStage(stage))

        try:
            job = await self._keep_logo_reference(job, workdir)
            video = await deconstruct_video(
                video_path, workdir, self._transcriber, self._settings, on_stage=on_stage
            )
            job = await self._update(
                job,
                duration_seconds=video.metadata.duration_seconds,
                frames=[
                    FrameInfo(
                        index=f.index, timestamp_seconds=f.timestamp_seconds, spoken=f.spoken_text
                    )
                    for f in video.frames
                ],
                transcript=video.transcript,
                transcript_simulated=is_simulated(self._transcriber),
            )
            job = await self._advance(job, JobStage.REVIEWING, model=self._settings.anthropic_model)
            outcome = await self._agent.review(video)
        except asyncio.CancelledError as exc:
            await self._fail(job, exc, raw="Cancelled: service shut down")
            raise
        except Exception as exc:
            logger.exception("QA job %s failed", job_id)
            return await self._fail(job, exc, raw=f"{type(exc).__name__}: {exc}")

        report = outcome.report
        notified, notification_error = False, None
        if self._notifier is not None and not report.passed_qa and job.editor_slack_id:
            # A Slack outage must not lose the report, so notification errors are recorded
            # on the job rather than failing it.
            try:
                notified = await self._notifier.notify(report, job.video_name, job.editor_slack_id)
            except Exception as exc:
                logger.exception("Could not notify editor for job %s", job_id)
                notification_error = f"{type(exc).__name__}: {exc}"
        logger.info("QA job %s completed: passed_qa=%s", job_id, report.passed_qa)
        return await self._advance(
            job,
            JobStage.DONE,
            status=JobStatus.COMPLETED,
            report=report,
            model_report=outcome.model_report,
            finished_at=utc_now(),
            editor_notified=notified,
            notification_error=notification_error,
        )

    async def _keep_logo_reference(self, job: Job, workdir: Path) -> Job:
        """Copy the logo image the reviewer is about to be shown, so the report can show it."""
        logo = self._settings.agency_logo_path
        if logo is None:
            return job
        stored = f"logo{logo.suffix.lower()}"
        await asyncio.to_thread(shutil.copyfile, logo, workdir / stored)
        return await self._update(job, logo_reference=logo.name, logo_file=stored)

    async def _fail(self, job: Job, exc: BaseException, *, raw: str) -> Job:
        return await self._update(
            job,
            status=JobStatus.ERROR,
            error=raw,
            error_message=describe_failure(exc),
            finished_at=utc_now(),
        )

    async def _advance(self, job: Job, stage: JobStage, **changes: Any) -> Job:
        if stage == job.stage:
            return await self._update(job, **changes) if changes else job
        started = {**job.stage_started_at, stage: utc_now()}
        return await self._update(job, stage=stage, stage_started_at=started, **changes)

    async def _update(self, job: Job, **changes: Any) -> Job:
        updated = job.model_copy(update={**changes, "updated_at": utc_now()})
        await self._store.save(updated)
        return updated


class JobRunner:
    """Runs pipeline jobs as background tasks, at most ``max_concurrent`` at once.

    Holds a reference to every task until it finishes (the event loop only keeps weak ones).
    When a task ends, however it ends, the extracted audio is deleted; the video and frames
    stay in the job's folder for the dashboard.
    """

    def __init__(self, pipeline: QAPipeline, *, max_concurrent: int) -> None:
        self._pipeline = pipeline
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def active_jobs(self) -> int:
        return len(self._tasks)

    def submit(self, job_id: str, video_path: Path, workdir: Path) -> asyncio.Task[None]:
        task = asyncio.create_task(self._run(job_id, video_path, workdir), name=f"qa-job-{job_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(self, job_id: str, video_path: Path, workdir: Path) -> None:
        try:
            async with self._semaphore:
                await self._pipeline.run(job_id, video_path, workdir)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("QA job %s crashed", job_id)
        finally:
            await asyncio.to_thread((workdir / AUDIO_FILENAME).unlink, missing_ok=True)

    async def shutdown(self) -> None:
        """Cancel running and queued jobs and wait for their cleanup."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
