"""HTTP API and dashboard: accept an upload, check it in the background, show the result.

Run with ``uvicorn --factory video_qa.main:create_app`` and open http://localhost:8000.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Annotated, BinaryIO

import httpx
from anthropic import AsyncAnthropic
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from video_qa.agent import ClaudeQAAgent, QAAgent
from video_qa.config import Settings
from video_qa.deconstruction import VideoRejectedError, probe_video
from video_qa.example import create_example
from video_qa.findings import (
    REVIEWER_NOT_CONNECTED,
    describe_failure,
    describe_upload_too_large,
)
from video_qa.jobs import (
    InMemoryJobStore,
    Job,
    JobStage,
    JobStore,
    clear_media_dir,
    job_media_dir,
    new_job_id,
    parse_job_id,
    prune_jobs,
    utc_now,
)
from video_qa.notifier import (
    InvalidSlackUserIdError,
    Notifier,
    SlackNotifier,
    validate_slack_user_id,
)
from video_qa.pipeline import JobRunner, QAPipeline
from video_qa.services import AppServices, Services
from video_qa.transcription import MockTranscriber, Transcriber
from video_qa.web import WEB_DIR
from video_qa.web.pages import router as pages_router
from video_qa.web.views import JOBS_PATH, JobDetail, JobSummary, job_detail, job_summary

logger = logging.getLogger(__name__)

_UPLOAD_CHUNK_BYTES = 1024 * 1024
# Allowance for multipart boundaries and the form fields around the file itself.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024


class UploadTooLargeError(Exception):
    pass


class FriendlyError(Exception):
    """An API error with a technical ``detail`` and a plain-language ``message`` for the UI."""

    def __init__(self, status_code: int, detail: str, message: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.message = message


class ContentLengthLimitMiddleware:
    """Reject oversized uploads from the Content-Length header, before the body is read.

    The handler also counts bytes as it copies the file, which covers chunked uploads that
    don't declare a length.
    """

    def __init__(
        self, app: ASGIApp, *, path: str, max_body_bytes: int, max_upload_bytes: int
    ) -> None:
        self._app = app
        self._path = path
        self._max_body_bytes = max_body_bytes
        self._max_upload_bytes = max_upload_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == self._path:
            declared = Headers(scope=scope).get("content-length", "")
            if declared.isdigit() and int(declared) > self._max_body_bytes:
                response = JSONResponse(
                    {
                        "detail": f"Upload exceeds {self._max_body_bytes} bytes",
                        "message": describe_upload_too_large(self._max_upload_bytes, int(declared)),
                    },
                    status_code=413,
                )
                await response(scope, receive, send)
                return
        await self._app(scope, receive, send)


async def require_job(job_id: str, services: Services) -> Job:
    """The job named in the path. Unknown or malformed ids are both a plain 404."""
    canonical = parse_job_id(job_id)
    job = await services.store.get(canonical) if canonical else None
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


JobDep = Annotated[Job, Depends(require_job)]


def reviewer_configured(settings: Settings) -> bool:
    """Whether the Claude reviewer has credentials it can use."""
    return settings.anthropic_api_key is not None or bool(os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def create_app(
    *,
    settings: Settings | None = None,
    agent: QAAgent | None = None,
    transcriber: Transcriber | None = None,
    notifier: Notifier | None = None,
    store: JobStore | None = None,
) -> FastAPI:
    """Build the app. Anything not injected is created from settings: a Claude agent, the mock
    transcriber, and (only with NOTIFIER=slack) a Slack notifier."""
    settings = settings or Settings()
    store = store or InMemoryJobStore()
    transcriber = transcriber or MockTranscriber()
    reviewer_connected = agent is not None or reviewer_configured(settings)
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await clear_media_dir(settings.media_dir)
        async with AsyncExitStack() as stack:
            qa_agent = agent
            if qa_agent is None:
                api_key = settings.anthropic_api_key
                anthropic_client = (
                    AsyncAnthropic(api_key=api_key.get_secret_value())
                    if api_key
                    else AsyncAnthropic()
                )
                stack.push_async_callback(anthropic_client.close)
                qa_agent = ClaudeQAAgent(anthropic_client, settings)

            qa_notifier: Notifier | None = None
            if settings.notifier == "slack":
                qa_notifier = notifier
                if qa_notifier is None:
                    webhook = settings.slack_webhook_url
                    http_client = httpx.AsyncClient(timeout=settings.slack_timeout_seconds)
                    stack.push_async_callback(http_client.aclose)
                    qa_notifier = SlackNotifier(
                        webhook_url=webhook.get_secret_value() if webhook else None,
                        http_client=http_client,
                        hook_deadline_seconds=settings.hook_deadline_seconds,
                        frame_interval_seconds=settings.frame_interval_seconds,
                    )

            pipeline = QAPipeline(
                settings=settings,
                transcriber=transcriber,
                agent=qa_agent,
                notifier=qa_notifier,
                store=store,
            )
            runner = JobRunner(pipeline, max_concurrent=settings.max_concurrent_jobs)
            # Registered last so it runs first: jobs stop before their HTTP clients close.
            stack.push_async_callback(runner.shutdown)
            if settings.seed_example:
                await create_example(settings, store)
            app.state.services = AppServices(
                settings=settings,
                store=store,
                runner=runner,
                reviewer_connected=reviewer_connected,
            )
            yield

    app = FastAPI(title="Video QA", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        ContentLengthLimitMiddleware,
        path=JOBS_PATH,
        max_body_bytes=settings.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES,
        max_upload_bytes=settings.max_upload_bytes,
    )
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    app.include_router(pages_router)

    @app.exception_handler(FriendlyError)
    async def friendly_error(request: Request, exc: FriendlyError) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.detail, "message": exc.message}, status_code=exc.status_code
        )

    @app.get("/healthz")
    async def healthz(services: Services) -> dict[str, str | int]:
        return {"status": "ok", "active_jobs": services.runner.active_jobs}

    @app.post(JOBS_PATH, status_code=202, response_model=JobDetail)
    async def create_qa_job(
        services: Services,
        response: Response,
        file: Annotated[UploadFile, File(description="The finished video, e.g. an .mp4")],
        editor_name: Annotated[str | None, Form(description="Who edited the video")] = None,
        editor_slack_id: Annotated[
            str | None, Form(description="Slack user ID to ping; required with NOTIFIER=slack")
        ] = None,
    ) -> JobDetail:
        settings = services.settings
        if not services.reviewer_connected:
            raise FriendlyError(
                503,
                "The Claude reviewer has no credentials: set ANTHROPIC_API_KEY.",
                REVIEWER_NOT_CONNECTED,
            )
        slack_id = (editor_slack_id or "").strip() or None
        if settings.notifier == "slack" and slack_id is None:
            raise FriendlyError(
                422,
                "editor_slack_id is required when NOTIFIER=slack",
                "Add the editor's Slack member ID so they can be told about the result.",
            )
        if slack_id is not None:
            try:
                validate_slack_user_id(slack_id)
            except InvalidSlackUserIdError as exc:
                raise FriendlyError(
                    422,
                    str(exc),
                    "That doesn't look like a Slack member ID. It starts with U or W, "
                    "for example U0123ABCD.",
                ) from exc

        video_name = _display_name(file.filename)
        job_id = new_job_id()
        workdir = job_media_dir(settings.media_dir, job_id)
        video_path = workdir / f"video{_safe_suffix(video_name)}"
        try:
            await asyncio.to_thread(workdir.mkdir, parents=True)
            await _save_upload(file, video_path, settings.max_upload_bytes)
            metadata = await probe_video(video_path, settings)
        except UploadTooLargeError as exc:
            await _remove(workdir)
            raise FriendlyError(
                413, str(exc), describe_upload_too_large(settings.max_upload_bytes)
            ) from exc
        except VideoRejectedError as exc:
            await _remove(workdir)
            raise FriendlyError(422, str(exc), describe_failure(exc)) from exc
        except BaseException:
            await _remove(workdir)
            raise

        now = utc_now()
        job = Job(
            job_id=job_id,
            video_name=video_name,
            video_file=video_path.name,
            editor_name=(editor_name or "").strip()[:100] or None,
            editor_slack_id=slack_id,
            created_at=now,
            updated_at=now,
            stage_started_at={JobStage.RECEIVED: now},
            duration_seconds=metadata.duration_seconds,
        )
        await services.store.save(job)
        await prune_jobs(services.store, settings.media_dir, keep=settings.max_stored_jobs)
        services.runner.submit(job_id, video_path, workdir)
        response.headers["Location"] = f"{JOBS_PATH}/{job_id}"
        return job_detail(job, settings)

    @app.get(JOBS_PATH, response_model=list[JobSummary])
    async def list_qa_jobs(services: Services) -> list[JobSummary]:
        """Every job, newest first."""
        return [job_summary(job, services.settings) for job in await services.store.list()]

    @app.get(JOBS_PATH + "/{job_id}", response_model=JobDetail)
    async def get_qa_job(job: JobDep, services: Services) -> JobDetail:
        return job_detail(job, services.settings)

    @app.get(JOBS_PATH + "/{job_id}/video")
    async def get_video(job: JobDep, services: Services) -> FileResponse:
        """The uploaded video. Answers Range requests with 206, so browsers can seek."""
        folder = job_media_dir(services.settings.media_dir, job.job_id)
        return await _file_response(folder / job.video_file)

    @app.get(JOBS_PATH + "/{job_id}/frames/{index}.jpg")
    async def get_frame(job: JobDep, index: int, services: Services) -> FileResponse:
        if not 0 <= index < len(job.frames):
            raise HTTPException(status_code=404, detail="No such frame")
        folder = job_media_dir(services.settings.media_dir, job.job_id)
        return await _file_response(
            folder / "frames" / f"frame_{index + 1:05d}.jpg",
            media_type="image/jpeg",
            cache_control="private, max-age=86400",
        )

    @app.get(JOBS_PATH + "/{job_id}/logo")
    async def get_logo(job: JobDep, services: Services) -> FileResponse:
        if job.logo_file is None:
            raise HTTPException(status_code=404, detail="No logo reference for this job")
        folder = job_media_dir(services.settings.media_dir, job.job_id)
        return await _file_response(folder / job.logo_file)

    return app


async def _file_response(
    path: Path, *, media_type: str | None = None, cache_control: str = "no-cache"
) -> FileResponse:
    if not await asyncio.to_thread(path.is_file):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": cache_control})


async def _save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    """Copy the upload to ``destination`` in chunks, stopping as soon as it passes the cap."""
    total = 0
    handle: BinaryIO = await asyncio.to_thread(destination.open, "wb")
    try:
        while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > max_bytes:
                raise UploadTooLargeError(f"Upload exceeds {max_bytes} bytes")
            await asyncio.to_thread(handle.write, chunk)
    finally:
        await asyncio.to_thread(handle.close)
    return total


async def _remove(path: Path) -> None:
    await asyncio.to_thread(shutil.rmtree, path, ignore_errors=True)


def _display_name(filename: str | None) -> str:
    """The uploaded file's base name, whichever OS's separators the client used."""
    name = re.split(r"[\\/]", filename or "")[-1].strip()
    return name[:200] or "upload.mp4"


def _safe_suffix(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) else ".mp4"


def _configure_logging(level: str) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("video_qa").setLevel(level.upper())
