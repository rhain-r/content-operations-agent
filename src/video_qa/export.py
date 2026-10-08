"""Static showcase: check videos on this machine, then render read-only report pages.

    python -m video_qa.export --videos demo_videos/ --out docs/

The output is plain files (HTML, CSS, JS, JPEGs, MP4s, JSON) with relative links only, so it
can be published anywhere, including a GitHub Pages project site under ``/<repo>/``. Nothing in
it calls the API.

Reviewing and rendering are separate. A video's review is saved as ``report.json`` beside its
page; later runs reuse that file and only render the HTML again, so adjusting the design never
spends API calls. ``--refresh`` forces a new review.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import string
import subprocess
import sys
import tempfile
import unicodedata
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ffmpeg
from anthropic import AsyncAnthropic

from video_qa.agent import ClaudeQAAgent, QAAgent
from video_qa.config import Settings
from video_qa.deconstruction import VideoRejectedError, probe_video
from video_qa.example import create_example
from video_qa.findings import check_definitions, describe_failure, megabytes
from video_qa.jobs import (
    InMemoryJobStore,
    Job,
    JobStage,
    JobStatus,
    job_media_dir,
    new_job_id,
    utc_now,
)
from video_qa.pipeline import QAPipeline
from video_qa.schemas import VideoMetadata
from video_qa.transcription import MockTranscriber, Transcriber
from video_qa.web import WEB_DIR
from video_qa.web.pages import templates
from video_qa.web.views import build_video_page, static_links, video_rows

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
MAX_FILE_BYTES = 25 * 1024 * 1024
WEB_VIDEO_LONG_EDGE = 1280
WEB_VIDEO_FILE = "video.mp4"
REPORT_FILE = "report.json"
REPORT_FORMAT = "video-qa-showcase/1"

# The parts of a job that are published. Ids, editor names, Slack ids, raw errors and
# notification state stay on this machine.
PUBLIC_FIELDS = {
    "video_name",
    "created_at",
    "finished_at",
    "duration_seconds",
    "frames",
    "transcript",
    "transcript_simulated",
    "logo_reference",
    "logo_file",
    "model",
    "report",
    "model_report",
    "stage_started_at",
    "is_example",
}
EXAMPLE_SLUG = "example"


class ExportError(RuntimeError):
    """The export can't go on; the message says what to do."""


class _SkipVideoError(Exception):
    """This video couldn't be checked, so it isn't published."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass
class ExportResult:
    published: list[str] = field(default_factory=list)  # slugs, in gallery order
    reviewed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (file name, reason)
    total_bytes: int = 0
    file_count: int = 0
    largest: tuple[str, int] = ("", 0)


# --- Names ------------------------------------------------------------------------------------


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug[:60].strip("-") or "video"


def assign_slugs(files: Sequence[Path]) -> dict[Path, str]:
    taken: set[str] = {EXAMPLE_SLUG}  # reserved for the hand-written Example
    slugs: dict[Path, str] = {}
    for path in files:
        base = slugify(path.stem)
        slug, n = base, 2
        while slug in taken:
            slug, n = f"{base}-{n}", n + 1
        taken.add(slug)
        slugs[path] = slug
    return slugs


def editor_placeholder(index: int) -> str:
    letters = string.ascii_uppercase
    return f"Editor {letters[index]}" if index < len(letters) else f"Editor {index + 1}"


def _job_id_for(slug: str) -> str:
    """A stable id per published video, so links don't change between exports."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"video-qa-showcase:{slug}").hex


# --- Export -----------------------------------------------------------------------------------


async def export_showcase(
    videos_dir: Path,
    out_dir: Path,
    *,
    settings: Settings,
    agent: QAAgent | None,
    transcriber: Transcriber | None = None,
    refresh: bool = False,
    repo_url: str | None = None,
    max_file_bytes: int = MAX_FILE_BYTES,
    include_example: bool = False,
) -> ExportResult:
    """Check (or reuse) every video in ``videos_dir`` and write the static site to ``out_dir``.

    ``agent`` may be None when every video already has a saved review; a review that's needed
    without one raises ``ExportError``. ``include_example`` adds the hand-written Example
    (a test pattern with a made-up report, no AI), labelled as such on every page.
    """
    settings = settings.model_copy(update={"notifier": "none"})
    transcriber = transcriber or MockTranscriber()
    sources = await asyncio.to_thread(_find_videos, videos_dir, include_example)
    slugs = assign_slugs(sources)
    result = ExportResult()
    jobs: dict[str, Job] = {}

    for source in sources:
        slug = slugs[source]
        folder = out_dir / "videos" / slug
        try:
            cached = None if refresh else await asyncio.to_thread(_load_review, folder, slug)
            if cached is None:
                # Unreadable files are skipped whether or not a reviewer is connected.
                await _probe_or_skip(source, settings)
                if agent is None:
                    raise ExportError(
                        f"{source.name} needs reviewing, but the AI reviewer isn't connected. "
                        "Add ANTHROPIC_API_KEY to .env and run the export again. "
                        "(Videos that already have a report.json are reused without it.)"
                    )
                print(f"Reviewing {source.name} ...", flush=True)
                job = await _review(source, folder, slug, settings, agent, transcriber)
                result.reviewed.append(slug)
            else:
                print(f"Reusing the saved review of {source.name}", flush=True)
                job = cached
                result.reused.append(slug)
            await asyncio.to_thread(_ensure_frames, source, folder, job)
            await _ensure_web_video(source, folder, settings, refresh, max_file_bytes)
        except _SkipVideoError as skip:
            result.skipped.append((source.name, skip.reason))
            if skip.detail:
                logger.info("%s couldn't be checked: %s", source.name, skip.detail)
            await asyncio.to_thread(shutil.rmtree, folder, ignore_errors=True)
            continue
        # Real editor names never leave this machine; published videos get placeholders.
        jobs[slug] = job.model_copy(update={"editor_name": editor_placeholder(len(jobs))})

    if include_example:
        example = await _export_example(out_dir / "videos" / EXAMPLE_SLUG, settings)
        if example is None:
            raise ExportError("The Example couldn't be made; see the log above for ffmpeg's error.")
        jobs[EXAMPLE_SLUG] = example

    await asyncio.to_thread(_write_site, out_dir, jobs, settings, repo_url, max_file_bytes, result)
    return result


def _find_videos(videos_dir: Path, include_example: bool = False) -> list[Path]:
    if not videos_dir.is_dir():
        if include_example:
            return []
        raise ExportError(f"There's no folder called {videos_dir}. Create it and put videos in it.")
    videos = sorted(
        p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos and not include_example:
        raise ExportError(
            f"No videos found in {videos_dir} (looked for {', '.join(sorted(VIDEO_EXTENSIONS))})."
        )
    return videos


async def _probe_or_skip(source: Path, settings: Settings) -> VideoMetadata:
    try:
        return await probe_video(source, settings)
    except VideoRejectedError as exc:
        raise _SkipVideoError(describe_failure(exc), str(exc)) from exc


async def _review(
    source: Path,
    folder: Path,
    slug: str,
    settings: Settings,
    agent: QAAgent,
    transcriber: Transcriber,
) -> Job:
    """Run one video through the same pipeline the API uses and keep what the page needs."""
    metadata = await _probe_or_skip(source, settings)
    store = InMemoryJobStore()
    pipeline = QAPipeline(
        settings=settings, transcriber=transcriber, agent=agent, notifier=None, store=store
    )
    now = utc_now()
    job = Job(
        job_id=new_job_id(),
        video_name=source.name,
        video_file=WEB_VIDEO_FILE,
        created_at=now,
        updated_at=now,
        stage_started_at={JobStage.RECEIVED: now},
        duration_seconds=metadata.duration_seconds,
    )
    await store.save(job)
    workdir = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="video-qa-export-"))
    try:
        finished = await pipeline.run(job.job_id, source, workdir)
        if finished.status is not JobStatus.COMPLETED:
            raise _SkipVideoError(
                finished.error_message or "It couldn't be checked.", finished.error
            )
        published = finished.model_copy(update={"job_id": _job_id_for(slug)})
        await asyncio.to_thread(_save_review, workdir, folder, published)
    finally:
        await asyncio.to_thread(shutil.rmtree, workdir, ignore_errors=True)
    return published


async def _export_example(folder: Path, settings: Settings) -> Job | None:
    """The hand-written Example from ``video_qa.example``, copied into the showcase layout."""
    media = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="video-qa-example-"))
    try:
        store = InMemoryJobStore()
        job = await create_example(settings.model_copy(update={"media_dir": media}), store)
        if job is None:
            return None
        published = job.model_copy(update={"job_id": _job_id_for(EXAMPLE_SLUG)})
        await asyncio.to_thread(shutil.rmtree, folder, ignore_errors=True)
        await asyncio.to_thread(_save_review, job_media_dir(media, job.job_id), folder, published)
        await asyncio.to_thread(
            shutil.copyfile,
            job_media_dir(media, job.job_id) / job.video_file,
            folder / WEB_VIDEO_FILE,
        )
        return published.model_copy(update={"video_file": WEB_VIDEO_FILE})
    finally:
        await asyncio.to_thread(shutil.rmtree, media, ignore_errors=True)


def _save_review(workdir: Path, folder: Path, job: Job) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frames = folder / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir()
    for frame in job.frames:
        shutil.copyfile(
            workdir / "frames" / f"frame_{frame.index + 1:05d}.jpg", frames / f"{frame.index}.jpg"
        )
    if job.logo_file:
        shutil.copyfile(workdir / job.logo_file, folder / job.logo_file)
    record = {
        "format": REPORT_FORMAT,
        "job": job.model_dump(mode="json", include=PUBLIC_FIELDS),
    }
    (folder / REPORT_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _load_review(folder: Path, slug: str) -> Job | None:
    path = folder / REPORT_FILE
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != REPORT_FORMAT:
        raise ExportError(
            f"{path} was written by a different version of the exporter. "
            "Run again with --refresh to review that video again."
        )
    return Job.model_validate(
        {
            **data["job"],
            "job_id": _job_id_for(slug),
            "status": JobStatus.COMPLETED,
            "stage": JobStage.DONE,
            "video_file": WEB_VIDEO_FILE,
        }
    )


def _ensure_frames(source: Path, folder: Path, job: Job) -> None:
    missing = [f.index for f in job.frames if not (folder / "frames" / f"{f.index}.jpg").is_file()]
    if missing:
        raise ExportError(
            f"Frames are missing for {source.name} (for example frames/{missing[0]}.jpg). "
            "Run again with --refresh to review it again."
        )


async def _ensure_web_video(
    source: Path, folder: Path, settings: Settings, refresh: bool, max_file_bytes: int
) -> None:
    """H.264 + AAC, long edge at most 1280 px, moov atom first so playback starts at once.
    Made again only when missing, older than the source, or with --refresh."""
    target = folder / WEB_VIDEO_FILE
    if refresh or not await asyncio.to_thread(_is_newer, target, source):
        metadata = await _probe_or_skip(source, settings)
        await asyncio.to_thread(_transcode_for_web, source, target, metadata, settings)
    size = await asyncio.to_thread(lambda: target.stat().st_size)
    if size > max_file_bytes:
        raise ExportError(
            f"The web copy of {source.name} is {megabytes(size)}, over the "
            f"{megabytes(max_file_bytes)} limit for a single file. Trim the video, then run "
            "the export again."
        )


def _is_newer(target: Path, source: Path) -> bool:
    return target.is_file() and target.stat().st_mtime >= source.stat().st_mtime


def _transcode_for_web(
    source: Path, target: Path, metadata: VideoMetadata, settings: Settings
) -> None:
    edge = WEB_VIDEO_LONG_EDGE
    source_input = ffmpeg.input(str(source))
    picture = source_input[str(metadata.video_stream_index)].filter(
        "scale",
        # Never upscale, and keep both sides even, which H.264 in yuv420p requires.
        w=f"if(gte(iw,ih),trunc(min(iw,{edge})/2)*2,-2)",
        h=f"if(gte(iw,ih),-2,trunc(min(ih,{edge})/2)*2)",
    )
    streams: list[Any] = [picture]
    options: dict[str, Any] = {
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
        "crf": 26,
        "preset": "medium",
        "movflags": "+faststart",
    }
    if metadata.audio_stream_index is not None:
        streams.append(source_input[str(metadata.audio_stream_index)])
        options.update({"acodec": "aac", "b:a": "128k", "ac": 2})
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part.mp4")
    try:
        (
            ffmpeg.output(*streams, str(partial), **options)
            .global_args("-hide_banner", "-nostdin", "-loglevel", "error")
            .overwrite_output()
            .run(cmd=settings.ffmpeg_cmd, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        detail = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise ExportError(
            f"ffmpeg couldn't make a web copy of {source.name}: {detail or 'no details'}. "
            "The export needs an ffmpeg build with the libx264 encoder."
        ) from exc
    partial.replace(target)


def _write_site(
    out_dir: Path,
    jobs: dict[str, Job],
    settings: Settings,
    repo_url: str | None,
    max_file_bytes: int,
    result: ExportResult,
) -> None:
    videos_out = out_dir / "videos"
    videos_out.mkdir(parents=True, exist_ok=True)
    # Unpublish videos that were removed from the source folder or couldn't be checked.
    for folder in videos_out.iterdir():
        if folder.is_dir() and folder.name not in jobs:
            shutil.rmtree(folder)

    static_out = out_dir / "static"
    shutil.rmtree(static_out, ignore_errors=True)
    shutil.copytree(WEB_DIR / "static", static_out)

    ordered = sorted(
        jobs.items(), key=lambda item: (not item[1].is_example, item[1].created_at), reverse=True
    )
    slugs = {job.job_id: slug for slug, job in ordered}
    shared = {"static_mode": True, "repo_url": repo_url}

    home_links = static_links(slugs, depth=0)
    home = templates.env.get_template("home.html").render(
        **shared,
        links=home_links,
        rows=video_rows([job for _, job in ordered], settings, home_links),
        has_active=False,
        checks=check_definitions(settings),
        reviewer_connected=True,
    )
    (out_dir / "index.html").write_text(home, encoding="utf-8")

    page_template = templates.env.get_template("video.html")
    for slug, job in ordered:
        links = static_links(slugs, depth=2)
        html = page_template.render(
            **shared,
            links=links,
            page=build_video_page(job, settings, links),
            agency_name=settings.agency_name,
            logo_description=settings.agency_logo_description,
        )
        (videos_out / slug / "index.html").write_text(html, encoding="utf-8")
        result.published.append(slug)

    too_big: list[str] = []
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        result.total_bytes += size
        result.file_count += 1
        if size > result.largest[1]:
            result.largest = (path.relative_to(out_dir).as_posix(), size)
        if size > max_file_bytes:
            too_big.append(f"{path.relative_to(out_dir).as_posix()} ({megabytes(size)})")
    if too_big:
        raise ExportError(
            f"These files are over the {megabytes(max_file_bytes)} limit: {', '.join(too_big)}"
        )


# --- Command line -----------------------------------------------------------------------------


def _default_repo_url() -> str | None:
    """The GitHub page for this checkout's ``origin`` remote, if there is one."""
    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.fullmatch(
        r"(?:https://github\.com/|git@github\.com:)([\w.-]+/[\w.-]+?)(?:\.git)?", remote
    )
    return f"https://github.com/{match.group(1)}" if match else None


def _reviewer_configured(settings: Settings) -> bool:
    return settings.anthropic_api_key is not None or bool(os.environ.get("ANTHROPIC_AUTH_TOKEN"))


async def _run(args: argparse.Namespace) -> ExportResult:
    settings = Settings()
    common: dict[str, Any] = {
        "settings": settings,
        "refresh": args.refresh,
        "repo_url": args.repo_url,
        "include_example": args.include_example,
    }
    if not _reviewer_configured(settings):
        return await export_showcase(args.videos, args.out, agent=None, **common)
    key = settings.anthropic_api_key
    async with (
        AsyncAnthropic(api_key=key.get_secret_value()) if key else AsyncAnthropic()
    ) as client:
        return await export_showcase(
            args.videos, args.out, agent=ClaudeQAAgent(client, settings), **common
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m video_qa.export",
        description="Check the videos in a folder and write a static, read-only showcase.",
    )
    parser.add_argument("--videos", type=Path, default=Path("demo_videos"))
    parser.add_argument("--out", type=Path, default=Path("docs"))
    parser.add_argument(
        "--refresh", action="store_true", help="review every video again (spends API calls)"
    )
    parser.add_argument(
        "--include-example",
        action="store_true",
        help="also publish the hand-written Example (a labelled test pattern, no AI)",
    )
    parser.add_argument(
        "--repo-url",
        default=_default_repo_url(),
        help="link shown on the showcase home page (default: this checkout's GitHub remote)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        result = asyncio.run(_run(args))
    except ExportError as exc:
        print(f"\nExport stopped: {exc}", file=sys.stderr)
        return 1

    print(f"\nPublished {len(result.published)} video(s) to {args.out}:")
    for slug in result.published:
        if slug == EXAMPLE_SLUG:
            note = "hand-written example, no AI"
        else:
            note = "reviewed now" if slug in result.reviewed else "saved review reused"
        print(f"  videos/{slug}/  ({note})")
    if result.skipped:
        print("\nSkipped, because they couldn't be checked (not published):")
        for name, reason in result.skipped:
            print(f"  {name}: {reason}")
    largest_name, largest_size = result.largest
    print(
        f"\nSite size: {megabytes(result.total_bytes)} in {result.file_count} files "
        f"(largest: {largest_name}, {megabytes(largest_size)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
