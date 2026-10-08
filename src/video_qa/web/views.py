"""View models: what the pages and the JSON API show about a job.

Check logic lives in ``video_qa.findings``; this module only arranges its output for display
(stepper states, timeline geometry, thumbnails, technical details).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from video_qa.config import Settings
from video_qa.findings import (
    CheckResult,
    Finding,
    Verdict,
    build_checks,
    editor_summary,
    summarize,
)
from video_qa.jobs import Job, JobStage, JobStatus, utc_now
from video_qa.schemas import format_timestamp

JOBS_PATH = "/v1/qa-jobs"

STAGE_LABELS: dict[JobStage, str] = {
    JobStage.RECEIVED: "Uploaded",
    JobStage.READING: "Reading the video",
    JobStage.SPLITTING: "Splitting it into frames",
    JobStage.LISTENING: "Listening to what's said",
    JobStage.REVIEWING: "Reviewing against the checklist",
    JobStage.DONE: "Done",
}
STAGE_ORDER = list(JobStage)

PageState = Literal["checking", "passed", "failed", "error"]

# Findings on these checks point at sampled frames, so each one stands for a whole interval.
_FRAME_SAMPLED_CHECKS = {"logo", "footage"}


@dataclass(frozen=True)
class Links:
    """Where pages, media and static assets live.

    On localhost these are absolute paths served by the API. In the static showcase they are
    relative file paths from the page being rendered (``root`` climbs back to the site root),
    so the site works under any prefix, such as GitHub Pages' ``/<repo>/``.
    """

    root: str = "/"
    slugs: Mapping[str, str] | None = None  # static showcase only: job_id -> folder name

    @property
    def static(self) -> bool:
        return self.slugs is not None

    def home(self) -> str:
        return self.root or "./"

    def asset(self, name: str) -> str:
        return f"{self.root}static/{name}"

    def _folder(self, job_id: str) -> str:
        assert self.slugs is not None
        return f"{self.root}videos/{self.slugs[job_id]}/"

    def video_page(self, job_id: str) -> str:
        return self._folder(job_id) if self.static else f"/videos/{job_id}"

    def video_file(self, job: Job) -> str:
        if self.static:
            return self._folder(job.job_id) + job.video_file
        return f"{JOBS_PATH}/{job.job_id}/video"

    def frame(self, job_id: str, index: int) -> str:
        if self.static:
            return f"{self._folder(job_id)}frames/{index}.jpg"
        return f"{JOBS_PATH}/{job_id}/frames/{index}.jpg"

    def logo(self, job: Job) -> str | None:
        if job.logo_file is None:
            return None
        if self.static:
            return self._folder(job.job_id) + job.logo_file
        return f"{JOBS_PATH}/{job.job_id}/logo"


LIVE_LINKS = Links()


def static_links(slugs: Mapping[str, str], *, depth: int) -> Links:
    """Links for a static page ``depth`` folders below the site root."""
    return Links(root="../" * depth, slugs=slugs)


def when_label(moment: datetime, now: datetime | None = None, *, absolute: bool = False) -> str:
    """ "Today, 14:05" on localhost; a plain date in the showcase, where "today" would go stale."""
    local = moment.astimezone()
    if absolute:
        return f"{local.day} {local.strftime('%b %Y')}"
    today = (now or utc_now()).astimezone().date()
    clock = local.strftime("%H:%M")
    if local.date() == today:
        return f"Today, {clock}"
    if local.date() == today - timedelta(days=1):
        return f"Yesterday, {clock}"
    return f"{local.day} {local.strftime('%b')}, {clock}"


def job_checks(job: Job, settings: Settings) -> list[CheckResult]:
    if job.report is None:
        return []
    return build_checks(
        job.report,
        settings,
        frame_timestamps=[frame.timestamp_seconds for frame in job.frames],
        duration_seconds=job.duration_seconds,
    )


def page_state(job: Job, verdict: Verdict | None) -> PageState:
    if job.status is JobStatus.ERROR:
        return "error"
    if not job.finished or verdict is None:
        return "checking"
    return "passed" if verdict.passed else "failed"


def status_label(job: Job, verdict: Verdict | None) -> str:
    state = page_state(job, verdict)
    if state == "error":
        return "Couldn't check"
    if state == "checking":
        if job.status is JobStatus.QUEUED:
            return "Waiting its turn"
        return f"Checking · {STAGE_LABELS[_active_stage(job)]}"
    assert verdict is not None
    return verdict.short_label


# --- JSON API ---------------------------------------------------------------------------------


class JobSummary(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    status_label: str
    video_name: str
    editor_name: str | None
    is_example: bool
    created_at: datetime
    finished_at: datetime | None
    duration_seconds: float | None
    passed_qa: bool | None
    issue_count: int | None
    thumbnail_url: str | None


class JobDetail(Job):
    status_label: str
    verdict: Verdict | None
    checks: list[CheckResult]
    editor_summary: str | None


def job_summary(job: Job, settings: Settings) -> JobSummary:
    checks = job_checks(job, settings)
    verdict = summarize(checks) if checks else None
    return JobSummary(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        status_label=status_label(job, verdict),
        video_name=job.video_name,
        editor_name=job.editor_name,
        is_example=job.is_example,
        created_at=job.created_at,
        finished_at=job.finished_at,
        duration_seconds=job.duration_seconds,
        passed_qa=verdict.passed if verdict else None,
        issue_count=verdict.issue_count if verdict else None,
        thumbnail_url=LIVE_LINKS.frame(job.job_id, 0) if job.frames else None,
    )


def job_detail(job: Job, settings: Settings) -> JobDetail:
    checks = job_checks(job, settings)
    verdict = summarize(checks) if checks else None
    return JobDetail.model_validate(
        {
            **job.model_dump(exclude={"model_passed_qa"}),
            "status_label": status_label(job, verdict),
            "verdict": verdict,
            "checks": checks,
            "editor_summary": (
                editor_summary(job.report, job.video_name, settings) if job.report else None
            ),
        }
    )


# --- Home page --------------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoRow:
    job_id: str
    url: str
    video_name: str
    editor_name: str | None
    when: str
    length: str | None
    state: PageState
    status_label: str
    thumbnail_url: str | None
    is_example: bool


def video_rows(jobs: list[Job], settings: Settings, links: Links = LIVE_LINKS) -> list[VideoRow]:
    rows: list[VideoRow] = []
    for job in jobs:
        checks = job_checks(job, settings)
        verdict = summarize(checks) if checks else None
        rows.append(
            VideoRow(
                job_id=job.job_id,
                url=links.video_page(job.job_id),
                video_name=job.video_name,
                editor_name=job.editor_name,
                when=when_label(job.created_at, absolute=links.static),
                length=format_timestamp(job.duration_seconds) if job.duration_seconds else None,
                state=page_state(job, verdict),
                status_label=status_label(job, verdict),
                thumbnail_url=links.frame(job.job_id, 0) if job.frames else None,
                is_example=job.is_example,
            )
        )
    return rows


# --- Video page -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    stage: JobStage
    label: str
    state: Literal["done", "current", "upcoming"]


def _active_stage(job: Job) -> JobStage:
    # "received" means the upload is done and reading is next.
    if job.stage is JobStage.RECEIVED:
        return JobStage.READING
    return job.stage


def build_steps(job: Job) -> list[Step]:
    active = STAGE_ORDER.index(_active_stage(job))
    steps: list[Step] = []
    for index, stage in enumerate(STAGE_ORDER):
        if job.stage is JobStage.DONE or index < active:
            state: Literal["done", "current", "upcoming"] = "done"
        elif index == active:
            state = "current"
        else:
            state = "upcoming"
        steps.append(Step(stage=stage, label=STAGE_LABELS[stage], state=state))
    return steps


@dataclass(frozen=True)
class TimelineBlock:
    left: float
    width: float
    seek: float
    label: str


@dataclass(frozen=True)
class TimelineMark:
    left: float
    label: str
    kind: str


@dataclass(frozen=True)
class TimelineLane:
    key: str
    label: str
    passed: bool
    blocks: list[TimelineBlock]
    marks: list[TimelineMark]


@dataclass(frozen=True)
class Tick:
    left: float
    label: str


@dataclass(frozen=True)
class Timeline:
    duration: float
    lanes: list[TimelineLane]
    ticks: list[Tick]


def seek_point(seconds: float) -> float:
    """Where a "jump to this problem" lands: one second early, so the lead-in is visible."""
    return round(max(0.0, seconds - 1.0), 3)


def build_timeline(
    checks: list[CheckResult], duration: float | None, interval: float
) -> Timeline | None:
    if not duration or duration <= 0:
        return None

    def pct(seconds: float) -> float:
        return round(min(max(seconds / duration, 0.0), 1.0) * 100, 3)

    lanes: list[TimelineLane] = []
    for check in checks:
        blocks: list[TimelineBlock] = []
        for finding in check.findings:
            start = min(max(finding.start_seconds, 0.0), duration)
            end = finding.end_seconds if finding.end_seconds is not None else start
            if check.key in _FRAME_SAMPLED_CHECKS:
                end += interval
            end = min(max(end, start + interval), duration)
            start = max(0.0, min(start, end - interval))
            seek = seek_point(finding.start_seconds)
            blocks.append(
                TimelineBlock(
                    left=pct(start),
                    width=round(pct(end) - pct(start), 3),
                    seek=seek,
                    label=f"{finding.headline}. Play from {format_timestamp(seek)}.",
                )
            )
        marks = [
            TimelineMark(left=pct(marker.seconds), label=marker.label, kind=marker.kind)
            for marker in check.markers
        ]
        lanes.append(
            TimelineLane(
                key=check.key,
                label=check.lane_label,
                passed=check.passed,
                blocks=blocks,
                marks=marks,
            )
        )

    step = next((s for s in (1, 2, 5, 10, 15, 30, 60, 120) if duration / s <= 10), 300)
    ticks = [
        Tick(left=pct(t), label=format_timestamp(t)) for t in range(0, int(duration) + 1, step)
    ]
    return Timeline(duration=duration, lanes=lanes, ticks=ticks)


@dataclass(frozen=True)
class FindingView:
    finding: Finding
    seek: float
    thumbnail_url: str | None
    thumbnail_alt: str


@dataclass(frozen=True)
class CheckView:
    check: CheckResult
    findings: list[FindingView]


def _check_views(job: Job, checks: list[CheckResult], links: Links) -> list[CheckView]:
    views: list[CheckView] = []
    for check in checks:
        findings: list[FindingView] = []
        for finding in check.findings:
            index = finding.frame_index
            has_frame = index is not None and 0 <= index < len(job.frames)
            findings.append(
                FindingView(
                    finding=finding,
                    seek=seek_point(finding.start_seconds),
                    thumbnail_url=links.frame(job.job_id, index)
                    if has_frame and index is not None
                    else None,
                    thumbnail_alt=(
                        f"Frame at {format_timestamp(job.frames[index].timestamp_seconds)}"
                        if has_frame and index is not None
                        else ""
                    ),
                )
            )
        views.append(CheckView(check=check, findings=findings))
    return views


@dataclass(frozen=True)
class StageTiming:
    label: str
    started: str
    took: str


@dataclass(frozen=True)
class Technical:
    job_id: str | None
    model: str
    status: str
    stage: str
    model_verdict: str
    enforced_verdict: str
    report_json: str | None
    model_report_json: str | None
    transcript: list[str]
    transcript_simulated: bool
    stage_timings: list[StageTiming]
    error: str | None
    notification: str | None
    frame_count: int


def _verdict_word(passed: bool | None) -> str:
    if passed is None:
        return "none (no report)"
    return "passed" if passed else "failed"


def _technical(job: Job, settings: Settings, *, public: bool) -> Technical:
    starts = sorted(job.stage_started_at.items(), key=lambda item: STAGE_ORDER.index(item[0]))
    timings: list[StageTiming] = []
    for position, (stage, started) in enumerate(starts):
        if stage is JobStage.DONE:
            took = ""
        else:
            following = starts[position + 1][1] if position + 1 < len(starts) else None
            until = following or job.finished_at or utc_now()
            took = f"{(until - started).total_seconds():.1f} s"
        timings.append(
            StageTiming(
                label=f"{stage.value} ({STAGE_LABELS[stage]})",
                started=started.astimezone().strftime("%H:%M:%S"),
                took=took,
            )
        )
    notification: str | None
    if settings.notifier == "slack":
        notification = (
            f"NOTIFIER=slack; editor_notified={job.editor_notified}; "
            f"notification_error={job.notification_error}"
        )
    else:
        notification = "NOTIFIER=none: nothing is sent to Slack."
    if public:
        # The public showcase leaves out what only means something on the machine that ran
        # the check: ids, raw errors and local configuration.
        notification = None
    return Technical(
        job_id=None if public else job.job_id,
        model=job.model or ("none: hand-written example" if job.is_example else "not reached"),
        status=job.status.value,
        stage=job.stage.value,
        model_verdict=(
            "not applicable: hand-written example"
            if job.is_example
            else _verdict_word(job.model_passed_qa)
        ),
        enforced_verdict=_verdict_word(job.report.passed_qa if job.report else None),
        report_json=job.report.model_dump_json(indent=2) if job.report else None,
        model_report_json=job.model_report.model_dump_json(indent=2) if job.model_report else None,
        transcript=[
            f"[{c.start_seconds:.2f}–{c.end_seconds:.2f} s] {c.text}" for c in job.transcript
        ],
        transcript_simulated=job.transcript_simulated,
        stage_timings=timings,
        error=None if public else job.error,
        notification=notification,
        frame_count=len(job.frames),
    )


@dataclass(frozen=True)
class VideoPage:
    job: Job
    state: PageState
    meta: list[str]
    steps: list[Step]
    verdict: Verdict | None
    checks: list[CheckView]
    timeline: Timeline | None
    summary_text: str | None
    summary_rows: int
    video_url: str
    poster_url: str | None
    logo_url: str | None
    technical: Technical
    technical_json: str | None
    links: Links


def _rows_for(text: str | None, width: int = 90) -> int:
    """Enough textarea rows to show ``text`` without scrolling, wrapped at about ``width``."""
    if not text:
        return 3
    return sum(len(line) // width + 1 for line in text.splitlines()) + 1


def build_video_page(job: Job, settings: Settings, links: Links = LIVE_LINKS) -> VideoPage:
    checks = job_checks(job, settings)
    verdict = summarize(checks) if checks else None
    summary = editor_summary(job.report, job.video_name, settings) if job.report else None
    meta: list[str] = []
    if job.editor_name:
        meta.append(f"Edited by {job.editor_name}")
    if links.static:
        meta.append(f"Checked {when_label(job.created_at, absolute=True)}")
    else:
        meta.append(f"Uploaded {when_label(job.created_at)}")
    if job.duration_seconds:
        meta.append(f"{format_timestamp(job.duration_seconds)} long")
    return VideoPage(
        job=job,
        state=page_state(job, verdict),
        meta=meta,
        steps=build_steps(job),
        verdict=verdict,
        checks=_check_views(job, checks, links),
        timeline=build_timeline(checks, job.duration_seconds, settings.frame_interval_seconds),
        summary_text=summary,
        summary_rows=_rows_for(summary),
        video_url=links.video_file(job),
        poster_url=links.frame(job.job_id, 0) if job.frames else None,
        logo_url=links.logo(job),
        technical=_technical(job, settings, public=links.static),
        technical_json=None
        if links.static
        else json.dumps(
            job.model_dump(mode="json", exclude={"report", "model_report", "frames"}), indent=2
        ),
        links=links,
    )
