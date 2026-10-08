"""Plain-language results for people who don't read JSON.

Turns a ``VideoQAReport`` into four ``CheckResult`` objects (always logo, hook, footage,
silence, in that order), a verdict, the copyable summary for the editor, and one plain sentence
for every way a check can fail to run. The dashboard renders only what this module produces;
no check logic lives in templates or JavaScript.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from video_qa.agent import QAAgentError, QARefusalError, QARequestTooLargeError, QATruncatedError
from video_qa.config import Settings
from video_qa.deconstruction import (
    FFmpegError,
    NoVideoStreamError,
    StillImageError,
    UnreadableVideoError,
    VideoRejectedError,
    VideoTooLongError,
)
from video_qa.notifier import collapse_timestamps, failure_lines
from video_qa.schemas import VideoQAReport, format_timestamp

CheckKey = Literal["logo", "hook", "footage", "silence"]


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class Basis(StrEnum):
    MEASURED = "measured"  # arithmetic on timestamps
    AI_JUDGEMENT = "ai_judgement"  # the model's opinion of what it saw


BASIS_LABELS = {Basis.MEASURED: "Measured from the audio", Basis.AI_JUDGEMENT: "AI's judgement"}


class Finding(BaseModel):
    start_seconds: float
    end_seconds: float | None = None
    headline: str
    detail: str
    how_to_fix: str
    said: str | None = Field(default=None, description="Footage mismatches: what was said.")
    shown: str | None = Field(default=None, description="Footage mismatches: what was shown.")
    frame_index: int | None = Field(
        default=None, description="The sampled frame closest to start_seconds, for a thumbnail."
    )

    @property
    def time_label(self) -> str:
        start = format_timestamp(self.start_seconds)
        if self.end_seconds is None:
            return start
        end = format_timestamp(self.end_seconds)
        return start if end == start else f"{start}–{end}"


class TimelineMarker(BaseModel):
    seconds: float
    label: str
    kind: Literal["deadline", "actual"]


class CheckDefinition(BaseModel):
    key: CheckKey
    title: str
    lane_label: str
    explanation: str
    basis: Basis

    @property
    def basis_label(self) -> str:
        return BASIS_LABELS[self.basis]


class CheckResult(CheckDefinition):
    status: CheckStatus
    outcome: str = Field(description="One sentence on what was found.")
    findings: list[Finding] = Field(default_factory=list)
    markers: list[TimelineMarker] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status is CheckStatus.PASSED


class Verdict(BaseModel):
    passed: bool
    checks_total: int
    checks_failed: int
    issue_count: int
    headline: str
    short_label: str


# --- Wording helpers --------------------------------------------------------------------------


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural_form or singular + 's'}"


def duration_words(seconds: float) -> str:
    """A length of time, as opposed to a point in the video: "1 second", "2.5 seconds"."""
    rounded = round(seconds, 1)
    text = f"{rounded:g}"
    return f"{text} second" if rounded == 1 else f"{text} seconds"


def megabytes(size_bytes: int) -> str:
    mb = size_bytes / (1024 * 1024)
    return f"{mb:.0f} MB" if mb >= 10 else f"{mb:.1f} MB"


def _every_interval(settings: Settings) -> str:
    interval = settings.frame_interval_seconds
    return "every second" if interval == 1 else f"every {duration_words(interval)}"


def _nearest_index(timestamps: Sequence[float], seconds: float) -> int | None:
    if not timestamps:
        return None
    return min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - seconds))


# --- The four checks --------------------------------------------------------------------------


def check_definitions(settings: Settings) -> list[CheckDefinition]:
    """What each check is and why, with thresholds from settings. Shared by the home page's
    "What we check" list and the report's check cards, so the wording never drifts."""
    deadline = format_timestamp(settings.hook_deadline_seconds)
    threshold = duration_words(settings.dead_air_threshold_seconds)
    return [
        CheckDefinition(
            key="logo",
            title="Logo stays on screen",
            lane_label="Logo",
            explanation=(
                f"{settings.agency_name}'s logo is a watermark that should stay visible for "
                f"the whole video, so we look for it {_every_interval(settings)}."
            ),
            basis=Basis.AI_JUDGEMENT,
        ),
        CheckDefinition(
            key="hook",
            title="Opening line starts quickly",
            lane_label="Opening line",
            explanation=(
                "The hook is the spoken opening line that makes people keep watching. "
                f"It has to start by {deadline}, before viewers scroll away."
            ),
            basis=Basis.MEASURED,
        ),
        CheckDefinition(
            key="footage",
            title="Footage matches what's being said",
            lane_label="Footage",
            explanation=(
                "B-roll is the extra footage shown while someone talks. When the speaker "
                "mentions something, the picture should show that, not something else."
            ),
            basis=Basis.AI_JUDGEMENT,
        ),
        CheckDefinition(
            key="silence",
            title="No silent gaps",
            lane_label="Silence",
            explanation=(
                "Dead air is a stretch where nobody speaks. "
                f"Gaps longer than {threshold} make viewers lose interest."
            ),
            basis=Basis.MEASURED,
        ),
    ]


def build_checks(
    report: VideoQAReport,
    settings: Settings,
    *,
    frame_timestamps: Sequence[float] = (),
    duration_seconds: float | None = None,
) -> list[CheckResult]:
    definitions = {d.key: d for d in check_definitions(settings)}
    checks = [
        _logo_check(definitions["logo"], report, settings, duration_seconds),
        _hook_check(definitions["hook"], report, settings),
        _footage_check(definitions["footage"], report),
        _silence_check(definitions["silence"], report, settings),
    ]
    for check in checks:
        for finding in check.findings:
            finding.frame_index = _nearest_index(frame_timestamps, finding.start_seconds)
    return checks


def _result(
    definition: CheckDefinition,
    *,
    outcome: str,
    findings: list[Finding],
    markers: list[TimelineMarker] | None = None,
    failed: bool | None = None,
) -> CheckResult:
    is_failed = bool(findings) if failed is None else failed
    return CheckResult(
        **definition.model_dump(),
        status=CheckStatus.FAILED if is_failed else CheckStatus.PASSED,
        outcome=outcome,
        findings=findings,
        markers=markers or [],
    )


def _logo_check(
    definition: CheckDefinition,
    report: VideoQAReport,
    settings: Settings,
    duration_seconds: float | None,
) -> CheckResult:
    findings: list[Finding] = []
    for start, end in collapse_timestamps(
        report.logo_missing_timestamps, interval_seconds=settings.frame_interval_seconds
    ):
        first, last = format_timestamp(start), format_timestamp(end)
        if first == last:
            findings.append(
                Finding(
                    start_seconds=start,
                    headline=f"Logo missing at {first}",
                    detail=f"The logo isn't visible at {first}.",
                    how_to_fix="Put the watermark back on that shot.",
                )
            )
        else:
            findings.append(
                Finding(
                    start_seconds=start,
                    end_seconds=end,
                    headline=f"Logo missing from {first} to {last}",
                    detail=f"The logo disappears from {first} to {last}.",
                    how_to_fix="Put the watermark back on those shots.",
                )
            )
    if not report.logo_present and not findings:
        findings.append(
            Finding(
                start_seconds=0.0,
                end_seconds=duration_seconds,
                headline="Logo not found",
                detail="The logo wasn't visible anywhere in the video.",
                how_to_fix="Add the watermark to the whole video.",
            )
        )
    if not report.logo_present:
        outcome = "The logo wasn't found anywhere in the video."
    elif findings:
        outcome = f"The logo is missing in {plural(len(findings), 'place')}."
    else:
        outcome = f"The logo is on screen {_every_interval(settings)} we checked."
    return _result(definition, outcome=outcome, findings=findings)


def _hook_check(
    definition: CheckDefinition, report: VideoQAReport, settings: Settings
) -> CheckResult:
    deadline_seconds = settings.hook_deadline_seconds
    deadline = format_timestamp(deadline_seconds)
    hook = report.hook_time_seconds
    markers = [
        TimelineMarker(seconds=deadline_seconds, label=f"Must start by {deadline}", kind="deadline")
    ]
    if hook is None:
        finding = Finding(
            start_seconds=0.0,
            end_seconds=deadline_seconds,
            headline="No spoken opening line",
            detail=f"No spoken opening line was found, so nothing grabs viewers before {deadline}.",
            how_to_fix=f"Add a spoken opening line that starts by {deadline}.",
        )
        return _result(
            definition,
            outcome="No spoken opening line was found.",
            findings=[finding],
            markers=markers,
        )

    starts = format_timestamp(hook)
    markers.append(TimelineMarker(seconds=hook, label=f"Starts {starts}", kind="actual"))
    if hook <= deadline_seconds:
        return _result(
            definition,
            outcome=f"The opening line starts at {starts}, in time for the {deadline} deadline.",
            findings=[],
            markers=markers,
        )

    # 3.4 s would print as "0:03" right next to a 0:03 deadline, which reads as a pass.
    just_after = starts == deadline
    when = f"just after {deadline}" if just_after else f"at {starts}"
    until = f"just after {deadline}" if just_after else starts
    finding = Finding(
        start_seconds=deadline_seconds,
        end_seconds=hook,
        headline=f"Opening line starts {when}",
        detail=(
            f"The opening line doesn't start until {until}, "
            f"but it needs to start by {deadline} so viewers don't scroll away."
        ),
        how_to_fix=f"Trim the start, or move the first line earlier, so it begins by {deadline}.",
    )
    outcome = (
        f"The opening line starts just after the {deadline} deadline."
        if just_after
        else f"The opening line starts at {starts}, after the {deadline} deadline."
    )
    return _result(definition, outcome=outcome, findings=[finding], markers=markers)


def _footage_check(definition: CheckDefinition, report: VideoQAReport) -> CheckResult:
    findings = [
        Finding(
            start_seconds=mismatch.timestamp_seconds,
            headline=f"Footage doesn't match at {format_timestamp(mismatch.timestamp_seconds)}",
            detail=(
                f"At {format_timestamp(mismatch.timestamp_seconds)} the picture doesn't show "
                "what's being talked about."
            ),
            how_to_fix=(
                f"Swap in a shot at {format_timestamp(mismatch.timestamp_seconds)} that shows "
                "what the speaker is talking about."
            ),
            said=mismatch.spoken,
            shown=mismatch.shown,
        )
        for mismatch in sorted(report.audio_visual_mismatch, key=lambda m: m.timestamp_seconds)
    ]
    outcome = (
        f"Found {plural(len(findings), 'place')} where the footage doesn't match what's said."
        if findings
        else "The footage matches what's being said throughout."
    )
    return _result(definition, outcome=outcome, findings=findings)


def _silence_check(
    definition: CheckDefinition, report: VideoQAReport, settings: Settings
) -> CheckResult:
    threshold = duration_words(settings.dead_air_threshold_seconds)
    findings: list[Finding] = []
    for segment in sorted(report.dead_air, key=lambda s: s.start_seconds):
        start = format_timestamp(segment.start_seconds)
        end = format_timestamp(segment.end_seconds)
        length = duration_words(segment.end_seconds - segment.start_seconds)
        findings.append(
            Finding(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                headline=f"Silent from {start} to {end}",
                detail=f"Nobody speaks for {length}, from {start} to {end}.",
                how_to_fix="Tighten the cut to close the gap, or add voice-over.",
            )
        )
    outcome = (
        f"Found {plural(len(findings), 'silent gap')} longer than {threshold}."
        if findings
        else f"There are no silent gaps longer than {threshold}."
    )
    return _result(definition, outcome=outcome, findings=findings)


def summarize(checks: Sequence[CheckResult]) -> Verdict:
    failed = [check for check in checks if not check.passed]
    issues = sum(len(check.findings) for check in failed)
    total = len(checks)
    if not failed:
        return Verdict(
            passed=True,
            checks_total=total,
            checks_failed=0,
            issue_count=0,
            headline=f"Ready to post: passed all {total} checks",
            short_label="Ready to post",
        )
    return Verdict(
        passed=False,
        checks_total=total,
        checks_failed=len(failed),
        issue_count=issues,
        headline=f"Needs fixes: {len(failed)} of {total} checks failed",
        short_label=f"Needs fixes · {plural(issues, 'thing')}",
    )


def editor_summary(report: VideoQAReport, video_name: str, settings: Settings) -> str:
    """The Slack message as plain text, minus the user tag, for pasting anywhere."""
    lines = [
        _unescape_slack(line)
        for line in failure_lines(
            report,
            hook_deadline_seconds=settings.hook_deadline_seconds,
            frame_interval_seconds=settings.frame_interval_seconds,
        )
    ]
    if report.passed_qa and not lines:
        out = [f"QA passed for {video_name}: all 4 checks passed."]
    else:
        out = [f"QA failed for {video_name}", *lines]
    feedback = report.feedback_for_editor.strip()
    if feedback:
        out += ["", "Editor feedback", feedback]
    return "\n".join(out)


def _unescape_slack(text: str) -> str:
    # Exact inverse of notifier.escape_slack_text, which escapes & first.
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


# --- Why a video couldn't be checked ----------------------------------------------------------

REVIEWER_NOT_CONNECTED = "The AI reviewer isn't connected yet, so new videos can't be checked."


def describe_failure(exc: BaseException) -> str:
    """One plain sentence for the person looking at the dashboard. The raw error is kept
    separately for the technical details."""
    if isinstance(exc, VideoTooLongError):
        return (
            f"This video is {format_timestamp(exc.duration_seconds)} long, and the longest "
            f"video we can check is {format_timestamp(exc.max_seconds)}. "
            "Please upload a shorter cut."
        )
    if isinstance(exc, NoVideoStreamError):
        return (
            "This file has sound but no picture, so there's nothing to check. "
            "Please upload the finished video file."
        )
    if isinstance(exc, StillImageError):
        return "This is a still image, not a video. Please upload the finished video file."
    if isinstance(exc, UnreadableVideoError):
        return (
            "This file couldn't be opened as a video. "
            "Please upload the finished video, for example an .mp4 file."
        )
    if isinstance(exc, VideoRejectedError | FFmpegError):
        return "Something went wrong reading this video. Please export it again and re-upload it."
    if isinstance(exc, QARefusalError):
        return (
            "The AI reviewer declined to review this video, so there's no verdict. "
            "A person will need to check it by hand."
        )
    if isinstance(exc, QATruncatedError):
        return (
            "The AI reviewer's answer was cut off before it finished, so the check is "
            "incomplete. Please try this video again."
        )
    if isinstance(exc, QARequestTooLargeError):
        return "This video is too big to send to the AI reviewer in one go. Try a shorter cut."
    if isinstance(exc, QAAgentError):
        return "The AI reviewer's answer couldn't be read. Please try this video again."
    if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return (
            "The AI reviewer isn't connected: its API key was turned down. "
            "Check the key in the settings, then try again."
        )
    if isinstance(exc, anthropic.CredentialsError) or (
        isinstance(exc, TypeError) and "authentication method" in str(exc)
    ):
        return (
            "The AI reviewer isn't connected: no API key is set up. "
            "Add the key, restart the app, then try again."
        )
    if isinstance(exc, anthropic.APIConnectionError):
        return "The AI reviewer couldn't be reached. Check the internet connection, then try again."
    if isinstance(exc, anthropic.APIStatusError) and (
        exc.status_code == 429 or exc.status_code >= 500
    ):
        return "The AI reviewer is busy right now. Please try again in a few minutes."
    if isinstance(exc, anthropic.NotFoundError):
        return "The AI reviewer named in the settings doesn't exist. Check the model name."
    if isinstance(exc, anthropic.APIStatusError):
        return "The AI reviewer turned this request down. The technical details say why."
    if isinstance(exc, asyncio.CancelledError):
        return "The checker was stopped before this video was finished. Please upload it again."
    return (
        "Something went wrong on our side while checking this video. "
        "The technical details say what happened."
    )


def describe_upload_too_large(max_bytes: int, size_bytes: int | None = None) -> str:
    if size_bytes is not None:
        return (
            f"This file is {megabytes(size_bytes)}, and the largest file we can take is "
            f"{megabytes(max_bytes)}. Please export a smaller file."
        )
    return (
        f"This file is bigger than {megabytes(max_bytes)}, the largest file we can take. "
        "Please export a smaller file."
    )
