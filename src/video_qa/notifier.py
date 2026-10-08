"""Slack notification for videos that fail QA."""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Sequence
from typing import Protocol

import httpx

from video_qa.schemas import VideoQAReport, format_timestamp

logger = logging.getLogger(__name__)

SLACK_USER_ID = re.compile(r"^[UW][A-Z0-9]+$")


class Notifier(Protocol):
    async def notify(self, report: VideoQAReport, video_name: str, editor_slack_id: str) -> bool:
        """Tell the editor about a failed report. Returns whether a message went out."""
        ...


class InvalidSlackUserIdError(ValueError):
    pass


def validate_slack_user_id(user_id: str) -> str:
    if not SLACK_USER_ID.fullmatch(user_id):
        raise InvalidSlackUserIdError(
            f"{user_id!r} is not a Slack user ID (expected U or W followed by A-Z/0-9)"
        )
    return user_id


def escape_slack_text(text: str) -> str:
    """Escape the three characters Slack treats as control sequences, so text written by the
    model (or a filename) can't smuggle in ``<!channel>``, ``<@U…>`` or links."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def collapse_timestamps(
    timestamps: Sequence[float], *, interval_seconds: float
) -> list[tuple[float, float]]:
    """Merge timestamps of consecutive frames into (first, last) ranges."""
    ranges: list[tuple[float, float]] = []
    for t in sorted(set(timestamps)):
        if ranges and t - ranges[-1][1] <= interval_seconds * 1.5:
            ranges[-1] = (ranges[-1][0], t)
        else:
            ranges.append((t, t))
    return ranges


def failure_lines(
    report: VideoQAReport, *, hook_deadline_seconds: float, frame_interval_seconds: float
) -> list[str]:
    lines: list[str] = []
    for start, end in collapse_timestamps(
        report.logo_missing_timestamps, interval_seconds=frame_interval_seconds
    ):
        if start == end:
            lines.append(f"❌ Logo missing at {format_timestamp(start)}")
        else:
            lines.append(f"❌ Logo missing {format_timestamp(start)}–{format_timestamp(end)}")

    deadline = format_timestamp(hook_deadline_seconds)
    hook = report.hook_time_seconds
    if hook is None:
        lines.append(f"❌ No spoken hook (must start within {deadline})")
    elif hook > hook_deadline_seconds:
        # 3.4 s must not read as "0:03" next to a 0:03 deadline.
        precise = int(hook) == int(hook_deadline_seconds)
        lines.append(
            f"❌ Hook lands at {format_timestamp(hook, precise=precise)} "
            f"(must start within {deadline})"
        )

    for mismatch in report.audio_visual_mismatch:
        lines.append(
            f"❌ B-roll mismatch at {format_timestamp(mismatch.timestamp_seconds)} - speaker "
            f"mentions {escape_slack_text(mismatch.spoken)} but shows "
            f"{escape_slack_text(mismatch.shown)}"
        )

    for segment in report.dead_air:
        lines.append(
            f"❌ Dead air {format_timestamp(segment.start_seconds)}–"
            f"{format_timestamp(segment.end_seconds)}"
        )
    return lines


def build_failure_message(
    report: VideoQAReport,
    video_name: str,
    editor_slack_id: str,
    *,
    hook_deadline_seconds: float,
    frame_interval_seconds: float,
) -> str:
    validate_slack_user_id(editor_slack_id)
    lines = [f"<@{editor_slack_id}> QA failed for *{escape_slack_text(video_name)}*"]
    lines += failure_lines(
        report,
        hook_deadline_seconds=hook_deadline_seconds,
        frame_interval_seconds=frame_interval_seconds,
    )
    feedback = report.feedback_for_editor.strip()
    if feedback:
        lines += ["", "*Editor feedback*", escape_slack_text(feedback)]
    return "\n".join(lines)


class SlackNotifier:
    """Posts failures to a Slack incoming webhook.

    Without a webhook URL it runs as a mock: the payload is logged and kept in ``sent``.
    """

    def __init__(
        self,
        *,
        webhook_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        hook_deadline_seconds: float = 3.0,
        frame_interval_seconds: float = 1.0,
    ) -> None:
        if webhook_url and http_client is None:
            raise ValueError("http_client is required when webhook_url is set")
        self._webhook_url = webhook_url or None
        self._http = http_client
        self._hook_deadline = hook_deadline_seconds
        self._frame_interval = frame_interval_seconds
        # Mock mode only; bounded so a long-running mock server doesn't grow without limit.
        self.sent: deque[dict[str, str]] = deque(maxlen=200)

    @property
    def is_mock(self) -> bool:
        return self._webhook_url is None

    async def notify(self, report: VideoQAReport, video_name: str, editor_slack_id: str) -> bool:
        if report.passed_qa:
            return False
        payload = {
            "text": build_failure_message(
                report,
                video_name,
                editor_slack_id,
                hook_deadline_seconds=self._hook_deadline,
                frame_interval_seconds=self._frame_interval,
            )
        }
        if self._webhook_url is None or self._http is None:
            logger.info("Slack (mock, SLACK_WEBHOOK_URL unset) would post: %s", payload["text"])
            self.sent.append(payload)
            return True
        response = await self._http.post(self._webhook_url, json=payload)
        response.raise_for_status()
        return True
