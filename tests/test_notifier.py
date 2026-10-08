from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from video_qa.notifier import (
    InvalidSlackUserIdError,
    SlackNotifier,
    build_failure_message,
    collapse_timestamps,
    failure_lines,
)
from video_qa.schemas import AudioVisualMismatch, VideoQAReport

from helpers import segment

WEBHOOK = "https://hooks.slack.com/services/T000/B000/XXXX"


def failed_report(**changes: Any) -> VideoQAReport:
    report = VideoQAReport(
        passed_qa=False,
        logo_present=True,
        logo_missing_timestamps=[4.004],
        hook_time_seconds=5.2,
        audio_visual_mismatch=[
            AudioVisualMismatch(timestamp_seconds=15.015, spoken="kitchen", shown="bathroom")
        ],
        dead_air=[segment(21.3, 23.4)],
        feedback_for_editor="Re-time the hook so it opens on the first line.",
    )
    return report.model_copy(update=changes)


def lines_for(report: VideoQAReport) -> list[str]:
    return failure_lines(report, hook_deadline_seconds=3.0, frame_interval_seconds=1.0)


def test_failure_lines_match_the_slack_format() -> None:
    assert lines_for(failed_report()) == [
        "❌ Logo missing at 0:04",
        "❌ Hook lands at 0:05 (must start within 0:03)",
        "❌ B-roll mismatch at 0:15 - speaker mentions kitchen but shows bathroom",
        "❌ Dead air 0:21–0:23",
    ]


def test_full_message_tags_editor_and_appends_feedback() -> None:
    text = build_failure_message(
        failed_report(),
        "kitchen-ad-v3.mp4",
        "U0123ABCD",
        hook_deadline_seconds=3.0,
        frame_interval_seconds=1.0,
    )
    assert text.splitlines() == [
        "<@U0123ABCD> QA failed for *kitchen-ad-v3.mp4*",
        "❌ Logo missing at 0:04",
        "❌ Hook lands at 0:05 (must start within 0:03)",
        "❌ B-roll mismatch at 0:15 - speaker mentions kitchen but shows bathroom",
        "❌ Dead air 0:21–0:23",
        "",
        "*Editor feedback*",
        "Re-time the hook so it opens on the first line.",
    ]


def test_consecutive_logo_gaps_collapse_into_ranges() -> None:
    report = failed_report(
        logo_missing_timestamps=[4.004, 5.005, 6.006, 9.009],
        hook_time_seconds=1.0,
        audio_visual_mismatch=[],
        dead_air=[],
    )
    assert lines_for(report) == ["❌ Logo missing 0:04–0:06", "❌ Logo missing at 0:09"]
    assert collapse_timestamps([2.0, 1.0, 1.0, 5.0], interval_seconds=1.0) == [
        (1.0, 2.0),
        (5.0, 5.0),
    ]


def test_hook_lines() -> None:
    clean: dict[str, Any] = {
        "logo_missing_timestamps": [],
        "audio_visual_mismatch": [],
        "dead_air": [],
    }
    # 3.4 s would read "0:03" next to a 0:03 deadline, so it gets a decimal.
    assert lines_for(failed_report(hook_time_seconds=3.4, **clean)) == [
        "❌ Hook lands at 0:03.4 (must start within 0:03)"
    ]
    assert lines_for(failed_report(hook_time_seconds=None, **clean)) == [
        "❌ No spoken hook (must start within 0:03)"
    ]
    assert lines_for(failed_report(hook_time_seconds=2.9, **clean)) == []


def test_model_text_cannot_inject_mentions() -> None:
    report = failed_report(
        audio_visual_mismatch=[
            AudioVisualMismatch(timestamp_seconds=1.0, spoken="<!here>", shown="<https://x.y|z>")
        ],
        feedback_for_editor="<!channel> fix the logo & ping <@U999>",
    )
    text = build_failure_message(
        report, "<!everyone>.mp4", "U0123ABCD", hook_deadline_seconds=3.0, frame_interval_seconds=1
    )
    assert "<!" not in text
    assert "<@U999>" not in text
    assert "<https" not in text
    assert "&lt;!channel&gt; fix the logo &amp; ping &lt;@U999&gt;" in text
    assert "*&lt;!everyone&gt;.mp4*" in text
    assert text.startswith("<@U0123ABCD> ")  # the one mention we intend


@pytest.mark.parametrize("bad_id", ["", "u0123", "U0123 ABC", "<!channel>", "U012>", "C0123ABCD"])
def test_rejects_invalid_slack_user_ids(bad_id: str) -> None:
    with pytest.raises(InvalidSlackUserIdError):
        build_failure_message(
            failed_report(), "a.mp4", bad_id, hook_deadline_seconds=3.0, frame_interval_seconds=1
        )


async def test_mock_mode_logs_and_keeps_payload() -> None:
    notifier = SlackNotifier()
    assert notifier.is_mock
    assert await notifier.notify(failed_report(), "ad.mp4", "W0123ABCD") is True
    assert len(notifier.sent) == 1
    assert notifier.sent[0]["text"].startswith("<@W0123ABCD> QA failed for *ad.mp4*")
    assert "❌ Dead air 0:21–0:23" in notifier.sent[0]["text"].splitlines()


async def test_passing_report_sends_nothing() -> None:
    notifier = SlackNotifier()
    passed = failed_report(
        passed_qa=True,
        logo_missing_timestamps=[],
        hook_time_seconds=0.5,
        audio_visual_mismatch=[],
        dead_air=[],
    )
    assert await notifier.notify(passed, "ad.mp4", "U0123ABCD") is False
    assert list(notifier.sent) == []


async def test_webhook_mode_posts_text_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="ok")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        notifier = SlackNotifier(webhook_url=WEBHOOK, http_client=client)
        assert await notifier.notify(failed_report(), "ad.mp4", "U0123ABCD") is True

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == WEBHOOK
    payload = json.loads(requests[0].content)
    assert list(payload) == ["text"]
    assert payload["text"].splitlines()[1] == "❌ Logo missing at 0:04"
    assert list(notifier.sent) == []  # payloads are only kept in mock mode


async def test_webhook_errors_propagate() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    async with httpx.AsyncClient(transport=transport) as client:
        notifier = SlackNotifier(webhook_url=WEBHOOK, http_client=client)
        with pytest.raises(httpx.HTTPStatusError):
            await notifier.notify(failed_report(), "ad.mp4", "U0123ABCD")


def test_webhook_mode_requires_an_http_client() -> None:
    with pytest.raises(ValueError, match="http_client"):
        SlackNotifier(webhook_url=WEBHOOK)
