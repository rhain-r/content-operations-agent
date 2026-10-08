from __future__ import annotations

import asyncio
from typing import Any

import anthropic
import httpx2
import pytest

from video_qa.agent import QARefusalError, QATruncatedError
from video_qa.deconstruction import NoVideoStreamError, VideoTooLongError
from video_qa.findings import (
    Basis,
    CheckStatus,
    build_checks,
    check_definitions,
    describe_failure,
    describe_upload_too_large,
    duration_words,
    editor_summary,
    summarize,
)
from video_qa.schemas import AudioVisualMismatch, VideoQAReport, format_timestamp

from helpers import make_settings, segment


def report(**changes: Any) -> VideoQAReport:
    clean = VideoQAReport(
        passed_qa=True,
        logo_present=True,
        logo_missing_timestamps=[],
        hook_time_seconds=0.4,
        audio_visual_mismatch=[],
        dead_air=[],
        feedback_for_editor="Looks good.",
    )
    return clean.model_copy(update=changes)


def checks_for(r: VideoQAReport, **settings: Any) -> dict[str, Any]:
    return {c.key: c for c in build_checks(r, make_settings(**settings))}


def test_clean_report_passes_all_four_in_fixed_order() -> None:
    checks = build_checks(report(), make_settings())
    assert [c.key for c in checks] == ["logo", "hook", "footage", "silence"]
    assert [c.title for c in checks] == [
        "Logo stays on screen",
        "Opening line starts quickly",
        "Footage matches what's being said",
        "No silent gaps",
    ]
    assert all(c.status is CheckStatus.PASSED and c.findings == [] for c in checks)
    assert [c.basis for c in checks] == [
        Basis.AI_JUDGEMENT,
        Basis.MEASURED,
        Basis.AI_JUDGEMENT,
        Basis.MEASURED,
    ]
    assert checks[0].basis_label == "AI's judgement"
    assert checks[1].basis_label == "Measured from the audio"
    verdict = summarize(checks)
    assert verdict.passed
    assert verdict.headline == "Ready to post: passed all 4 checks"
    assert verdict.short_label == "Ready to post"


def test_logo_gaps_collapse_into_ranges() -> None:
    logo = checks_for(report(logo_missing_timestamps=[4.004, 5.005, 6.006, 9.009]))["logo"]
    assert logo.status is CheckStatus.FAILED
    assert logo.outcome == "The logo is missing in 2 places."
    first, second = logo.findings
    assert first.headline == "Logo missing from 0:04 to 0:06"
    assert first.detail == "The logo disappears from 0:04 to 0:06."
    assert first.how_to_fix == "Put the watermark back on those shots."
    assert (first.start_seconds, first.end_seconds) == (4.004, 6.006)
    assert first.time_label == "0:04–0:06"
    assert second.headline == "Logo missing at 0:09"
    assert second.end_seconds is None
    assert second.time_label == "0:09"


def test_logo_not_found_anywhere() -> None:
    logo = checks_for(report(logo_present=False))["logo"]
    assert logo.status is CheckStatus.FAILED
    assert logo.outcome == "The logo wasn't found anywhere in the video."
    assert logo.findings[0].headline == "Logo not found"


def test_late_hook() -> None:
    hook = checks_for(report(hook_time_seconds=5.2))["hook"]
    assert hook.status is CheckStatus.FAILED
    (finding,) = hook.findings
    assert finding.headline == "Opening line starts at 0:05"
    assert finding.detail == (
        "The opening line doesn't start until 0:05, but it needs to start by 0:03 "
        "so viewers don't scroll away."
    )
    assert (finding.start_seconds, finding.end_seconds) == (3.0, 5.2)
    assert [(m.kind, m.label) for m in hook.markers] == [
        ("deadline", "Must start by 0:03"),
        ("actual", "Starts 0:05"),
    ]


def test_hook_just_after_the_deadline_does_not_read_as_a_pass() -> None:
    hook = checks_for(report(hook_time_seconds=3.4))["hook"]
    assert hook.status is CheckStatus.FAILED
    assert hook.findings[0].headline == "Opening line starts just after 0:03"
    assert hook.outcome == "The opening line starts just after the 0:03 deadline."
    assert hook.findings[0].detail.startswith("The opening line doesn't start until just after")


def test_missing_hook_is_a_sentence() -> None:
    hook = checks_for(report(hook_time_seconds=None))["hook"]
    assert hook.status is CheckStatus.FAILED
    assert hook.outcome == "No spoken opening line was found."
    assert hook.findings[0].headline == "No spoken opening line"
    assert [m.kind for m in hook.markers] == ["deadline"]


def test_footage_mismatch_keeps_said_and_shown() -> None:
    mismatch = AudioVisualMismatch(timestamp_seconds=15.015, spoken="kitchen", shown="bathroom")
    footage = checks_for(report(audio_visual_mismatch=[mismatch]))["footage"]
    assert footage.status is CheckStatus.FAILED
    (finding,) = footage.findings
    assert finding.headline == "Footage doesn't match at 0:15"
    assert (finding.said, finding.shown) == ("kitchen", "bathroom")
    assert "0:15" in finding.how_to_fix


def test_silence() -> None:
    silence = checks_for(report(dead_air=[segment(21.3, 23.4)]))["silence"]
    assert silence.status is CheckStatus.FAILED
    assert silence.outcome == "Found 1 silent gap longer than 1 second."
    (finding,) = silence.findings
    assert finding.headline == "Silent from 0:21 to 0:23"
    assert finding.detail == "Nobody speaks for 2.1 seconds, from 0:21 to 0:23."


def test_thresholds_come_from_settings() -> None:
    late_for_default = report(hook_time_seconds=4.2)
    assert checks_for(late_for_default)["hook"].status is CheckStatus.FAILED
    relaxed = checks_for(late_for_default, hook_deadline_seconds=5, dead_air_threshold_seconds=2.5)
    assert relaxed["hook"].status is CheckStatus.PASSED
    assert "start by 0:05" in relaxed["hook"].explanation
    assert relaxed["silence"].explanation.endswith(
        "Gaps longer than 2.5 seconds make viewers lose interest."
    )
    definitions = {d.key: d for d in check_definitions(make_settings(agency_name="Northside"))}
    assert definitions["logo"].explanation.startswith("Northside's logo")


def test_findings_point_at_the_nearest_frame() -> None:
    checks = build_checks(
        report(logo_missing_timestamps=[2.002], dead_air=[segment(3.4, 5.0)]),
        make_settings(),
        frame_timestamps=[0.0, 1.001, 2.002, 3.003, 4.004],
    )
    by_key = {c.key: c for c in checks}
    assert by_key["logo"].findings[0].frame_index == 2
    assert by_key["silence"].findings[0].frame_index == 3


def test_verdict_counts_checks_and_things() -> None:
    failing = report(
        passed_qa=False,
        logo_missing_timestamps=[1.0, 2.0, 7.0],
        hook_time_seconds=None,
        dead_air=[segment(0.0, 2.0), segment(5.0, 7.0)],
    )
    verdict = summarize(build_checks(failing, make_settings()))
    assert verdict.headline == "Needs fixes: 3 of 4 checks failed"
    assert verdict.issue_count == 5  # 2 logo ranges + 1 hook + 2 silences
    assert verdict.short_label == "Needs fixes · 5 things"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (4.004, "0:04"), (59.99, "0:59"), (60, "1:00"), (75.9, "1:15"), (600, "10:00")],
)
def test_times_are_m_ss(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_duration_words() -> None:
    assert duration_words(1) == "1 second"
    assert duration_words(2.14) == "2.1 seconds"
    assert duration_words(3) == "3 seconds"


def test_editor_summary_is_the_slack_text_without_the_tag() -> None:
    failing = report(
        passed_qa=False,
        logo_missing_timestamps=[4.004],
        hook_time_seconds=5.2,
        audio_visual_mismatch=[
            AudioVisualMismatch(timestamp_seconds=15.0, spoken="kitchen & bath", shown="<garden>")
        ],
        dead_air=[segment(21.3, 23.4)],
        feedback_for_editor="Fix the open.",
    )
    assert editor_summary(failing, "ad.mp4", make_settings()).splitlines() == [
        "QA failed for ad.mp4",
        "❌ Logo missing at 0:04",
        "❌ Hook lands at 0:05 (must start within 0:03)",
        "❌ B-roll mismatch at 0:15 - speaker mentions kitchen & bath but shows <garden>",
        "❌ Dead air 0:21–0:23",
        "",
        "Editor feedback",
        "Fix the open.",
    ]
    assert editor_summary(report(), "ad.mp4", make_settings()).splitlines()[0] == (
        "QA passed for ad.mp4: all 4 checks passed."
    )


# --- Plain-language failure reasons -----------------------------------------------------------

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    return cls("boom", response=httpx2.Response(status, request=REQUEST), body=None)


@pytest.mark.parametrize(
    ("error", "starts_with"),
    [
        (
            VideoTooLongError("x", duration_seconds=135.0, max_seconds=100.0),
            "This video is 2:15 long, and the longest video we can check is 1:40.",
        ),
        (NoVideoStreamError("x"), "This file has sound but no picture"),
        (QARefusalError(None, "req_1"), "The AI reviewer declined to review this video"),
        (QATruncatedError("max_tokens", 16000, "req_1"), "The AI reviewer's answer was cut off"),
        (anthropic.APIConnectionError(request=REQUEST), "The AI reviewer couldn't be reached"),
        (anthropic.APITimeoutError(request=REQUEST), "The AI reviewer couldn't be reached"),
        (
            status_error(anthropic.AuthenticationError, 401),
            "The AI reviewer isn't connected: its API key was turned down",
        ),
        (
            TypeError("Could not resolve authentication method. Expected one of api_key"),
            "The AI reviewer isn't connected: no API key is set up",
        ),
        (status_error(anthropic.RateLimitError, 429), "The AI reviewer is busy right now"),
        (status_error(anthropic.InternalServerError, 529), "The AI reviewer is busy right now"),
        (asyncio.CancelledError(), "The checker was stopped before this video was finished"),
        (RuntimeError("anything else"), "Something went wrong on our side"),
    ],
)
def test_every_failure_has_one_plain_sentence(error: BaseException, starts_with: str) -> None:
    message = describe_failure(error)
    assert message.startswith(starts_with)
    assert "Error" not in message  # no exception class names leak into the sentence


def test_upload_too_large_message() -> None:
    assert describe_upload_too_large(500 * 1024 * 1024, 812 * 1024 * 1024) == (
        "This file is 812 MB, and the largest file we can take is 500 MB. "
        "Please export a smaller file."
    )
    assert describe_upload_too_large(500 * 1024 * 1024).startswith(
        "This file is bigger than 500 MB"
    )
