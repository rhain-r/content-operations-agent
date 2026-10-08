"""The agent runs through the real Anthropic SDK; only the HTTP transport is faked."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic

from video_qa.agent import (
    ClaudeQAAgent,
    QAAgentError,
    QARefusalError,
    QATruncatedError,
    enforce_verdict,
)
from video_qa.schemas import AudioVisualMismatch, DeconstructedVideo, VideoQAReport

from helpers import fake_video, make_settings, segment, words

FRAME_TIMES = [0.0, 1.001, 2.002, 3.003, 4.004]
SPOKEN = words(
    ("Stop", 0.4, 0.7),
    ("scrolling", 0.8, 1.1),
    ("this", 1.2, 1.4),
    ("kitchen", 1.5, 1.9),
    ("changed", 3.2, 3.6),
    ("everything", 3.7, 4.2),
)
DURATION = 5.5


def model_report(**changes: Any) -> VideoQAReport:
    """What the model might send back: it claims a pass and forgets to copy the dead air."""
    report = VideoQAReport(
        passed_qa=True,
        logo_present=True,
        logo_missing_timestamps=[2.0],
        hook_time_seconds=0.41,
        audio_visual_mismatch=[],
        dead_air=[],
        feedback_for_editor="Tight edit. Logo dips out around 0:02.",
    )
    return report.model_copy(update=changes)


def message_body(
    text: str, *, stop_reason: str = "end_turn", stop_details: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "id": "msg_01Test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "c2ln"},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": {"input_tokens": 5000, "output_tokens": 300},
    }


class FakeMessagesAPI:
    """httpx2 MockTransport handler that records requests and replies with a canned message."""

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(200, json=self.body, headers={"request-id": "req_test123"})

    @property
    def sent(self) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(self.requests[-1].content)
        return payload


def client_for(api: FakeMessagesAPI) -> AsyncAnthropic:
    transport = httpx2.MockTransport(api)
    return AsyncAnthropic(
        api_key="test-key", max_retries=0, http_client=httpx2.AsyncClient(transport=transport)
    )


@pytest.fixture
def video(tmp_path: Path) -> DeconstructedVideo:
    return fake_video(tmp_path, timestamps=FRAME_TIMES, spoken=SPOKEN, duration=DURATION)


async def test_review_sends_interleaved_frames_and_enforces_verdict(
    video: DeconstructedVideo, caplog: pytest.LogCaptureFixture
) -> None:
    api = FakeMessagesAPI(message_body(model_report().model_dump_json()))
    settings = make_settings(agency_name="Northside Studio")

    async with client_for(api) as client:
        with caplog.at_level(logging.WARNING, logger="video_qa"):
            outcome = await ClaudeQAAgent(client, settings).review(video)
    report = outcome.report

    # --- the outgoing request
    assert len(api.requests) == 1
    request = api.requests[0]
    assert (request.method, request.url.path) == ("POST", "/v1/messages")
    body = api.sent
    assert body["model"] == "claude-sonnet-5-5"
    assert body["max_tokens"] == 16_000
    assert "temperature" not in body
    output_format = body["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert set(output_format["schema"]["properties"]) == set(VideoQAReport.model_fields)
    assert output_format["schema"]["additionalProperties"] is False

    system = body["system"]
    assert "elite short-form video editor" in system
    assert "Northside Studio" in system
    assert settings.agency_logo_description in system  # no reference image configured
    assert "within the first 3 s" in system
    assert "longer than 1 s" in system

    content = body["messages"][0]["content"]
    image_positions = [i for i, block in enumerate(content) if block["type"] == "image"]
    assert len(image_positions) == len(FRAME_TIMES)
    for frame, position in zip(video.frames, image_positions, strict=True):
        label = content[position - 1]
        assert label["type"] == "text"
        assert label["text"].startswith(f"[Frame {frame.index} | ")
        source = content[position]["source"]
        assert source["type"] == "base64"
        assert source["media_type"] == "image/jpeg"
        assert base64.b64decode(source["data"]) == frame.path.read_bytes()
    labels = [block["text"] for block in content if block["type"] == "text"]
    assert '[Frame 0 | 0:00 | t=0.000s] Spoken: "Stop scrolling"' in labels
    assert "[Frame 2 | 0:02 | t=2.002s] Spoken: (nothing)" in labels
    assert '[Frame 3 | 0:03 | t=3.003s] Spoken: "changed everything"' in labels

    context = content[-1]["text"]
    assert "[0.40-1.90] Stop scrolling this kitchen" in context
    assert json.dumps([s.model_dump() for s in video.dead_air]) in context
    assert '"filename":"ad.mp4"' in context

    # --- the verdict the code enforces
    assert video.dead_air == [segment(1.9, 3.2), segment(4.2, 5.5)]
    assert report.dead_air == video.dead_air  # overwritten with the computed segments
    assert report.logo_missing_timestamps == [2.002]  # snapped to the real frame
    assert report.logo_present is True
    assert report.hook_time_seconds == 0.4  # snapped to the first word's start
    assert report.passed_qa is False  # model said True; logo gap and dead air say otherwise
    assert report.feedback_for_editor == "Tight edit. Logo dips out around 0:02."
    assert "Overriding model verdict" in caplog.text
    # The model's own answer is kept alongside, untouched, for the dashboard's tech details.
    assert outcome.model_report == model_report()
    assert outcome.model_report.passed_qa is True


async def test_logo_reference_image_comes_first(video: DeconstructedVideo, logo_png: Path) -> None:
    api = FakeMessagesAPI(message_body(model_report().model_dump_json()))
    settings = make_settings(agency_logo_path=logo_png)

    async with client_for(api) as client:
        await ClaudeQAAgent(client, settings).review(video)

    content = api.sent["messages"][0]["content"]
    assert [block["type"] for block in content[:3]] == ["text", "image", "text"]
    assert content[0]["text"] == "Reference image: Our Agency's logo."
    assert content[1]["source"]["media_type"] == "image/png"
    assert base64.b64decode(content[1]["source"]["data"]).startswith(b"\x89PNG")
    assert content[2]["text"].startswith("Frames, one every 1 s.")
    assert sum(block["type"] == "image" for block in content) == len(FRAME_TIMES) + 1
    assert "The first image is a reference image" in api.sent["system"]


async def test_refusal_raises_typed_error(video: DeconstructedVideo) -> None:
    api = FakeMessagesAPI(
        message_body(
            '{"passed_qa": tr',
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": "cyber", "explanation": "Declined."},
        )
    )
    async with client_for(api) as client:
        with pytest.raises(QARefusalError) as caught:
            await ClaudeQAAgent(client, make_settings()).review(video)
    assert caught.value.category == "cyber"
    assert caught.value.explanation == "Declined."
    assert caught.value.request_id == "req_test123"


async def test_max_tokens_raises_typed_error(video: DeconstructedVideo) -> None:
    truncated = model_report().model_dump_json()[:40]
    api = FakeMessagesAPI(message_body(truncated, stop_reason="max_tokens"))
    async with client_for(api) as client:
        with pytest.raises(QATruncatedError, match="stop_reason=max_tokens"):
            await ClaudeQAAgent(client, make_settings()).review(video)


async def test_schema_mismatch_raises_agent_error(video: DeconstructedVideo) -> None:
    api = FakeMessagesAPI(message_body('{"passed_qa": "maybe"}'))
    async with client_for(api) as client:
        with pytest.raises(QAAgentError, match="did not match the report schema"):
            await ClaudeQAAgent(client, make_settings()).review(video)


# --- enforce_verdict --------------------------------------------------------------------------

CONTINUOUS_SPEECH = words(*((f"w{i}", 0.3 + i * 0.5, 0.7 + i * 0.5) for i in range(10)))


def test_clean_video_passes_even_if_model_said_fail(tmp_path: Path) -> None:
    video = fake_video(tmp_path, timestamps=FRAME_TIMES, spoken=CONTINUOUS_SPEECH, duration=5.0)
    assert video.dead_air == []
    report = enforce_verdict(
        model_report(passed_qa=False, logo_missing_timestamps=[], hook_time_seconds=0.3),
        video,
        make_settings(),
    )
    assert report.passed_qa is True


def test_logo_absent_with_no_frames_listed_means_missing_everywhere(tmp_path: Path) -> None:
    video = fake_video(tmp_path, timestamps=FRAME_TIMES, spoken=CONTINUOUS_SPEECH, duration=5.0)
    report = enforce_verdict(
        model_report(logo_present=False, logo_missing_timestamps=[]), video, make_settings()
    )
    assert report.logo_missing_timestamps == FRAME_TIMES
    assert report.logo_present is False
    assert report.passed_qa is False


def test_logo_listed_missing_in_every_frame_is_not_present(tmp_path: Path) -> None:
    video = fake_video(tmp_path, timestamps=FRAME_TIMES, spoken=CONTINUOUS_SPEECH, duration=5.0)
    report = enforce_verdict(
        model_report(logo_missing_timestamps=[0, 1, 2, 3, 4, 4.1]), video, make_settings()
    )
    assert report.logo_missing_timestamps == FRAME_TIMES
    assert report.logo_present is False


def test_late_hook_fails(tmp_path: Path) -> None:
    late = words(("Finally", 3.4, 3.8), ("here", 3.9, 4.2))
    video = fake_video(tmp_path, timestamps=FRAME_TIMES, spoken=late, duration=4.5)
    report = enforce_verdict(
        model_report(logo_missing_timestamps=[], hook_time_seconds=3.5), video, make_settings()
    )
    assert report.hook_time_seconds == 3.4
    assert report.passed_qa is False


def test_hook_is_none_without_speech(tmp_path: Path) -> None:
    video = fake_video(tmp_path, timestamps=FRAME_TIMES, spoken=[], duration=5.0)
    report = enforce_verdict(
        model_report(logo_missing_timestamps=[], hook_time_seconds=1.0), video, make_settings()
    )
    assert report.hook_time_seconds is None
    assert report.dead_air == [segment(0.0, 5.0)]
    assert report.passed_qa is False


def test_mismatch_fails_and_is_snapped_to_its_frame(tmp_path: Path) -> None:
    video = fake_video(tmp_path, timestamps=FRAME_TIMES, spoken=CONTINUOUS_SPEECH, duration=5.0)
    mismatch = AudioVisualMismatch(timestamp_seconds=3.0, spoken="kitchen", shown="bathroom")
    report = enforce_verdict(
        model_report(
            logo_missing_timestamps=[], hook_time_seconds=0.3, audio_visual_mismatch=[mismatch]
        ),
        video,
        make_settings(),
    )
    assert report.audio_visual_mismatch[0].timestamp_seconds == 3.003
    assert report.passed_qa is False
