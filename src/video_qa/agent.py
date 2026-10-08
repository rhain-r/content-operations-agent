"""Multimodal QA agent: shows Claude the deconstructed video and enforces the verdict in code."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Sequence
from typing import Literal, Protocol

import pydantic
from anthropic import AsyncAnthropic, transform_schema
from anthropic.types import (
    ImageBlockParam,
    JSONOutputFormatParam,
    RefusalStopDetails,
    TextBlockParam,
)

from video_qa.config import MAX_IMAGES_PER_REQUEST, MAX_REQUEST_BYTES, Settings
from video_qa.deconstruction import render_reference_image
from video_qa.schemas import (
    DeconstructedVideo,
    Frame,
    VideoQAReport,
    WordTimestamp,
    format_timestamp,
)

logger = logging.getLogger(__name__)

ContentBlock = TextBlockParam | ImageBlockParam

# Native structured output. transform_schema is the SDK's own Pydantic-to-API conversion (the
# one messages.parse uses): it drops keywords the API doesn't support, such as minimum, and
# model_validate_json enforces them client-side afterwards.
REPORT_OUTPUT_FORMAT: JSONOutputFormatParam = {
    "type": "json_schema",
    "schema": transform_schema(VideoQAReport),
}

# Room kept free in the request for the system prompt, output schema and JSON framing.
_REQUEST_OVERHEAD_BYTES = 512 * 1024
# Stop reasons where the output may be cut off and so may not match the schema.
_TRUNCATING_STOP_REASONS = {"max_tokens", "model_context_window_exceeded"}


class QAAgentError(RuntimeError):
    """Claude did not return a usable QA report."""


class QARefusalError(QAAgentError):
    def __init__(self, details: RefusalStopDetails | None, request_id: str | None) -> None:
        self.category = details.category if details else None
        self.explanation = details.explanation if details else None
        self.request_id = request_id
        super().__init__(
            f"Claude declined to review the video (category={self.category}, "
            f"request_id={request_id}): {self.explanation or 'no explanation given'}"
        )


class QATruncatedError(QAAgentError):
    def __init__(self, stop_reason: str, max_tokens: int, request_id: str | None) -> None:
        self.stop_reason = stop_reason
        self.request_id = request_id
        super().__init__(
            f"Claude's report was cut off (stop_reason={stop_reason}, max_tokens={max_tokens}, "
            f"request_id={request_id}); raise ANTHROPIC_MAX_TOKENS or send fewer frames."
        )


class QARequestTooLargeError(QAAgentError):
    """The frames don't fit in one Messages API request."""


class ReviewOutcome(pydantic.BaseModel):
    report: VideoQAReport = pydantic.Field(description="The verdict as enforced by the code.")
    model_report: VideoQAReport = pydantic.Field(description="The report as the model sent it.")


class QAAgent(Protocol):
    async def review(self, video: DeconstructedVideo) -> ReviewOutcome: ...


class ClaudeQAAgent:
    def __init__(self, client: AsyncAnthropic, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._logo_png: bytes | None = None

    async def review(self, video: DeconstructedVideo) -> ReviewOutcome:
        logo_png = await self._reference_logo()
        system = build_system_prompt(self._settings, has_reference_logo=logo_png is not None)
        content = await asyncio.to_thread(build_user_content, video, self._settings, logo_png)

        # This is the request messages.parse(output_format=VideoQAReport) sends. parse() itself
        # isn't used because it validates the JSON while building the response object, so a
        # refused or truncated reply would raise a bare ValidationError and the stop_reason
        # would be lost. Checking stop_reason before validating keeps those cases typed.
        message = await self._client.messages.create(
            model=self._settings.anthropic_model,
            max_tokens=self._settings.anthropic_max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"format": REPORT_OUTPUT_FORMAT},
        )
        request_id = message._request_id
        if message.stop_reason == "refusal":
            raise QARefusalError(message.stop_details, request_id)
        if message.stop_reason in _TRUNCATING_STOP_REASONS:
            raise QATruncatedError(
                str(message.stop_reason), self._settings.anthropic_max_tokens, request_id
            )
        text = "".join(block.text for block in message.content if block.type == "text")
        try:
            report = VideoQAReport.model_validate_json(text)
        except pydantic.ValidationError as exc:
            raise QAAgentError(
                f"Claude's output did not match the report schema "
                f"(stop_reason={message.stop_reason}, request_id={request_id})"
            ) from exc
        logger.info(
            "QA review of %s: request_id=%s input_tokens=%d output_tokens=%d",
            video.metadata.filename,
            request_id,
            message.usage.input_tokens,
            message.usage.output_tokens,
        )
        return ReviewOutcome(
            report=enforce_verdict(report, video, self._settings), model_report=report
        )

    async def _reference_logo(self) -> bytes | None:
        path = self._settings.agency_logo_path
        if path is None:
            return None
        if self._logo_png is None:
            self._logo_png = await render_reference_image(path, self._settings)
        return self._logo_png


# --- Prompt -----------------------------------------------------------------------------------


def build_system_prompt(settings: Settings, *, has_reference_logo: bool) -> str:
    agency = settings.agency_name
    deadline = f"{settings.hook_deadline_seconds:g}"
    dead_air = f"{settings.dead_air_threshold_seconds:g}"
    if has_reference_logo:
        logo_source = (
            f"The first image is a reference image of {agency}'s logo. Look for that exact logo; "
            "other brands' logos and on-screen text don't count."
        )
    else:
        logo_source = (
            f"No reference image is available, so identify {agency}'s logo from this "
            f"description: {settings.agency_logo_description}. Other brands' logos don't count."
        )
    return f"""\
You are an elite short-form video editor doing the final QA pass on a finished video before \
{agency} delivers it to the client. You receive one frame every \
{settings.frame_interval_seconds:g} s, each labelled with its exact timestamp and the words \
spoken until the next frame, followed by the full timestamped transcript, a pre-computed \
dead-air list, and the video metadata.

Run four checks.

1. Logo. {logo_source} The logo is a persistent watermark, so it should be visible in every \
frame. Set logo_present to whether it appears in the video at all, and list in \
logo_missing_timestamps the exact timestamp (the t= value from the frame label) of every frame \
where it is absent, covered, or cropped out.

2. Hook. The spoken hook must start within the first {deadline} s. Set hook_time_seconds to \
the start time of the hook's first word, read from the transcript's word timings. If nothing \
is spoken, set it to null.

3. B-roll. Where a frame shows b-roll, compare what is shown with what is being said at that \
timestamp. A speaker talking on camera is not b-roll. Flag only clear contradictions, where \
the speaker names one thing and the footage plainly shows another, not footage that is merely \
loosely related or generic. Give the frame's timestamp, what is spoken, and what is shown.

4. Dead air. Silence longer than {dead_air} s is dead air. It has already been measured from \
the word timestamps; copy the provided list into dead_air unchanged.

If you are unsure whether a logo is missing or whether b-roll contradicts the audio, don't \
flag it; say in the feedback what you were unsure about and where. Write feedback_for_editor \
for the editor who will fix the cut: specific and actionable, citing m:ss timestamps and \
saying what to change. If everything passes, say so in a sentence. Set passed_qa to true only \
if all four checks pass."""


class _ContentBuilder:
    def __init__(self) -> None:
        self.blocks: list[ContentBlock] = []
        self.image_count = 0
        self.approx_bytes = 0

    def text(self, text: str) -> None:
        self.blocks.append({"type": "text", "text": text})
        self.approx_bytes += len(text.encode("utf-8"))

    def image(self, data: bytes, media_type: Literal["image/jpeg", "image/png"]) -> None:
        encoded = base64.standard_b64encode(data).decode("ascii")
        self.blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": encoded},
            }
        )
        self.image_count += 1
        self.approx_bytes += len(encoded)


def build_user_content(
    video: DeconstructedVideo, settings: Settings, reference_logo_png: bytes | None
) -> list[ContentBlock]:
    """Logo reference, then each frame's label followed by its image, then the timeline data.

    Reads every frame from disk, so call it off the event loop.
    """
    builder = _ContentBuilder()
    if reference_logo_png is not None:
        builder.text(f"Reference image: {settings.agency_name}'s logo.")
        builder.image(reference_logo_png, "image/png")
    builder.text(
        f"Frames, one every {settings.frame_interval_seconds:g} s. Each label gives the frame's "
        "exact timestamp and the words spoken from that frame until the next one."
    )
    for frame in video.frames:
        builder.text(frame_label(frame))
        builder.image(frame.path.read_bytes(), "image/jpeg")
    builder.text(timeline_context(video, settings))

    if builder.image_count > MAX_IMAGES_PER_REQUEST:
        raise QARequestTooLargeError(
            f"{builder.image_count} images exceed the {MAX_IMAGES_PER_REQUEST}-image limit"
        )
    if builder.approx_bytes > MAX_REQUEST_BYTES - _REQUEST_OVERHEAD_BYTES:
        raise QARequestTooLargeError(
            f"Frames total {builder.approx_bytes / 1e6:.1f} MB base64-encoded, over the "
            f"{MAX_REQUEST_BYTES // (1024 * 1024)} MB request limit; raise FRAME_JPEG_QSCALE "
            "or lower FRAME_LONG_EDGE_PX."
        )
    return builder.blocks


def frame_label(frame: Frame) -> str:
    spoken = f'"{frame.spoken_text}"' if frame.words else "(nothing)"
    t = frame.timestamp_seconds
    return f"[Frame {frame.index} | {format_timestamp(t)} | t={t:.3f}s] Spoken: {spoken}"


def timeline_context(video: DeconstructedVideo, settings: Settings) -> str:
    sections: list[str] = []
    if video.transcript:
        lines = [f"[{c.start_seconds:.2f}-{c.end_seconds:.2f}] {c.text}" for c in video.transcript]
        sections.append("Transcript, chunked on pauses (seconds):\n" + "\n".join(lines))
        hook_window = max(10.0, 3 * settings.hook_deadline_seconds)
        early = [w for w in video.words if w.start < hook_window]
        sections.append(
            f"Word start times for the first {hook_window:g} s, for pinpointing the hook:\n"
            + " | ".join(f"{w.word} {w.start:.2f}" for w in early)
        )
    elif video.metadata.has_audio:
        sections.append("Transcript: no speech was detected.")
    else:
        sections.append("Transcript: none, the video has no audio track.")

    dead_air = [segment.model_dump() for segment in video.dead_air]
    sections.append(
        f"Dead air longer than {settings.dead_air_threshold_seconds:g} s, pre-computed from the "
        f"word timestamps. Copy into dead_air unchanged:\n{json.dumps(dead_air)}"
    )
    sections.append(
        "Video metadata:\n" + video.metadata.model_dump_json(exclude={"expected_frame_count"})
    )
    sections.append("Run the four QA checks and return the report.")
    return "\n\n".join(sections)


# --- Verdict ----------------------------------------------------------------------------------


def enforce_verdict(
    report: VideoQAReport, video: DeconstructedVideo, settings: Settings
) -> VideoQAReport:
    """The code, not the model, owns the verdict.

    Timestamps are snapped to real frames and word starts, dead air is replaced with the
    computed segments, and passed_qa is recomputed from the four checks.
    """
    frame_times = [frame.timestamp_seconds for frame in video.frames]
    missing = sorted({_nearest(t, frame_times) for t in report.logo_missing_timestamps})
    if not report.logo_present and not missing:
        # "Not present" with no frames listed means it is missing throughout.
        missing = list(frame_times)
    logo_present = report.logo_present and len(missing) < len(frame_times)

    hook = _snap_to_word_start(report.hook_time_seconds, video.words)
    mismatches = [
        m.model_copy(update={"timestamp_seconds": _nearest(m.timestamp_seconds, frame_times)})
        for m in report.audio_visual_mismatch
    ]
    dead_air = list(video.dead_air)

    problems: list[str] = []
    if missing:
        problems.append(f"logo missing in {len(missing)} frame(s)")
    if hook is None:
        problems.append("no spoken hook")
    elif hook > settings.hook_deadline_seconds:
        problems.append(f"hook starts at {hook:.2f}s")
    if mismatches:
        problems.append(f"{len(mismatches)} b-roll mismatch(es)")
    if dead_air:
        problems.append(f"{len(dead_air)} dead-air segment(s)")
    passed = not problems

    if passed != report.passed_qa:
        logger.warning(
            "Overriding model verdict for %s: model said passed_qa=%s, checks say %s (%s)",
            video.metadata.filename,
            report.passed_qa,
            passed,
            "; ".join(problems) or "all checks pass",
        )
    return report.model_copy(
        update={
            "passed_qa": passed,
            "logo_present": logo_present,
            "logo_missing_timestamps": missing,
            "hook_time_seconds": hook,
            "audio_visual_mismatch": mismatches,
            "dead_air": dead_air,
        }
    )


def _nearest(value: float, candidates: Sequence[float]) -> float:
    return min(candidates, key=lambda candidate: abs(candidate - value))


def _snap_to_word_start(value: float | None, words: Sequence[WordTimestamp]) -> float | None:
    if value is None or not words:
        return None
    return _nearest(value, [word.start for word in words])
