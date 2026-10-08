"""Break a video into timestamped frames, a 16 kHz mono WAV, and a word-level transcript.

ffmpeg-python is blocking (it shells out and waits), so every ffmpeg/ffprobe call runs through
``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from bisect import bisect_right
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import ffmpeg

from video_qa.config import Settings
from video_qa.schemas import DeconstructedVideo, Frame, VideoMetadata, WordTimestamp
from video_qa.transcription import Transcriber, compute_dead_air, group_words_into_chunks

logger = logging.getLogger(__name__)

FRAME_FILENAME_PATTERN = "frame_%05d.jpg"
_FRAME_GLOB = "frame_*.jpg"

# One showinfo line per frame that passed the select filter, e.g.
# [Parsed_showinfo_1 @ 000001e9] n:   4 pts: 120120 pts_time:4.004   duration: 1001 ...
_SHOWINFO_LINE = re.compile(r"Parsed_showinfo_\d+\b.*?\bn:\s*(\d+)\s.*?\bpts_time:\s*(\S+)")


class VideoRejectedError(ValueError):
    """The upload can't be QA'd: unreadable, no video stream, or too long."""


class UnreadableVideoError(VideoRejectedError):
    pass


class NoVideoStreamError(VideoRejectedError):
    pass


class StillImageError(VideoRejectedError):
    pass


class VideoTooLongError(VideoRejectedError):
    def __init__(self, message: str, *, duration_seconds: float, max_seconds: float) -> None:
        super().__init__(message)
        self.duration_seconds = duration_seconds
        self.max_seconds = max_seconds


# Progress points reported to ``deconstruct_video``'s ``on_stage`` callback, in order.
DeconstructionStage = Literal["reading", "splitting", "listening"]
StageCallback = Callable[[DeconstructionStage], Awaitable[None]]


class FFmpegError(RuntimeError):
    """ffmpeg failed on a file that ffprobe had accepted."""


# --- Probing ----------------------------------------------------------------------------------


async def probe_video(path: Path, settings: Settings) -> VideoMetadata:
    """Probe ``path`` and reject it before any extraction work if it can't be QA'd."""
    try:
        probe = await asyncio.to_thread(_probe, path, settings.ffprobe_cmd)
    except ffmpeg.Error as exc:
        raise UnreadableVideoError(f"Not a readable video file: {_stderr_tail(exc)}") from exc
    size_bytes = await asyncio.to_thread(lambda: path.stat().st_size)
    return metadata_from_probe(probe, filename=path.name, size_bytes=size_bytes, settings=settings)


def _probe(path: Path, ffprobe_cmd: str) -> dict[str, Any]:
    result: dict[str, Any] = ffmpeg.probe(str(path), cmd=ffprobe_cmd)
    return result


def metadata_from_probe(
    probe: dict[str, Any], *, filename: str, size_bytes: int, settings: Settings
) -> VideoMetadata:
    fmt: dict[str, Any] = probe.get("format") or {}
    streams: list[dict[str, Any]] = probe.get("streams") or []
    container = str(fmt.get("format_name", ""))

    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise NoVideoStreamError("The file has no video stream.")
    if container == "image2" or container.endswith("_pipe"):
        raise StillImageError("The file is a still image, not a video.")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    video_duration = _positive_float(video.get("duration"))
    duration = _positive_float(fmt.get("duration")) or video_duration
    if duration is None:
        raise UnreadableVideoError("Could not determine the video's duration.")

    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    interval = settings.frame_interval_seconds
    expected = expected_frame_count(video_duration or duration, fps, interval)
    if expected > settings.frame_budget:
        raise VideoTooLongError(
            f"The video is {duration:.1f} s long and would produce {expected} frames at one every "
            f"{interval:g} s; the limit is {settings.frame_budget} frames.",
            duration_seconds=duration,
            max_seconds=settings.max_video_seconds,
        )

    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if _rotation_degrees(video) % 180 == 90:
        width, height = height, width

    return VideoMetadata(
        filename=filename,
        container=container,
        duration_seconds=round(duration, 3),
        width=width,
        height=height,
        fps=round(fps, 3) if fps else None,
        video_codec=video.get("codec_name"),
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        size_bytes=size_bytes,
        expected_frame_count=expected,
        video_stream_index=int(video["index"]),
        audio_stream_index=int(audio["index"]) if audio else None,
    )


def expected_frame_count(duration: float, fps: float | None, interval: float) -> int:
    """How many frames ``select='gte(t,selected_n*interval)'`` will pass.

    A frame is selected for each k with k * interval <= the last frame's timestamp, and the
    last frame starts one frame-duration before the stream ends.
    """
    last_frame = max(duration - (1.0 / fps if fps else 0.0), 0.0)
    return math.floor(last_frame / interval + 1e-9) + 1


def _positive_float(value: object) -> float | None:
    try:
        number = float(str(value))
    except ValueError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def _parse_rate(value: object) -> float | None:
    numerator, _, denominator = str(value).partition("/")
    try:
        rate = float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def _rotation_degrees(stream: dict[str, Any]) -> int:
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return int(float(side_data["rotation"]))
    rotate = (stream.get("tags") or {}).get("rotate")
    return int(float(rotate)) if rotate is not None else 0


# --- Extraction -------------------------------------------------------------------------------


def _scale_args(long_edge: int) -> dict[str, str]:
    """Scale so the long edge is at most ``long_edge`` px, keeping aspect ratio, never upscaling.

    Evaluated by ffmpeg against the decoded (auto-rotated) frame, so phone footage with rotation
    metadata scales correctly.
    """
    return {
        "w": f"if(gte(iw,ih),min(iw,{long_edge}),-2)",
        "h": f"if(gte(iw,ih),-2,min(ih,{long_edge}))",
    }


def _run_ffmpeg(
    stream: Any, settings: Settings, *, capture_stdout: bool = False
) -> tuple[bytes, str]:
    command = stream.global_args("-hide_banner", "-nostdin", "-nostats").overwrite_output()
    try:
        stdout, stderr = command.run(
            cmd=settings.ffmpeg_cmd, capture_stdout=capture_stdout, capture_stderr=True
        )
    except ffmpeg.Error as exc:
        raise FFmpegError(f"ffmpeg failed: {_stderr_tail(exc)}") from exc
    return bytes(stdout or b""), bytes(stderr or b"").decode("utf-8", errors="replace")


def _stderr_tail(exc: Any, lines: int = 3) -> str:
    text = bytes(getattr(exc, "stderr", None) or b"").decode("utf-8", errors="replace").strip()
    return " | ".join(text.splitlines()[-lines:]) or "no diagnostic output"


async def extract_audio(
    path: Path, out_path: Path, metadata: VideoMetadata, settings: Settings
) -> Path:
    """Write the first audio stream to ``out_path`` as 16 kHz mono 16-bit PCM WAV."""
    if metadata.audio_stream_index is None:
        raise ValueError("the video has no audio stream")
    stream = ffmpeg.input(str(path))[str(metadata.audio_stream_index)].output(
        str(out_path), ac=1, ar=settings.audio_sample_rate, acodec="pcm_s16le", format="wav"
    )
    await asyncio.to_thread(_run_ffmpeg, stream, settings)
    return out_path


async def extract_frames(
    path: Path, frames_dir: Path, metadata: VideoMetadata, settings: Settings
) -> list[tuple[Path, float]]:
    """Extract one JPEG per interval and return each file with its real timestamp in seconds.

    ``fps=1`` would retime frames onto a 1 fps grid, labelling them up to half a second away
    from where they really are. Instead ``select`` keeps the first decoded frame at or after
    each multiple of the interval, ``showinfo`` logs that frame's actual ``pts_time``, and
    ``-fps_mode passthrough`` writes each selected frame once, untouched.
    """
    await asyncio.to_thread(frames_dir.mkdir, parents=True, exist_ok=True)
    interval = settings.frame_interval_seconds
    stream = (
        ffmpeg.input(str(path))[str(metadata.video_stream_index)]
        .filter("select", f"gte(t,selected_n*{interval:g})")
        .filter("showinfo")
        .filter("scale", **_scale_args(settings.frame_long_edge_px))
        .output(
            str(frames_dir / FRAME_FILENAME_PATTERN),
            fps_mode="passthrough",
            **{"q:v": settings.frame_jpeg_qscale},
        )
    )
    _, stderr = await asyncio.to_thread(_run_ffmpeg, stream, settings)
    files = await asyncio.to_thread(lambda: sorted(frames_dir.glob(_FRAME_GLOB)))
    timestamps = resolve_frame_timestamps(parse_showinfo_timestamps(stderr), len(files), interval)
    return list(zip(files, timestamps, strict=True))


def parse_showinfo_timestamps(stderr: str) -> list[float]:
    """``pts_time`` of every frame showinfo logged, ordered by its frame number."""
    by_frame: dict[int, float] = {}
    for match in _SHOWINFO_LINE.finditer(stderr):
        try:
            by_frame[int(match.group(1))] = float(match.group(2))
        except ValueError:
            continue
    return [by_frame[n] for n in sorted(by_frame)]


def resolve_frame_timestamps(
    parsed: Sequence[float], frame_count: int, interval: float
) -> list[float]:
    """Use the parsed timestamps when they line up one-to-one with the written files;
    otherwise fall back to index x interval."""
    if len(parsed) == frame_count:
        return [round(t, 3) for t in parsed]
    logger.warning(
        "showinfo reported %d frames but %d files were written; falling back to index x %gs",
        len(parsed),
        frame_count,
        interval,
    )
    return [round(i * interval, 3) for i in range(frame_count)]


def map_words_to_frames(
    timestamps: Sequence[float], words: Sequence[WordTimestamp]
) -> list[list[WordTimestamp]]:
    """Give each frame the words that start between it and the next frame.

    Every word lands on exactly one frame; words that start before the first frame go to it.
    """
    buckets: list[list[WordTimestamp]] = [[] for _ in timestamps]
    if not buckets:
        return buckets
    for word in sorted(words, key=lambda w: w.start):
        index = bisect_right(timestamps, word.start) - 1
        buckets[max(index, 0)].append(word)
    return buckets


async def render_reference_image(path: Path, settings: Settings) -> bytes:
    """Re-encode an image (e.g. the agency logo) as PNG within the same size limit as frames."""
    stream = (
        ffmpeg.input(str(path))
        .filter("scale", **_scale_args(settings.frame_long_edge_px))
        .output("pipe:", format="image2pipe", vcodec="png", **{"frames:v": 1})
    )
    png, _ = await asyncio.to_thread(_run_ffmpeg, stream, settings, capture_stdout=True)
    if not png:
        raise FFmpegError(f"ffmpeg produced no image for {path.name}")
    return png


# --- Orchestration ----------------------------------------------------------------------------


async def deconstruct_video(
    path: Path,
    workdir: Path,
    transcriber: Transcriber,
    settings: Settings,
    *,
    on_stage: StageCallback | None = None,
) -> DeconstructedVideo:
    """Probe, then extract frames and audio concurrently, transcribe, and lay every frame on
    the audio timeline.

    ``on_stage`` hears "reading" before the probe, "splitting" while frames are extracted
    (audio extraction and transcription run alongside), and "listening" once the frames are
    done and only the transcript is still outstanding.
    """

    async def report(stage: DeconstructionStage) -> None:
        if on_stage is not None:
            await on_stage(stage)

    await report("reading")
    metadata = await probe_video(path, settings)
    frames_dir = workdir / "frames"
    wav_path = workdir / "audio.wav"

    async def audio_and_words() -> tuple[Path | None, list[WordTimestamp]]:
        if not metadata.has_audio:
            return None, []
        await extract_audio(path, wav_path, metadata, settings)
        words = await transcriber.transcribe(wav_path)
        return wav_path, sorted(words, key=lambda w: w.start)

    await report("splitting")
    audio_task = asyncio.create_task(audio_and_words())
    try:
        extracted = await extract_frames(path, frames_dir, metadata, settings)
        await report("listening")
        audio_path, words = await audio_task
    except BaseException:
        audio_task.cancel()
        await asyncio.gather(audio_task, return_exceptions=True)
        raise
    if not extracted:
        raise VideoRejectedError("ffmpeg could not decode any frames from the video.")
    if len(extracted) > settings.frame_budget:
        raise VideoRejectedError(
            f"The video produced {len(extracted)} frames; the limit is {settings.frame_budget}."
        )

    timestamps = [timestamp for _, timestamp in extracted]
    frames = [
        Frame(index=i, timestamp_seconds=timestamp, path=frame_path, words=frame_words)
        for i, ((frame_path, timestamp), frame_words) in enumerate(
            zip(extracted, map_words_to_frames(timestamps, words), strict=True)
        )
    ]
    logger.info(
        "Deconstructed %s: %.1fs, %d frames, %d words",
        metadata.filename,
        metadata.duration_seconds,
        len(frames),
        len(words),
    )
    return DeconstructedVideo(
        metadata=metadata,
        frames=frames,
        audio_path=audio_path,
        words=words,
        transcript=group_words_into_chunks(
            words,
            pause_seconds=settings.chunk_pause_seconds,
            max_chunk_seconds=settings.chunk_max_seconds,
        ),
        dead_air=compute_dead_air(
            words, metadata.duration_seconds, threshold_seconds=settings.dead_air_threshold_seconds
        ),
    )
