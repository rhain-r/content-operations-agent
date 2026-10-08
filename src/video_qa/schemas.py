"""Data models shared across the pipeline, including the structured output Claude returns."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


def format_timestamp(seconds: float, *, precise: bool = False) -> str:
    """``m:ss`` like a video player shows it (truncated, not rounded); ``m:ss.s`` if precise."""
    seconds = max(seconds, 0.0)
    minutes, remainder = divmod(seconds, 60)
    if precise:
        tenths = int(remainder * 10)
        return f"{int(minutes)}:{tenths // 10:02d}.{tenths % 10}"
    return f"{int(minutes)}:{int(remainder):02d}"


# --- Transcript -------------------------------------------------------------------------------


class WordTimestamp(BaseModel):
    model_config = ConfigDict(frozen=True)

    word: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def _end_after_start(self) -> WordTimestamp:
        if self.end < self.start:
            raise ValueError(
                f"word {self.word!r} ends ({self.end}) before it starts ({self.start})"
            )
        return self


class TranscriptChunk(BaseModel):
    start_seconds: float
    end_seconds: float
    text: str


# --- Deconstructed video ----------------------------------------------------------------------


class VideoMetadata(BaseModel):
    filename: str
    container: str
    duration_seconds: float
    width: int
    height: int
    fps: float | None
    video_codec: str | None
    has_audio: bool
    audio_codec: str | None
    size_bytes: int
    expected_frame_count: int
    video_stream_index: int
    audio_stream_index: int | None


class Frame(BaseModel):
    index: int
    timestamp_seconds: float
    path: Path
    words: list[WordTimestamp] = Field(
        default_factory=list,
        description="Words that start between this frame and the next one.",
    )

    @property
    def spoken_text(self) -> str:
        return " ".join(word.word for word in self.words)


class DeadAirSegment(BaseModel):
    start_seconds: float = Field(ge=0, description="Start of the silence, in seconds.")
    end_seconds: float = Field(ge=0, description="End of the silence, in seconds.")


class DeconstructedVideo(BaseModel):
    metadata: VideoMetadata
    frames: list[Frame]
    audio_path: Path | None
    words: list[WordTimestamp]
    transcript: list[TranscriptChunk]
    dead_air: list[DeadAirSegment]


# --- Structured output from the QA agent ------------------------------------------------------


class AudioVisualMismatch(BaseModel):
    timestamp_seconds: float = Field(
        ge=0, description="Exact timestamp of the frame, the t= value from its label."
    )
    spoken: str = Field(description="What the speaker says at that moment, short quote or gist.")
    shown: str = Field(description="What the b-roll actually shows, in a few words.")


class VideoQAReport(BaseModel):
    passed_qa: bool = Field(description="True only if all four checks pass.")
    logo_present: bool = Field(description="Whether the agency logo is visible in the video.")
    logo_missing_timestamps: list[float] = Field(
        description="Exact timestamp (t= value) of every frame where the logo is not visible."
    )
    hook_time_seconds: float | None = Field(
        description="Start time of the spoken hook's first word, or null if nothing is spoken."
    )
    audio_visual_mismatch: list[AudioVisualMismatch] = Field(
        description="B-roll that clearly contradicts what is being said at that moment."
    )
    dead_air: list[DeadAirSegment] = Field(
        description="The pre-computed dead-air segments, copied unchanged."
    )
    feedback_for_editor: str = Field(
        description="Specific, actionable notes for the editor, citing m:ss timestamps."
    )
