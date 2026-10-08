"""Application settings, read from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, FilePath, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Messages API limits the pipeline plans around.
MAX_IMAGES_PER_REQUEST = 100
MAX_REQUEST_BYTES = 32 * 1024 * 1024


class Settings(BaseSettings):
    """Every field maps to the upper-cased env var of the same name, e.g. ``ANTHROPIC_MODEL``."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Claude ---------------------------------------------------------------------------------
    # Passed to the SDK when set (so it can live in .env). When unset, the SDK resolves its own
    # credentials: ANTHROPIC_AUTH_TOKEN, an `ant auth login` profile, and so on.
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    # Non-streaming requests above ~21k max_tokens are refused by the SDK (10-minute rule).
    anthropic_max_tokens: int = Field(default=16_000, ge=1_024, le=21_000)

    # --- Agency and QA checklist ----------------------------------------------------------------
    agency_name: str = "Our Agency"
    agency_logo_path: FilePath | None = None
    agency_logo_description: str = (
        "the agency name as a small wordmark watermark in one corner of the frame"
    )
    hook_deadline_seconds: float = Field(default=3.0, gt=0)
    dead_air_threshold_seconds: float = Field(default=1.0, gt=0)

    # --- Deconstruction -------------------------------------------------------------------------
    max_frames: int = Field(default=100, ge=1, le=MAX_IMAGES_PER_REQUEST)
    frame_interval_seconds: float = Field(default=1.0, gt=0)
    frame_long_edge_px: int = Field(default=1024, ge=64, le=2000)
    frame_jpeg_qscale: int = Field(default=4, ge=2, le=31)
    audio_sample_rate: int = Field(default=16_000, gt=0)
    chunk_pause_seconds: float = Field(default=0.6, gt=0)
    chunk_max_seconds: float = Field(default=6.0, gt=0)
    ffmpeg_cmd: str = "ffmpeg"
    ffprobe_cmd: str = "ffprobe"

    # --- Notifications --------------------------------------------------------------------------
    # "none": results only appear in the dashboard. "slack": failures also ping the editor.
    notifier: Literal["none", "slack"] = "none"
    slack_webhook_url: SecretStr | None = None
    slack_timeout_seconds: float = Field(default=10.0, gt=0)

    # --- API / jobs -----------------------------------------------------------------------------
    max_upload_bytes: int = Field(default=500 * 1024 * 1024, gt=0)
    max_concurrent_jobs: int = Field(default=2, ge=1)
    log_level: str = "INFO"

    # --- Dashboard ------------------------------------------------------------------------------
    # Each job's video and frames live in MEDIA_DIR/<job_id>/. Job folders found here at
    # startup are deleted, because the job store is in memory and forgets them on restart.
    media_dir: Path = Path("data/jobs")
    max_stored_jobs: int = Field(default=20, ge=1)
    # Add a hand-written example report at startup, so every screen works without an API key.
    seed_example: bool = False

    @field_validator("anthropic_api_key", "agency_logo_path", "slack_webhook_url", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Treat ``FOO=`` in a .env file the same as leaving FOO out."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def frame_budget(self) -> int:
        """Most frames a video may produce once the logo reference image is counted."""
        reference_images = 1 if self.agency_logo_path is not None else 0
        return min(self.max_frames, MAX_IMAGES_PER_REQUEST - reference_images)

    @property
    def max_video_seconds(self) -> float:
        """Longest video accepted: one frame per interval, up to the frame budget."""
        return self.frame_budget * self.frame_interval_seconds
