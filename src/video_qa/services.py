"""What request handlers need from the running app, as a FastAPI dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from video_qa.config import Settings
from video_qa.jobs import JobStore
from video_qa.pipeline import JobRunner


@dataclass(frozen=True)
class AppServices:
    settings: Settings
    store: JobStore
    runner: JobRunner
    reviewer_connected: bool


def get_services(request: Request) -> AppServices:
    services: AppServices = request.app.state.services
    return services


Services = Annotated[AppServices, Depends(get_services)]
