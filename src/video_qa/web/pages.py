"""HTML pages for the dashboard: home, a video's progress/report, and the not-found page."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from video_qa.findings import REVIEWER_NOT_CONNECTED, check_definitions, megabytes
from video_qa.jobs import parse_job_id
from video_qa.schemas import format_timestamp
from video_qa.services import Services
from video_qa.web import WEB_DIR
from video_qa.web.views import LIVE_LINKS, build_video_page, video_rows

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=WEB_DIR / "templates")
templates.env.filters["mmss"] = format_timestamp
# Defaults for live pages; the static exporter overrides these per page.
templates.env.globals["static_mode"] = False
templates.env.globals["links"] = LIVE_LINKS
templates.env.globals["repo_url"] = None
# Appended to the CSS/JS URLs so browsers fetch them again after an update.
templates.env.globals["asset_version"] = str(
    int(max(path.stat().st_mtime for path in (WEB_DIR / "static").iterdir()))
)


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, services: Services) -> HTMLResponse:
    settings = services.settings
    jobs = await services.store.list()
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "rows": video_rows(jobs, settings),
            "has_active": any(not job.finished for job in jobs),
            "checks": check_definitions(settings),
            "reviewer_connected": services.reviewer_connected,
            "reviewer_not_connected": REVIEWER_NOT_CONNECTED,
            "max_length": format_timestamp(settings.max_video_seconds),
            "max_size": megabytes(settings.max_upload_bytes),
            "max_upload_bytes": settings.max_upload_bytes,
            "slack_required": settings.notifier == "slack",
        },
    )


@router.get("/partials/video-list", response_class=HTMLResponse)
async def video_list(request: Request, services: Services) -> HTMLResponse:
    jobs = await services.store.list()
    return templates.TemplateResponse(
        request,
        "partials/video_list.html",
        {
            "rows": video_rows(jobs, services.settings),
            "has_active": any(not job.finished for job in jobs),
        },
    )


@router.get("/videos/{job_id}", response_class=HTMLResponse)
async def video_page(job_id: str, request: Request, services: Services) -> HTMLResponse:
    canonical = parse_job_id(job_id)
    job = await services.store.get(canonical) if canonical else None
    if job is None:
        return templates.TemplateResponse(request, "not_found.html", {}, status_code=404)
    settings = services.settings
    return templates.TemplateResponse(
        request,
        "video.html",
        {
            "page": build_video_page(job, settings),
            "agency_name": settings.agency_name,
            "logo_description": settings.agency_logo_description,
        },
    )
