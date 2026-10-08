"""The dashboard pages, rendered from hand-built job records (no pipeline, no API key)."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from video_qa.agent import ReviewOutcome
from video_qa.jobs import InMemoryJobStore, Job, JobStage, JobStatus, utc_now
from video_qa.main import create_app
from video_qa.schemas import AudioVisualMismatch, DeconstructedVideo, VideoQAReport

from helpers import make_job, make_settings, segment


class UnusedAgent:
    async def review(self, video: DeconstructedVideo) -> ReviewOutcome:
        raise AssertionError("pages tests never run the pipeline")


PASSING = VideoQAReport(
    passed_qa=True,
    logo_present=True,
    logo_missing_timestamps=[],
    hook_time_seconds=0.4,
    audio_visual_mismatch=[],
    dead_air=[],
    feedback_for_editor="Clean cut. Ready to go.",
)
FAILING = VideoQAReport(
    passed_qa=False,
    logo_present=True,
    logo_missing_timestamps=[4.004],
    hook_time_seconds=0.4,
    audio_visual_mismatch=[
        AudioVisualMismatch(timestamp_seconds=2.002, spoken="our new kitchen", shown="a bathroom")
    ],
    dead_air=[segment(3.1, 4.6)],
    feedback_for_editor="Fix the logo at 0:04 & the shot at 0:02.",
)


def jobs_by_state() -> dict[str, Job]:
    now = utc_now()
    done = dict.fromkeys(JobStage, now)
    return {
        "checking": make_job(
            status=JobStatus.PROCESSING,
            stage=JobStage.SPLITTING,
            frames=[],
            stage_started_at={
                JobStage.RECEIVED: now,
                JobStage.READING: now,
                JobStage.SPLITTING: now,
            },
        ),
        "passed": make_job(
            status=JobStatus.COMPLETED,
            stage=JobStage.DONE,
            stage_started_at=done,
            finished_at=now,
            report=PASSING,
            model_report=PASSING,
            model="claude-sonnet-5-5",
            transcript_simulated=True,
        ),
        "failed": make_job(
            status=JobStatus.COMPLETED,
            stage=JobStage.DONE,
            stage_started_at=done,
            finished_at=now,
            report=FAILING,
            model_report=FAILING.model_copy(update={"passed_qa": True}),
            model="claude-sonnet-5-5",
            transcript_simulated=True,
        ),
        "error": make_job(
            status=JobStatus.ERROR,
            stage=JobStage.REVIEWING,
            stage_started_at={JobStage.RECEIVED: now, JobStage.REVIEWING: now},
            finished_at=now,
            error="APIConnectionError: Connection error.",
            error_message=(
                "The AI reviewer couldn't be reached. "
                "Check the internet connection, then try again."
            ),
        ),
    }


@pytest.fixture
def jobs() -> dict[str, Job]:
    return jobs_by_state()


def app_with(jobs: list[Job], tmp_path: Path, **kwargs: Any) -> TestClient:
    store = InMemoryJobStore()
    for job in jobs:
        asyncio.run(store.save(job))
    kwargs.setdefault("agent", UnusedAgent())
    settings = make_settings(media_dir=tmp_path / "media", **kwargs.pop("settings", {}))
    return TestClient(create_app(settings=settings, store=store, **kwargs))


@pytest.fixture
def client(jobs: dict[str, Job], tmp_path: Path) -> Iterator[TestClient]:
    with app_with(list(jobs.values()), tmp_path) as test_client:
        yield test_client


# --- Visible text -----------------------------------------------------------------------------


class _VisibleText(HTMLParser):
    """Text a viewer can see or hear: text nodes plus alt, aria-label, title and placeholder."""

    SKIP: ClassVar[set[str]] = {"script", "style"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skipping += 1
            return
        for name, value in attrs:
            if name in {"alt", "aria-label", "title", "placeholder"} and value:
                self.parts.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skipping -= 1

    def handle_data(self, data: str) -> None:
        if not self._skipping and data.strip():
            self.parts.append(data.strip())


def visible_text(html: str, *, include_technical: bool = False) -> str:
    if not include_technical:
        html = re.sub(r'<details class="technical">.*?</details>', "", html, flags=re.S)
    parser = _VisibleText()
    parser.feed(html)
    return "\n".join(parser.parts)


BANNED = [
    (r"\bjobs?\b", re.I),
    (r"\bframe index\b", re.I),
    (r"\bschema\b", re.I),
    (r"\bnull\b", re.I),
    (r"\bnone\b", re.I),
    (r"\b(?:true|false)\b", re.I),
    (r"\b(?:200|202|206|404|413|422|500|503)\b(?!\s*MB)", 0),
    (r"\b[a-z]+_[a-z_]+\b", 0),  # snake_case field names such as passed_qa
]


def banned_words_in(text: str) -> list[str]:
    return [m.group(0) for pattern, flags in BANNED for m in re.finditer(pattern, text, flags)]


# --- Report page, in each state ---------------------------------------------------------------

EXPECTED = {
    "checking": ["Checking this video", "Splitting it into frames", "Time so far"],
    "passed": ["Ready to post: passed all 4 checks", "This review was done by an AI"],
    "failed": ["Needs fixes: 3 of 4 checks failed", "How to fix:", "Copy summary for the editor"],
    "error": [
        "We couldn't check this video",
        "The AI reviewer couldn't be reached.",
        "This isn't a verdict on the video",
    ],
}


@pytest.mark.parametrize("state", ["checking", "passed", "failed", "error"])
def test_report_page_renders_in_plain_words(
    client: TestClient, jobs: dict[str, Job], state: str
) -> None:
    job = jobs[state]
    response = client.get(f"/videos/{job.job_id}")
    assert response.status_code == 200
    assert f'data-state="{state}"' in response.text
    text = visible_text(response.text)
    for phrase in EXPECTED[state]:
        assert phrase in text
    assert banned_words_in(text) == []
    # The technical section is still there for developers, with the field names.
    assert "Technical details" in response.text
    assert job.job_id in visible_text(response.text, include_technical=True)


def test_checking_page_highlights_the_current_step(
    client: TestClient, jobs: dict[str, Job]
) -> None:
    html = client.get(f"/videos/{jobs['checking'].job_id}").text
    current = re.search(r'<li class="step step-current"[^>]*aria-current="step"', html)
    assert current is not None
    assert 'data-stage="splitting"' in current.group(0)
    assert html.count("step step-done") == 2  # uploaded, reading
    assert "Done" in visible_text(html)


def test_passed_report_shows_four_passing_checks(client: TestClient, jobs: dict[str, Job]) -> None:
    text = visible_text(client.get(f"/videos/{jobs['passed'].job_id}").text)
    assert text.count("Passed") >= 4
    assert "Needs fixing" not in text
    for title in (
        "Logo stays on screen",
        "Opening line starts quickly",
        "Footage matches what's being said",
        "No silent gaps",
    ):
        assert title in text
    assert "Speech is simulated in this demo" in text
    assert "The logo check is real" in text


def test_failing_report_shows_where_and_how_to_fix(
    client: TestClient, jobs: dict[str, Job]
) -> None:
    html = client.get(f"/videos/{jobs['failed'].job_id}").text
    text = visible_text(html)
    assert text.count("Needs fixing") == 3
    assert text.count("Passed") >= 1  # the opening line is fine, and still shown
    assert "Logo missing at 0:04" in text
    assert "Footage doesn't match at 0:02" in text
    assert "What was said" in text
    assert "our new kitchen" in text
    assert "What was shown" in text
    assert "a bathroom" in text
    assert "Silent from 0:03 to 0:04" in text
    assert "Frame at 0:04" in text  # thumbnail alt text
    assert "Measured from the audio" in text
    assert "AI's judgement" in text
    assert "No logo image was given" in text
    # Timeline: four labelled lanes, deadline marker, and blocks that jump the video.
    for lane in ("Logo", "Opening line", "Footage", "Silence"):
        assert lane in text
    assert "Must start by 0:03" in text
    assert 'class="block"' in html
    assert 'data-seek="3.004"' in html
    # Model-written text is escaped, never interpreted as markup.
    assert "Fix the logo at 0:04 &amp; the shot at 0:02." in html


def test_couldnt_check_never_looks_like_a_failed_video(
    client: TestClient, jobs: dict[str, Job]
) -> None:
    html = client.get(f"/videos/{jobs['error'].job_id}").text
    text = visible_text(html)
    assert "verdict-error" in html
    assert "verdict-failed" not in html
    assert "Needs fix" not in text
    assert "Ready to post" not in text
    assert "APIConnectionError" not in text
    assert "APIConnectionError" in visible_text(html, include_technical=True)


def test_simulated_notice_follows_the_transcriber_flag(tmp_path: Path) -> None:
    real = make_job(
        status=JobStatus.COMPLETED, stage=JobStage.DONE, report=PASSING, transcript_simulated=False
    )
    with app_with([real], tmp_path) as client:
        text = visible_text(client.get(f"/videos/{real.job_id}").text)
    assert "Speech is simulated" not in text


def test_logo_card_shows_the_reference_logo(tmp_path: Path) -> None:
    job = make_job(
        status=JobStatus.COMPLETED,
        stage=JobStage.DONE,
        report=PASSING,
        logo_reference="northside-logo.png",
        logo_file="logo.png",
    )
    with app_with([job], tmp_path) as client:
        html = client.get(f"/videos/{job.job_id}").text
    assert f'src="/v1/qa-jobs/{job.job_id}/logo"' in html
    assert "The reviewer looked for this logo (northside-logo.png)." in visible_text(html)


# --- Home page --------------------------------------------------------------------------------


def test_home_lists_videos_with_counts(client: TestClient, jobs: dict[str, Job]) -> None:
    html = client.get("/").text
    text = visible_text(html)
    assert "Upload a finished video" in text
    assert "It's checked against four rules" in text
    assert "Videos up to 1:40 long and 500 MB in size." in text
    assert "Needs fixes · 3 things" in text
    assert "Ready to post" in text
    assert "Couldn't check" in text
    assert "Checking · Splitting it into frames" in text
    for definition in ("The hook is", "B-roll is", "Dead air is"):
        assert definition in text
    assert "Sam · Today" in text
    assert "0:06 long" in text
    assert "<fieldset >" in html or "<fieldset>" in html  # upload enabled
    assert banned_words_in(text) == []


def test_home_empty_state(tmp_path: Path) -> None:
    with app_with([], tmp_path) as client:
        html = client.get("/").text
    assert "No videos checked yet" in visible_text(html)
    assert 'data-active="false"' in html


def test_video_list_partial_marks_active_checks(client: TestClient) -> None:
    html = client.get("/partials/video-list").text
    assert 'data-active="true"' in html
    assert "<html" not in html


def test_home_explains_a_missing_reviewer_and_disables_upload(tmp_path: Path) -> None:
    store_jobs: list[Job] = []
    with app_with(store_jobs, tmp_path, agent=None) as client:
        html = client.get("/").text
    text = visible_text(html)
    assert "The AI reviewer isn't connected yet, so new videos can't be checked." in text
    assert "ANTHROPIC_API_KEY=" in text
    assert "<fieldset disabled>" in html


def test_seeded_example_works_without_an_api_key(media_dir: Path, tmp_path: Path) -> None:
    # media_dir is only requested so this test skips cleanly where ffmpeg is missing.
    with app_with([], tmp_path, agent=None, settings={"seed_example": True}) as client:
        listed = client.get("/v1/qa-jobs").json()
        assert len(listed) == 1
        example = listed[0]
        assert example["is_example"] is True
        assert example["status_label"] == "Needs fixes · 6 things"
        home = visible_text(client.get("/").text)
        assert "Example" in home
        page = client.get(f"/videos/{example['job_id']}")
        text = visible_text(page.text)
        assert "Needs fixes: 4 of 4 checks failed" in text
        assert "Example" in text
        assert "This is a hand-written example" in text
        # The speech notice claims the logo check is real, which isn't true of the example.
        assert "Speech is simulated" not in text
        assert "This review was done by an AI" not in text
        assert banned_words_in(text) == []
        video = client.get(f"/v1/qa-jobs/{example['job_id']}/video", headers={"Range": "bytes=0-9"})
        assert video.status_code == 206
        assert client.get(f"/v1/qa-jobs/{example['job_id']}/frames/0.jpg").status_code == 200
