"""Upload -> background job -> report, through the real app with a stub agent."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from video_qa.agent import ReviewOutcome
from video_qa.main import create_app
from video_qa.notifier import SlackNotifier
from video_qa.schemas import DeconstructedVideo, VideoQAReport

from helpers import make_settings

SLACK_ID = "U0123ABCD"


class StubAgent:
    """Stands in for ClaudeQAAgent: records what it was shown and returns a fixed report."""

    def __init__(self, report: VideoQAReport | None = None, error: BaseException | None = None):
        self.report = report
        self.error = error
        self.seen: list[DeconstructedVideo] = []

    async def review(self, video: DeconstructedVideo) -> ReviewOutcome:
        self.seen.append(video)
        if self.error is not None:
            raise self.error
        assert self.report is not None
        return ReviewOutcome(
            report=self.report, model_report=self.report.model_copy(update={"passed_qa": True})
        )


FAILING_REPORT = VideoQAReport(
    passed_qa=False,
    logo_present=True,
    logo_missing_timestamps=[1.001, 2.002],
    hook_time_seconds=0.4,
    audio_visual_mismatch=[],
    dead_air=[],
    feedback_for_editor="Logo drops out from 0:01 to 0:02; keep the watermark on top.",
)


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    return tmp_path / "media"


def job_dirs(media_root: Path) -> list[Path]:
    return sorted(p for p in media_root.iterdir() if p.is_dir()) if media_root.exists() else []


def make_client(media_root: Path, **kwargs: Any) -> TestClient:
    settings_overrides = kwargs.pop("settings_overrides", {})
    kwargs.setdefault("settings", make_settings(media_dir=media_root, **settings_overrides))
    return TestClient(create_app(**kwargs))


@pytest.fixture
def notifier() -> SlackNotifier:
    return SlackNotifier()


@pytest.fixture
def agent() -> StubAgent:
    return StubAgent(FAILING_REPORT)


@pytest.fixture
def client(agent: StubAgent, notifier: SlackNotifier, media_root: Path) -> Iterator[TestClient]:
    with make_client(media_root, agent=agent, notifier=notifier) as test_client:
        yield test_client


def upload(
    client: TestClient,
    video: Path,
    *,
    slack_id: str | None = None,
    editor_name: str | None = "Sam",
    name: str = "kitchen-ad-v3.mp4",
) -> Any:
    data = {}
    if slack_id is not None:
        data["editor_slack_id"] = slack_id
    if editor_name is not None:
        data["editor_name"] = editor_name
    with video.open("rb") as handle:
        return client.post("/v1/qa-jobs", files={"file": (name, handle, "video/mp4")}, data=data)


def wait_for(client: TestClient, job_id: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job: dict[str, Any] = client.get(f"/v1/qa-jobs/{job_id}").json()
        if job["status"] in {"completed", "error"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def wait_until_gone(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} still exists")
        time.sleep(0.05)


# --- The full flow ----------------------------------------------------------------------------


def test_upload_runs_job_and_reports(
    agent: StubAgent, notifier: SlackNotifier, synthetic_video: Path, media_root: Path
) -> None:
    with make_client(
        media_root, agent=agent, notifier=notifier, settings_overrides={"notifier": "slack"}
    ) as client:
        response = upload(client, synthetic_video, slack_id=SLACK_ID)

        assert response.status_code == 202, response.text
        created = response.json()
        assert created["status"] == "queued"
        assert created["stage"] == "received"
        assert created["video_name"] == "kitchen-ad-v3.mp4"
        assert created["editor_name"] == "Sam"
        assert created["duration_seconds"] == pytest.approx(5.5, abs=0.05)
        assert response.headers["location"] == f"/v1/qa-jobs/{created['job_id']}"

        job = wait_for(client, created["job_id"])
        assert job["status"] == "completed", job
        assert job["stage"] == "done"
        assert job["error"] is None
        assert VideoQAReport.model_validate(job["report"]) == FAILING_REPORT
        assert job["model_passed_qa"] is True  # the model's own verdict, kept beside ours
        assert job["editor_notified"] is True
        assert job["finished_at"] is not None
        assert list(job["stage_started_at"]) == [
            "received",
            "reading",
            "splitting",
            "listening",
            "reviewing",
            "done",
        ]
        assert [f["timestamp_seconds"] for f in job["frames"]] == pytest.approx(
            [0, 1.001, 2.002, 3.003, 4.004, 5.005], abs=1e-3
        )
        assert job["frames"][0]["spoken"].startswith("Stop scrolling")
        assert job["transcript"]
        assert job["transcript_simulated"] is True
        assert job["model"] == "claude-sonnet-5-5"
        assert [c["key"] for c in job["checks"]] == ["logo", "hook", "footage", "silence"]
        assert job["verdict"]["headline"] == "Needs fixes: 1 of 4 checks failed"
        assert job["status_label"] == "Needs fixes · 1 thing"
        assert job["editor_summary"].startswith("QA failed for kitchen-ad-v3.mp4")

        # The pipeline really deconstructed the upload before calling the agent.
        (video,) = agent.seen
        assert len(video.frames) == 6
        assert video.words
        assert video.metadata.filename == "video.mp4"

        (message,) = notifier.sent
        assert message["text"].splitlines()[:2] == [
            f"<@{SLACK_ID}> QA failed for *kitchen-ad-v3.mp4*",
            "❌ Logo missing 0:01–0:02",
        ]

        # The video and frames are kept for the dashboard; only the extracted audio goes.
        folder = media_root / created["job_id"]
        wait_until_gone(folder / "audio.wav")
        assert (folder / "video.mp4").is_file()
        assert video.frames[0].path.is_file()


def test_notifier_none_sends_nothing_to_slack(
    agent: StubAgent, notifier: SlackNotifier, synthetic_video: Path, media_root: Path
) -> None:
    # NOTIFIER defaults to "none": even an injected Slack notifier is never called, and a
    # Slack ID is optional.
    with make_client(media_root, agent=agent, notifier=notifier) as client:
        with_id = wait_for(
            client, upload(client, synthetic_video, slack_id=SLACK_ID).json()["job_id"]
        )
        without_id = wait_for(client, upload(client, synthetic_video).json()["job_id"])
    for job in (with_id, without_id):
        assert job["status"] == "completed"
        assert job["report"]["passed_qa"] is False
        assert job["editor_notified"] is False
    assert list(notifier.sent) == []


def test_slack_id_is_required_only_with_slack_notifier(
    agent: StubAgent, synthetic_video: Path, media_root: Path
) -> None:
    with make_client(media_root, agent=agent, settings_overrides={"notifier": "slack"}) as client:
        response = upload(client, synthetic_video)
    assert response.status_code == 422
    assert response.json()["message"] == (
        "Add the editor's Slack member ID so they can be told about the result."
    )
    assert job_dirs(media_root) == []


def test_passing_video_does_not_notify(
    notifier: SlackNotifier, synthetic_video: Path, media_root: Path
) -> None:
    passing = FAILING_REPORT.model_copy(update={"passed_qa": True, "logo_missing_timestamps": []})
    with make_client(
        media_root,
        agent=StubAgent(passing),
        notifier=notifier,
        settings_overrides={"notifier": "slack"},
    ) as client:
        job = wait_for(client, upload(client, synthetic_video, slack_id=SLACK_ID).json()["job_id"])
    assert job["status"] == "completed"
    assert job["report"]["passed_qa"] is True
    assert job["verdict"]["headline"] == "Ready to post: passed all 4 checks"
    assert job["editor_notified"] is False
    assert list(notifier.sent) == []


def test_pipeline_failure_marks_job_as_error(
    notifier: SlackNotifier, synthetic_video: Path, media_root: Path
) -> None:
    agent = StubAgent(error=RuntimeError("model unavailable"))
    with make_client(media_root, agent=agent, notifier=notifier) as client:
        job = wait_for(client, upload(client, synthetic_video).json()["job_id"])
        assert job["status"] == "error"
        assert job["stage"] == "reviewing"  # where it stopped
        assert job["error"] == "RuntimeError: model unavailable"
        assert job["error_message"].startswith("Something went wrong on our side")
        assert job["report"] is None
        assert job["checks"] == []
        assert job["status_label"] == "Couldn't check"
        wait_until_gone(media_root / job["job_id"] / "audio.wav")
        assert (media_root / job["job_id"] / "video.mp4").is_file()  # still viewable
    assert list(notifier.sent) == []


def test_missing_api_key_failure_is_explained(synthetic_video: Path, media_root: Path) -> None:
    error = TypeError("Could not resolve authentication method. Expected one of api_key...")
    with make_client(media_root, agent=StubAgent(error=error)) as client:
        job = wait_for(client, upload(client, synthetic_video).json()["job_id"])
    assert job["error_message"].startswith("The AI reviewer isn't connected: no API key")


# --- Rejected uploads -------------------------------------------------------------------------


def test_bad_file_is_rejected_with_422_and_cleaned_up(
    client: TestClient, media_dir: Path, media_root: Path
) -> None:
    junk = media_dir / "not-a-video.mp4"
    junk.write_bytes(b"definitely not an mp4")
    response = upload(client, junk)
    assert response.status_code == 422
    assert "Not a readable video file" in response.json()["detail"]
    assert response.json()["message"].startswith("This file couldn't be opened as a video.")
    assert job_dirs(media_root) == []


def test_audio_only_file_is_rejected(client: TestClient, audio_only: Path) -> None:
    response = upload(client, audio_only)
    assert response.status_code == 422
    assert response.json()["detail"] == "The file has no video stream."
    assert response.json()["message"].startswith("This file has sound but no picture")


def test_too_long_video_is_rejected_in_plain_words(
    agent: StubAgent, synthetic_video: Path, media_root: Path
) -> None:
    with make_client(media_root, agent=agent, settings_overrides={"max_frames": 3}) as client:
        response = upload(client, synthetic_video)
    assert response.status_code == 422
    assert response.json()["message"] == (
        "This video is 0:05 long, and the longest video we can check is 0:03. "
        "Please upload a shorter cut."
    )


def test_invalid_slack_id_is_rejected(client: TestClient, synthetic_video: Path) -> None:
    response = upload(client, synthetic_video, slack_id="<!channel>")
    assert response.status_code == 422
    assert "not a Slack user ID" in response.json()["detail"]
    assert "starts with U or W" in response.json()["message"]


def test_oversized_upload_is_rejected_by_content_length(
    agent: StubAgent, synthetic_video: Path, media_root: Path
) -> None:
    with make_client(
        media_root, agent=agent, settings_overrides={"max_upload_bytes": 100_000}
    ) as client:
        response = upload(client, synthetic_video)
    assert response.status_code == 413
    assert response.json()["message"].endswith(
        "the largest file we can take is 0.1 MB. Please export a smaller file."
    )
    assert job_dirs(media_root) == []


def test_oversized_upload_is_caught_while_streaming_to_disk(
    agent: StubAgent, synthetic_video: Path, media_root: Path
) -> None:
    # Small enough overshoot that the request passes the Content-Length check (which allows
    # for multipart overhead), so the cap in the copy loop is what stops it.
    cap = synthetic_video.stat().st_size - 1_000
    with make_client(
        media_root, agent=agent, settings_overrides={"max_upload_bytes": cap}
    ) as client:
        response = upload(client, synthetic_video)
    assert response.status_code == 413
    assert response.json()["detail"] == f"Upload exceeds {cap} bytes"
    assert response.json()["message"].startswith("This file is bigger than")
    assert job_dirs(media_root) == []


def test_upload_refused_when_reviewer_not_connected(
    synthetic_video: Path, media_root: Path
) -> None:
    # No agent injected and no ANTHROPIC_API_KEY: refuse up front instead of failing later.
    with make_client(media_root) as client:
        response = upload(client, synthetic_video)
    assert response.status_code == 503
    assert response.json()["message"] == (
        "The AI reviewer isn't connected yet, so new videos can't be checked."
    )


# --- Reading results back ---------------------------------------------------------------------


def test_list_endpoint_returns_newest_first(client: TestClient, synthetic_video: Path) -> None:
    first = upload(client, synthetic_video, name="first.mp4").json()["job_id"]
    second = upload(client, synthetic_video, name="second.mp4", editor_name=None).json()["job_id"]
    wait_for(client, first)
    wait_for(client, second)

    listed = client.get("/v1/qa-jobs").json()
    assert [item["job_id"] for item in listed] == [second, first]
    newest = listed[0]
    assert newest["video_name"] == "second.mp4"
    assert newest["editor_name"] is None
    assert newest["status"] == "completed"
    assert newest["status_label"] == "Needs fixes · 1 thing"
    assert newest["issue_count"] == 1
    assert newest["passed_qa"] is False
    assert newest["thumbnail_url"] == f"/v1/qa-jobs/{second}/frames/0.jpg"


def test_video_supports_range_requests(client: TestClient, synthetic_video: Path) -> None:
    job_id = upload(client, synthetic_video).json()["job_id"]
    wait_for(client, job_id)
    size = synthetic_video.stat().st_size

    full = client.get(f"/v1/qa-jobs/{job_id}/video")
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-type"] == "video/mp4"
    assert len(full.content) == size

    partial = client.get(f"/v1/qa-jobs/{job_id}/video", headers={"Range": "bytes=100-199"})
    assert partial.status_code == 206
    assert partial.headers["content-range"] == f"bytes 100-199/{size}"
    assert partial.content == synthetic_video.read_bytes()[100:200]


def test_frame_images(client: TestClient, synthetic_video: Path) -> None:
    job_id = upload(client, synthetic_video).json()["job_id"]
    wait_for(client, job_id)

    frame = client.get(f"/v1/qa-jobs/{job_id}/frames/0.jpg")
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"
    assert frame.content.startswith(b"\xff\xd8")

    assert client.get(f"/v1/qa-jobs/{job_id}/frames/6.jpg").status_code == 404
    assert client.get(f"/v1/qa-jobs/{job_id}/frames/-1.jpg").status_code == 404
    assert client.get(f"/v1/qa-jobs/{job_id}/frames/x.jpg").status_code == 422
    assert client.get(f"/v1/qa-jobs/{job_id}/logo").status_code == 404  # no logo configured


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/v1/qa-jobs/does-not-exist").status_code == 404
    assert client.get(f"/v1/qa-jobs/{uuid.uuid4()}").status_code == 404
    assert client.get(f"/v1/qa-jobs/{uuid.uuid4()}/video").status_code == 404
    assert client.get("/v1/qa-jobs/..%2F..%2Fsecrets/video").status_code == 404


def test_unknown_video_page_is_friendly(client: TestClient) -> None:
    for job_id in (str(uuid.uuid4()), "not-a-real-id"):
        response = client.get(f"/videos/{job_id}")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("text/html")
        assert "We couldn't find that video" in response.text
        assert 'href="/"' in response.text


def test_only_the_newest_jobs_are_kept(
    agent: StubAgent, synthetic_video: Path, media_root: Path
) -> None:
    with make_client(media_root, agent=agent, settings_overrides={"max_stored_jobs": 1}) as client:
        old = upload(client, synthetic_video).json()["job_id"]
        wait_for(client, old)
        new = upload(client, synthetic_video).json()["job_id"]
        wait_for(client, new)
        assert [item["job_id"] for item in client.get("/v1/qa-jobs").json()] == [new]
        assert client.get(f"/v1/qa-jobs/{old}").status_code == 404
    assert [p.name for p in job_dirs(media_root)] == [new]


def test_startup_clears_leftover_job_folders(agent: StubAgent, media_root: Path) -> None:
    leftover = media_root / uuid.uuid4().hex
    (leftover / "frames").mkdir(parents=True)
    unrelated = media_root / "keep-me"
    unrelated.mkdir()
    with make_client(media_root, agent=agent):
        assert not leftover.exists()
        assert unrelated.exists()  # only job-shaped folders are ever deleted


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "active_jobs": 0}
