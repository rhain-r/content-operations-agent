"""The static showcase export, end to end: lavfi clips, a stub reviewer, no API key, no network."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest

from video_qa.agent import ReviewOutcome, enforce_verdict
from video_qa.export import (
    ExportError,
    ExportResult,
    assign_slugs,
    export_showcase,
    slugify,
)
from video_qa.schemas import AudioVisualMismatch, DeconstructedVideo, VideoQAReport

from helpers import make_settings, run_ffmpeg

REPO_URL = "https://github.com/rhain-r/content-operations-agent"
KEY_SHAPED = re.compile(
    r"sk-ant-[A-Za-z0-9_\-]{10,}|\bsk-[A-Za-z0-9]{20,}|\bgh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}"
)


class CountingStubAgent:
    """Passes clips under 4 s; fails longer ones on the logo and the footage."""

    def __init__(self) -> None:
        self.calls = 0

    async def review(self, video: DeconstructedVideo) -> ReviewOutcome:
        self.calls += 1
        times = [f.timestamp_seconds for f in video.frames]
        hook = video.words[0].start if video.words else None
        if video.metadata.duration_seconds < 4:
            report = VideoQAReport(
                passed_qa=True,
                logo_present=True,
                logo_missing_timestamps=[],
                hook_time_seconds=hook,
                audio_visual_mismatch=[],
                dead_air=[],
                feedback_for_editor="Clean cut.",
            )
        else:
            report = VideoQAReport(
                passed_qa=True,
                logo_present=True,
                logo_missing_timestamps=[times[1]],
                hook_time_seconds=hook,
                audio_visual_mismatch=[
                    AudioVisualMismatch(
                        timestamp_seconds=times[2], spoken="the kitchen", shown="a test pattern"
                    )
                ],
                dead_air=[],
                feedback_for_editor="Put the logo back at 0:01.",
            )
        return ReviewOutcome(
            report=enforce_verdict(report, video, make_settings()), model_report=report
        )


@pytest.fixture
def demo_videos(tmp_path: Path, synthetic_video: Path, media_dir: Path) -> Path:
    folder = tmp_path / "demo_videos"
    folder.mkdir()
    shutil.copyfile(synthetic_video, folder / "Kitchen Tour (final).mp4")
    short = media_dir / "short.mp4"
    if not short.exists():
        run_ffmpeg(
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=3.5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3.5",
            "-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac", "-shortest", str(short),
        )  # fmt: skip
    shutil.copyfile(short, folder / "short-promo.mp4")
    (folder / "broken.mp4").write_bytes(b"not a video at all")
    (folder / "notes.txt").write_text("not a video, and not even looked at")
    return folder


async def export(
    videos: Path, out: Path, agent: CountingStubAgent | None, **kwargs: Any
) -> ExportResult:
    return await export_showcase(
        videos, out, settings=make_settings(), agent=agent, repo_url=REPO_URL, **kwargs
    )


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src", "poster"} and value is not None:
                self.urls.append(value)


def probe(path: Path) -> dict[str, object]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    data: dict[str, object] = json.loads(out)
    return data


async def test_export_writes_a_self_contained_static_site(
    demo_videos: Path, tmp_path: Path
) -> None:
    out = tmp_path / "docs"
    agent = CountingStubAgent()
    result = await export_showcase(
        demo_videos, out, settings=make_settings(), agent=agent, repo_url=REPO_URL
    )

    assert sorted(result.published) == ["kitchen-tour-final", "short-promo"]
    assert agent.calls == 2
    assert [name for name, _ in result.skipped] == ["broken.mp4"]
    assert result.skipped[0][1].startswith("This file couldn't be opened as a video.")
    assert not (out / "videos" / "broken").exists()

    for relative in (
        "index.html",
        "static/app.css",
        "static/app.js",
        "videos/kitchen-tour-final/index.html",
        "videos/kitchen-tour-final/video.mp4",
        "videos/kitchen-tour-final/report.json",
        "videos/kitchen-tour-final/frames/0.jpg",
        "videos/kitchen-tour-final/frames/5.jpg",
        "videos/short-promo/index.html",
    ):
        assert (out / relative).is_file(), relative

    # Every link, image, script and video is relative and points at a file in the output.
    html_files = sorted(out.rglob("*.html"))
    assert len(html_files) == 3
    for page in html_files:
        parser = _Links()
        parser.feed(page.read_text(encoding="utf-8"))
        assert parser.urls
        for url in parser.urls:
            if url == REPO_URL or url.startswith("#"):
                continue
            parts = urlsplit(url)
            assert not parts.scheme, f"{page}: {url} is absolute"
            assert not url.startswith("/"), f"{page}: {url} starts with /"
            target = page.parent / unquote(parts.path)
            if parts.path.endswith("/") or parts.path in {"", "./"}:
                target = target / "index.html"
            assert target.resolve().is_file(), f"{page}: {url} points at nothing"
            assert out.resolve() in target.resolve().parents

    # Nothing that only makes sense on this machine, and nothing shaped like a secret.
    for path in out.rglob("*"):
        if path.suffix not in {".html", ".json", ".css", ".js"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "/v1/" not in text, path
        assert "localhost" not in text, path
        assert not KEY_SHAPED.search(text), path
        assert str(tmp_path) not in text, path
        assert "job_id" not in text or path.suffix == ".js", path

    home = (out / "index.html").read_text(encoding="utf-8")
    assert 'id="upload-form"' not in home
    assert "read-only showcase" in home
    assert f'href="{REPO_URL}"' in home
    assert "Editor A" in home
    assert "Editor B" in home
    assert "Needs fixes · 2 things" in home
    assert "Ready to post" in home
    assert "data-refresh-url" not in home

    report_page = (out / "videos/kitchen-tour-final/index.html").read_text(encoding="utf-8")
    assert "data-status-url" not in report_page
    assert "Needs fixes: 2 of 4 checks failed" in report_page
    assert "Speech is simulated in this demo" in report_page
    assert "This review was done by an AI" in report_page

    record = json.loads((out / "videos/kitchen-tour-final/report.json").read_text())
    assert record["format"] == "video-qa-showcase/1"
    assert record["job"]["video_name"] == "Kitchen Tour (final).mp4"
    assert record["job"]["report"]["passed_qa"] is False
    assert "editor_name" not in record["job"]
    assert "error" not in record["job"]

    # The web copy: H.264 + AAC, long edge at most 1280, moov before mdat (+faststart).
    video = out / "videos/kitchen-tour-final/video.mp4"
    streams = {s["codec_type"]: s for s in probe(video)["streams"]}  # type: ignore[attr-defined]
    assert streams["video"]["codec_name"] == "h264"
    assert streams["audio"]["codec_name"] == "aac"
    assert max(streams["video"]["width"], streams["video"]["height"]) <= 1280
    data = video.read_bytes()
    assert 0 <= data.find(b"moov") < data.find(b"mdat")
    assert result.total_bytes > 0


async def test_saved_reviews_are_reused_unless_refreshed(demo_videos: Path, tmp_path: Path) -> None:
    out = tmp_path / "docs"
    agent = CountingStubAgent()
    await export(demo_videos, out, agent)
    assert agent.calls == 2

    (out / "index.html").unlink()
    again = await export_showcase(
        demo_videos, out, settings=make_settings(), agent=None, repo_url=REPO_URL
    )
    assert agent.calls == 2  # nothing reviewed again, and no reviewer was even needed
    assert sorted(again.reused) == ["kitchen-tour-final", "short-promo"]
    assert (out / "index.html").is_file()  # but the pages are rendered again

    await export(demo_videos, out, agent, refresh=True)
    assert agent.calls == 4


async def test_reviewing_without_a_reviewer_explains_what_to_do(
    demo_videos: Path, tmp_path: Path
) -> None:
    with pytest.raises(ExportError, match="ANTHROPIC_API_KEY"):
        await export(demo_videos, tmp_path / "docs", None)


async def test_files_over_the_size_limit_stop_the_export(demo_videos: Path, tmp_path: Path) -> None:
    with pytest.raises(ExportError, match=r"over the .* limit"):
        await export(demo_videos, tmp_path / "docs", CountingStubAgent(), max_file_bytes=20_000)


async def test_removed_videos_are_unpublished(demo_videos: Path, tmp_path: Path) -> None:
    out = tmp_path / "docs"
    await export(demo_videos, out, CountingStubAgent())
    (demo_videos / "short-promo.mp4").unlink()
    await export(demo_videos, out, None)
    assert not (out / "videos" / "short-promo").exists()
    assert "short-promo" not in (out / "index.html").read_text(encoding="utf-8")


async def test_include_example(media_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "docs"
    result = await export_showcase(
        tmp_path / "missing", out, settings=make_settings(), agent=None, include_example=True
    )
    assert result.published == ["example"]
    page = (out / "videos/example/index.html").read_text(encoding="utf-8")
    assert "This is a hand-written example" in page
    assert "Needs fixes: 4 of 4 checks failed" in page
    assert json.loads((out / "videos/example/report.json").read_text())["job"]["is_example"]
    assert (out / "videos/example/video.mp4").is_file()


def test_slugs_come_from_file_names() -> None:
    assert slugify("Kitchen Tour (final)") == "kitchen-tour-final"
    assert slugify("Café Été") == "cafe-ete"
    assert slugify("!!!") == "video"
    slugs = assign_slugs([Path("a/Promo.mp4"), Path("b/promo.mov"), Path("c/example.mp4")])
    assert list(slugs.values()) == ["promo", "promo-2", "example-2"]
