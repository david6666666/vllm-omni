# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.diffusion import backends

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


class Response:
    def __init__(self, payload=None, status=200):
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        return self.payload

    async def text(self):
        return str(self.payload)

    async def read(self):
        return b"video"


class VideoSession:
    def __init__(self, clock, *, post_status=200, poll_status="completed", http_error=None):
        self.clock = clock
        self.post_status = post_status
        self.poll_status = poll_status
        self.http_error = http_error
        self.deleted = []
        self.content_read = False
        self.polls = 0

    def post(self, *args, **kwargs):
        if self.http_error:
            raise self.http_error
        return Response({"id": "video-test", "status": "queued"}, self.post_status)

    def get(self, url):
        if url.endswith("/content"):
            self.content_read = True
            return Response()
        # A queued job takes over ten minutes, independent of wall-clock time.
        self.clock.now = 600.0 + self.polls + 1
        self.polls += 1
        status = self.poll_status
        if isinstance(status, list):
            status = status[self.polls - 1]
        return Response({"status": status, "error": "generation failed"})

    def delete(self, url):
        self.deleted.append(url)
        return Response()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timeout", "status", "success", "error"),
    [
        (600, "in_progress", False, "after 600s"),
        (1800, ["in_progress", "completed"], True, ""),
        (600, "completed", True, ""),
        (600, "failed", False, "Video job failed"),
    ],
)
async def test_video_poll_timeout_and_terminal_status(monkeypatch, timeout, status, success, error):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(backends, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    async def sleep(_):
        clock.now += 2

    monkeypatch.setattr(backends.asyncio, "sleep", sleep)
    session = VideoSession(clock, poll_status=status)
    progress = Mock()
    request = backends.RequestFuncInput(prompt="test", api_url="http://test/v1/videos", model="MiniMax-H3")
    request.video_poll_timeout = timeout
    output = await backends.async_request_v1_videos(request, session, progress)
    assert output.success is success
    assert error in output.error
    assert output.latency == (602.0 if isinstance(status, list) else 601.0)
    assert session.content_read is success
    assert session.deleted == ["http://test/v1/videos/video-test"]
    progress.update.assert_called_once_with(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["http_error", "timeout_exception", "invalid_references"])
async def test_video_failures_always_update_progress(monkeypatch, mode):
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(backends, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    session = VideoSession(
        SimpleNamespace(now=0.0),
        post_status=503 if mode == "http_error" else 200,
        http_error=TimeoutError() if mode == "timeout_exception" else None,
    )
    request = backends.RequestFuncInput(prompt="test", api_url="http://test/v1/videos", model="MiniMax-H3")
    if mode == "invalid_references":
        request.image_paths = ["image.png"]
        request.video_paths = ["video.mp4"]
    progress = Mock()
    output = await backends.async_request_v1_videos(request, session, progress)
    assert not output.success
    assert output.error
    if mode == "timeout_exception":
        assert "TimeoutError" in output.error
    assert output.latency == 1.0
    progress.update.assert_called_once_with(1)
