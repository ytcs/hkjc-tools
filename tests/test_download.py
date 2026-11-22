"""data.download_replay: complete files only, unique temporaries, retries (requests mocked)."""

import pytest
import requests

from hkjc.data import video


class _Resp:
    def __init__(self, body: bytes, length=None, status=200, drop=False):
        self.body, self.status_code, self.drop = body, status, drop
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, n):
        yield self.body[:3]
        if self.drop:
            raise requests.exceptions.ChunkedEncodingError("connection dropped")
        yield self.body[3:]


def _serve(monkeypatch, responses):
    calls = []

    def get(url, **kw):
        calls.append(url)
        return responses[min(len(calls), len(responses)) - 1]

    monkeypatch.setattr(video.requests, "get", get)
    monkeypatch.setattr(video.time, "sleep", lambda s: None)
    return calls


def test_truncated_and_dropped_bodies_are_retried_and_leave_no_partial(tmp_path, monkeypatch):
    calls = _serve(monkeypatch, [_Resp(b"abcdef", length=10), _Resp(b"abcdef", drop=True), _Resp(b"abcdef", 6)])
    out = video.download_replay("20240417", 7, "aerial", root=tmp_path)
    assert out.read_bytes() == b"abcdef" and len(calls) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == [out.name]


def test_persistent_failure_raises_and_leaves_nothing(tmp_path, monkeypatch):
    _serve(monkeypatch, [_Resp(b"abcdef", length=10)])
    with pytest.raises(OSError):
        video.download_replay("20240417", 7, "aerial", root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_missing_replay_is_not_retried(tmp_path, monkeypatch):
    calls = _serve(monkeypatch, [_Resp(b"", status=404)])
    with pytest.raises(requests.HTTPError):
        video.download_replay("20240417", 7, "aerial", root=tmp_path)
    assert len(calls) == 1 and list(tmp_path.iterdir()) == []


def test_concurrent_fetches_use_distinct_temporaries(tmp_path, monkeypatch):
    seen = []
    real_open = video.os.open

    def spy(path, *a, **kw):
        seen.append(str(path))
        return real_open(path, *a, **kw)

    monkeypatch.setattr(video.os, "open", spy)
    _serve(monkeypatch, [_Resp(b"abcdef", 6)])
    video.download_replay("20240417", 7, "aerial", root=tmp_path)
    video.download_replay("20240417", 7, "aerial", root=tmp_path, overwrite=True)
    assert len(seen) == 2 and seen[0] != seen[1] and all(s.endswith(".part") for s in seen)
