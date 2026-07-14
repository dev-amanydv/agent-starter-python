import aiohttp

import transcript
from transcript import TurnAggregator


def _collect():
    turns: list[tuple[str, str, float]] = []
    return turns, lambda role, content, created_at: turns.append(
        (role, content, created_at)
    )


def test_merges_consecutive_same_role_items():
    """A user answer split across several committed items becomes one turn."""
    turns, on_turn = _collect()
    agg = TurnAggregator(on_turn)

    agg.add("user", "I worked on an project, like,", 1.0)
    agg.add("user", "would ask the user about", 2.0)
    agg.add("user", "context of user's resume.", 3.0)
    assert turns == []

    agg.add("assistant", "That sounds relevant.", 4.0)
    assert turns == [
        (
            "user",
            "I worked on an project, like, would ask the user about context of user's resume.",
            1.0,
        )
    ]


def test_flush_emits_trailing_turn():
    """flush() (called at session end) persists the last open turn."""
    turns, on_turn = _collect()
    agg = TurnAggregator(on_turn)

    agg.add("assistant", "Welcome to the interview.", 1.0)
    agg.add("user", "Hey.", 2.0)
    agg.add("user", "I'm ready.", 3.0)
    agg.flush()

    assert turns == [
        ("assistant", "Welcome to the interview.", 1.0),
        ("user", "Hey. I'm ready.", 2.0),
    ]


def test_created_at_is_turn_start():
    """The merged turn keeps the timestamp of its first item."""
    turns, on_turn = _collect()
    agg = TurnAggregator(on_turn)

    agg.add("user", "part one", 10.0)
    agg.add("user", "part two", 20.0)
    agg.flush()

    assert turns == [("user", "part one part two", 10.0)]


def test_ignores_blank_and_unknown_roles():
    """Empty text and non user/assistant roles are dropped, not buffered."""
    turns, on_turn = _collect()
    agg = TurnAggregator(on_turn)

    agg.add("user", "   ", 1.0)
    agg.add("user", None, 2.0)
    agg.add("system", "ignored", 3.0)
    agg.flush()

    assert turns == []


def test_flush_is_idempotent_when_empty():
    """Flushing with nothing buffered emits no turn."""
    turns, on_turn = _collect()
    agg = TurnAggregator(on_turn)

    agg.flush()
    agg.flush()

    assert turns == []


# ── upload_recording ─────────────────────────────────────────────────────────
class _FakeResp:
    def __init__(self, status: int = 201) -> None:
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def text(self) -> str:
        return ""


class _FakeHttp:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "headers": headers})
        return _FakeResp()


async def test_upload_recording_skips_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "INTERNAL_SECRET", "secret")
    http = _FakeHttp()
    await transcript.upload_recording(http, "iv1", tmp_path / "missing.ogg")
    assert http.calls == []


async def test_upload_recording_skips_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "INTERNAL_SECRET", "secret")
    empty = tmp_path / "audio.ogg"
    empty.write_bytes(b"")
    http = _FakeHttp()
    await transcript.upload_recording(http, "iv1", empty)
    assert http.calls == []


async def test_upload_recording_skips_without_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "INTERNAL_SECRET", None)
    f = tmp_path / "audio.ogg"
    f.write_bytes(b"some-audio-bytes")
    http = _FakeHttp()
    await transcript.upload_recording(http, "iv1", f)
    assert http.calls == []


async def test_upload_recording_posts_multipart(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "INTERNAL_SECRET", "secret")
    f = tmp_path / "audio.ogg"
    f.write_bytes(b"some-audio-bytes")
    http = _FakeHttp()

    await transcript.upload_recording(http, "iv1", f)

    assert len(http.calls) == 1
    call = http.calls[0]
    assert call["url"].endswith("/interview/iv1/recording/upload")
    assert call["headers"]["x-internal-secret"] == "secret"
    assert isinstance(call["data"], aiohttp.FormData)
