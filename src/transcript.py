import logging
import os
from collections.abc import Callable
from pathlib import Path

import aiohttp

logger = logging.getLogger("agent.transcript")

BACKEND_URL = os.getenv("INTERVIEW_BACKEND_URL", "http://localhost:8000/api/v1")
INTERNAL_SECRET = os.getenv("AGENT_INTERNAL_SECRET")

_ROLE_MAP = {"user": "User", "assistant": "Assistant"}


async def _post(session: aiohttp.ClientSession, url: str, payload: dict) -> None:
    if not INTERNAL_SECRET:
        logger.warning("AGENT_INTERNAL_SECRET unset; skipping saving")
        return
    try:
        async with session.post(
            url,
            json=payload,
            headers={"x-internal-secret": INTERNAL_SECRET},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status >= 300:
                logger.warning("saving failed: %s %s", resp.status, await resp.text())
    except Exception:
        logger.exception("saving request errored (continuing)")


async def save_message(http, interview_id, role, content, created_at):
    livekit_role = _ROLE_MAP.get(role)
    if not (interview_id and livekit_role and content and content.strip()):
        return
    url = f"{BACKEND_URL}/interview/{interview_id}/messages"
    await _post(
        http, url, {"role": livekit_role, "content": content, "createdAt": created_at}
    )


async def complete_interview(http, interview_id):
    if not interview_id:
        return
    await _post(http, f"{BACKEND_URL}/interview/{interview_id}/complete", {})


def _probe_duration_ms(path: Path) -> int | None:
    """Best-effort recording duration via PyAV (bundled with livekit-agents).

    ``container.duration`` is in microseconds (AV_TIME_BASE). Storing it lets the
    web player render a correct scrubber even when a browser can't derive the
    duration from a streamed Ogg/Opus file.
    """
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration:
                return int(container.duration / 1000)
    except Exception:
        logger.debug("could not probe recording duration", exc_info=True)
    return None


def _transcode_to_m4a(src: Path, dst: Path) -> bool:
    """Transcode the recorder's OGG/Opus output to AAC in an MP4 (.m4a) container.

    Safari and iOS don't reliably play Opus, so we deliver AAC — which every browser
    supports — using PyAV's built-in ``aac`` encoder (no external libs like libmp3lame
    required). ``+faststart`` moves the moov atom to the front so the file streams and
    seeks over HTTP range requests. Returns True only when a non-empty file was written.
    """
    try:
        import av

        with (
            av.open(str(src)) as in_container,
            av.open(
                str(dst),
                mode="w",
                format="mp4",
                options={"movflags": "+faststart"},
            ) as out_container,
        ):
            in_stream = in_container.streams.audio[0]
            rate = in_stream.rate or 48000
            out_stream = out_container.add_stream("aac", rate=rate)
            # The AAC encoder wants planar float stereo; normalize before encoding.
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)

            def _mux(frames):
                for frame in frames:
                    for packet in out_stream.encode(frame):
                        out_container.mux(packet)

            for frame in in_container.decode(in_stream):
                frame.pts = None
                _mux(resampler.resample(frame))
            _mux(resampler.resample(None))  # flush resampler
            _mux([None])  # flush encoder (encode(None))
        return dst.exists() and dst.stat().st_size > 0
    except Exception:
        logger.exception("recording transcode to m4a failed")
        return False


async def upload_recording(
    http,
    interview_id,
    file_path,
    *,
    content_type: str = "audio/ogg",
    filename: str = "interview.ogg",
) -> None:
    """Upload the finalized session recording to the backend (which stores it in R2).

    Multipart POST guarded by the same internal secret as the transcript endpoints.
    Failures are swallowed — a missing recording must never break interview shutdown.
    """
    if not interview_id:
        return
    if not INTERNAL_SECRET:
        logger.warning("AGENT_INTERNAL_SECRET unset; skipping recording upload")
        return
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        logger.warning("recording file missing or empty at %s; skipping upload", path)
        return

    duration_ms = _probe_duration_ms(path)
    url = f"{BACKEND_URL}/interview/{interview_id}/recording/upload"
    try:
        with path.open("rb") as f:
            data = aiohttp.FormData()
            if duration_ms is not None:
                data.add_field("durationMs", str(duration_ms))
            data.add_field("file", f, filename=filename, content_type=content_type)
            async with http.post(
                url,
                data=data,
                headers={"x-internal-secret": INTERNAL_SECRET},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status >= 300:
                    logger.warning(
                        "recording upload failed: %s %s",
                        resp.status,
                        await resp.text(),
                    )
    except Exception:
        logger.exception("recording upload request errored (continuing)")


async def prepare_and_upload_recording(http, interview_id, ogg_path) -> None:
    """Transcode the OGG recording to Safari/iOS-friendly AAC, then upload it.

    Falls back to uploading the original OGG if transcoding fails, so a recording is
    never lost just because the AAC encode hit a problem.
    """
    if not interview_id:
        return
    src = Path(ogg_path)
    if not src.exists() or src.stat().st_size == 0:
        logger.warning("recording file missing or empty at %s; skipping upload", src)
        return

    m4a = src.with_suffix(".m4a")
    if _transcode_to_m4a(src, m4a):
        await upload_recording(
            http,
            interview_id,
            m4a,
            content_type="audio/mp4",
            filename="interview.m4a",
        )
    else:
        await upload_recording(http, interview_id, src)


class TurnAggregator:
    """Merge consecutive same-role conversation items into a single turn.

    LiveKit has no dedicated "turn committed" event; ``conversation_item_added``
    fires once per chat item, and turn detection can split one spoken answer
    across several items (e.g. when the speaker pauses mid-sentence). This
    buffers items and emits one merged message per speaker turn, flushing when
    the speaker changes or the session ends.

    ``on_turn`` is called ``on_turn(role, content, created_at)`` with the raw
    LiveKit role ("user"/"assistant"), the joined text, and the timestamp of the
    turn's first item.
    """

    def __init__(self, on_turn: Callable[[str, str, float], None]) -> None:
        self._on_turn = on_turn
        self._role: str | None = None
        self._parts: list[str] = []
        self._created_at: float | None = None

    def add(self, role: str, text: str | None, created_at: float) -> None:
        if role not in _ROLE_MAP:
            return
        text = (text or "").strip()
        if not text:
            return
        if self._role is not None and role != self._role:
            self.flush()
        if self._role is None:
            self._role = role
            self._created_at = created_at
        self._parts.append(text)

    def flush(self) -> None:
        if not self._parts:
            return
        role, content, created_at = self._role, " ".join(self._parts), self._created_at
        self._role = None
        self._parts = []
        self._created_at = None
        self._on_turn(role, content, created_at)
