import logging
import os
from collections.abc import Callable

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


async def complete_interview(http, interview_id, user_id):
    if not interview_id:
        return
    await _post(
        http, f"{BACKEND_URL}/interview/{interview_id}/complete", {"user_id": user_id}
    )


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
