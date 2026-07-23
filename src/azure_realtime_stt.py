"""Realtime (websocket) STT against Azure OpenAI.

``livekit-plugins-openai``'s ``STT.with_azure(use_realtime=True)`` cannot talk to Azure's
realtime transcription websocket: its ``_connect_ws`` builds an OpenAI-shaped request that
Azure rejects. Two independent incompatibilities, both verified against our endpoint:

1. The handshake. The plugin derives the URL from the client's ``base_url``, which
   ``azure_deployment`` makes deployment-scoped, sends no ``api-version``, and authenticates
   with ``Authorization: Bearer``. Azure serves realtime from a flat ``/openai/realtime``,
   requires ``api-version``, and wants an ``api-key`` header. The plugin's URL returns 404 —
   the failure previously attributed to Azure not exposing the endpoint at all.
2. The protocol. The plugin speaks the newer GA shape (``session.update`` with
   ``session.type = "transcription"``); Azure answers ``Unknown parameter: 'session.type'``
   and only accepts the older ``transcription_session.update``.

Everything else in the plugin is already wire-compatible: it streams
``input_audio_buffer.append`` / ``.commit`` and reads
``conversation.item.input_audio_transcription.delta`` / ``.completed`` plus the
``input_audio_buffer.speech_*`` events, all of which are identical across the two protocol
versions. So this subclass overrides ``_connect_ws`` and inherits the rest.

Caveat: Azure sends no ``.delta`` events on this api-version, only finals. Interim
transcripts therefore never fire, and ``preemptive_generation`` has nothing to act on
before the turn ends. Finals still arrive promptly (~0.4s after end-of-speech), which is
the point: batch STT cannot start uploading until the turn is already over.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlencode

import aiohttp
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins import openai as lk_openai
from openai import AsyncAzureOpenAI

REALTIME_API_VERSION = "2025-04-01-preview"


class AzureRealtimeSTT(lk_openai.STT):
    """Azure OpenAI STT over the realtime websocket, speaking Azure's preview protocol."""

    def __init__(
        self,
        *,
        azure_endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = REALTIME_API_VERSION,
        language: str = "en",
        prompt: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[Any] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            model=deployment,
            language=language,
            prompt=prompt,
            turn_detection=turn_detection,
            noise_reduction_type=noise_reduction_type,
            use_realtime=True,
            client=AsyncAzureOpenAI(
                max_retries=0,
                azure_endpoint=azure_endpoint,
                azure_deployment=deployment,
                api_version=api_version,
                api_key=api_key,
            ),
        )
        self._azure_endpoint = azure_endpoint.rstrip("/")
        self._azure_api_key = api_key
        self._azure_deployment = deployment
        self._azure_api_version = api_version

    def _realtime_url(self) -> str:
        query = urlencode(
            {
                "api-version": self._azure_api_version,
                "intent": "transcription",
                "deployment": self._azure_deployment,
            }
        )
        base = self._azure_endpoint
        if base.startswith("https"):
            base = base.replace("https", "wss", 1)
        elif base.startswith("http"):
            base = base.replace("http", "ws", 1)
        return f"{base}/openai/realtime?{query}"

    def _session_update(self) -> dict[str, Any]:
        transcription: dict[str, Any] = {"model": self._azure_deployment}
        if is_given(self._opts.prompt) and self._opts.prompt:
            transcription["prompt"] = self._opts.prompt
        if self._opts.language:
            transcription["language"] = self._opts.language.language

        session: dict[str, Any] = {
            "input_audio_format": "pcm16",
            "input_audio_transcription": transcription,
            "turn_detection": self._opts.turn_detection,
        }
        if (
            is_given(self._opts.noise_reduction_type)
            and self._opts.noise_reduction_type
        ):
            session["input_audio_noise_reduction"] = {
                "type": self._opts.noise_reduction_type
            }

        return {"type": "transcription_session.update", "session": session}

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        ws = await asyncio.wait_for(
            session.ws_connect(
                self._realtime_url(),
                headers={
                    "User-Agent": "LiveKit Agents",
                    "api-key": self._azure_api_key,
                },
            ),
            timeout,
        )
        await ws.send_json(self._session_update())
        return ws
