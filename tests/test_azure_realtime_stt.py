"""Guards on the Azure realtime STT handshake and protocol.

Each assertion here corresponds to a way the stock plugin's request is rejected by Azure
(verified against a live endpoint): a 404 from the deployment-scoped path, a 404 from a
missing/too-old api-version, and `Unknown parameter: 'session.type'` from the GA-shaped
session payload. These are cheap offline checks — they build the URL and payload without
opening a socket, so a regression shows up here instead of mid-interview.
"""

from urllib.parse import parse_qs, urlparse

import pytest

from azure_realtime_stt import REALTIME_API_VERSION, AzureRealtimeSTT


@pytest.fixture
def stt() -> AzureRealtimeSTT:
    return AzureRealtimeSTT(
        azure_endpoint="https://example-resource.openai.azure.com/",
        api_key="test-key",
        deployment="gpt-4o-transcribe",
    )


def test_url_is_websocket_scheme(stt: AzureRealtimeSTT) -> None:
    assert urlparse(stt._realtime_url()).scheme == "wss"


def test_url_path_is_not_deployment_scoped(stt: AzureRealtimeSTT) -> None:
    """Azure serves realtime from a flat /openai/realtime; the deployment-scoped path 404s."""
    path = urlparse(stt._realtime_url()).path
    assert path == "/openai/realtime"
    assert "/deployments/" not in path


def test_url_carries_api_version_intent_and_deployment(stt: AzureRealtimeSTT) -> None:
    """Omitting api-version 404s; the deployment moves into the query string."""
    query = parse_qs(urlparse(stt._realtime_url()).query)
    assert query["api-version"] == [REALTIME_API_VERSION]
    assert query["intent"] == ["transcription"]
    assert query["deployment"] == ["gpt-4o-transcribe"]


def test_api_version_is_realtime_capable(stt: AzureRealtimeSTT) -> None:
    """2025-03-01-preview (the batch REST version) 404s on the websocket."""
    assert REALTIME_API_VERSION >= "2025-04-01-preview"


def test_session_update_uses_azure_preview_event(stt: AzureRealtimeSTT) -> None:
    """The GA `session.update` shape is rejected with Unknown parameter: 'session.type'."""
    payload = stt._session_update()
    assert payload["type"] == "transcription_session.update"
    assert "type" not in payload["session"]


def test_session_update_configures_transcription(stt: AzureRealtimeSTT) -> None:
    session = stt._session_update()["session"]
    assert session["input_audio_format"] == "pcm16"
    assert session["input_audio_transcription"]["model"] == "gpt-4o-transcribe"
    assert session["input_audio_transcription"]["language"] == "en"
    assert session["turn_detection"]["type"] == "server_vad"


def test_streaming_capability_is_advertised(stt: AzureRealtimeSTT) -> None:
    """The session must route audio to the websocket rather than the batch REST path."""
    assert stt.capabilities.streaming


def test_optional_fields_are_omitted_when_unset(stt: AzureRealtimeSTT) -> None:
    session = stt._session_update()["session"]
    assert "prompt" not in session["input_audio_transcription"]
    assert "input_audio_noise_reduction" not in session


def test_prompt_and_noise_reduction_are_forwarded_when_set() -> None:
    configured = AzureRealtimeSTT(
        azure_endpoint="https://example-resource.openai.azure.com/",
        api_key="test-key",
        deployment="gpt-4o-transcribe",
        prompt="Respond in English.",
        noise_reduction_type="near_field",
    )
    session = configured._session_update()["session"]
    assert session["input_audio_transcription"]["prompt"] == "Respond in English."
    assert session["input_audio_noise_reduction"] == {"type": "near_field"}
