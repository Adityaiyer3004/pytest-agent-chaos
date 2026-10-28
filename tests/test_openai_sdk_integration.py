"""Integration tests against the official openai Python SDK, with its default HTTP client.

No custom http_client is passed: openai<3 sends through httpx, openai>=3 through
httpx2, and StreamChaosSandbox must intercept whichever one is installed. Before
0.1.3 it only patched httpx, so on openai>=3 every fault here silently never fired.

The upstream is a real local socket, not MockTransport, so SSE drops cut a body
that is actually still streaming.
"""

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pytest_agent_chaos.interceptors.llm_stream import StreamChaosSandbox, _transports
from pytest_agent_chaos.models import ChaosProfile, FaultType

openai = pytest.importorskip("openai")

# The injected drop raises the transport library's own RemoteProtocolError.
# openai>=3 re-raises it mid-stream as APIConnectionError (with it as __cause__);
# older SDKs let it propagate as-is.
DROP_ERRORS = tuple(t.lib.RemoteProtocolError for t in _transports())
STREAM_DROP_ERRORS = (openai.APIConnectionError, *DROP_ERRORS)


def _assert_injected_drop(excinfo: pytest.ExceptionInfo[BaseException]) -> None:
    err = excinfo.value
    assert isinstance(err if isinstance(err, DROP_ERRORS) else err.__cause__, DROP_ERRORS)


COMPLETION = {
    "id": "chatcmpl-chaos",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "recovered"},
            "finish_reason": "stop",
        }
    ],
}


def _chunk(content: str) -> bytes:
    chunk = {
        "id": "chatcmpl-chaos",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


SSE_EVENTS = [_chunk("Hel"), _chunk("lo"), b"data: [DONE]\n\n"]


class _OpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not body.get("stream"):
            payload = json.dumps(COMPLETION).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(sum(len(e) for e in SSE_EVENTS)))
        self.end_headers()
        for event in SSE_EVENTS:
            self.wfile.write(event)
            self.wfile.flush()
            time.sleep(0.05)  # make each event arrive as its own network read

    def log_message(self, *args: object) -> None:
        pass


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request: object, client_address: object) -> None:
        pass  # the client severing a dropped stream mid-body is the point, not an error


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    server = _QuietServer(("127.0.0.1", 0), _OpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def _profile(fault: FaultType, **overrides: object) -> ChaosProfile:
    # retry_after_seconds=0 makes the SDK fall back to its short (~0.5s) backoff.
    settings: dict[str, object] = {
        "fault_type": fault,
        "trigger_on_call": 1,
        "intercept_all_hosts": True,
        "retry_after_seconds": 0,
        "drop_after_chunks": 1,
        **overrides,
    }
    return ChaosProfile.model_validate(settings)


def _ask(client: "openai.OpenAI") -> str:
    resp = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
    )
    return str(resp.choices[0].message.content)


@pytest.mark.parametrize(
    "fault",
    [FaultType.RATE_LIMIT_429, FaultType.GATEWAY_TIMEOUT_504, FaultType.CONNECTION_TIMEOUT],
)
def test_sdk_built_in_retry_recovers_from_transport_fault(base_url: str, fault: FaultType) -> None:
    with StreamChaosSandbox(_profile(fault)) as sb:
        client = openai.OpenAI(api_key="test-key", base_url=base_url, max_retries=2)
        assert _ask(client) == "recovered"

    assert sb.telemetry.total_calls_intercepted == 2
    assert sb.telemetry.faults_injected == 1
    assert sb.telemetry.recovery_succeeded


def test_sdk_without_retries_surfaces_429_as_rate_limit_error(base_url: str) -> None:
    with StreamChaosSandbox(_profile(FaultType.RATE_LIMIT_429)) as sb:
        client = openai.OpenAI(api_key="test-key", base_url=base_url, max_retries=0)
        with pytest.raises(openai.RateLimitError):
            _ask(client)

    assert sb.telemetry.faults_injected == 1
    assert not sb.telemetry.recovery_succeeded


def test_sse_drop_on_non_streaming_call_is_retried_by_sdk(base_url: str) -> None:
    # The drop surfaces from send(), where the SDK treats it as a connection error.
    profile = _profile(FaultType.SSE_MIDSTREAM_DROP, drop_after_chunks=0)
    with StreamChaosSandbox(profile) as sb:
        client = openai.OpenAI(api_key="test-key", base_url=base_url, max_retries=2)
        assert _ask(client) == "recovered"

    assert sb.telemetry.faults_injected == 1
    assert sb.telemetry.recovery_succeeded


def test_sse_drop_mid_stream_reaches_caller_and_caller_retry_recovers(base_url: str) -> None:
    with StreamChaosSandbox(_profile(FaultType.SSE_MIDSTREAM_DROP)) as sb:
        client = openai.OpenAI(api_key="test-key", base_url=base_url, max_retries=2)

        # Mid-stream drops happen after the SDK has returned, so it can't retry them.
        received: list[str] = []
        stream = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        with pytest.raises(STREAM_DROP_ERRORS) as excinfo:
            for chunk in stream:
                received.append(chunk.choices[0].delta.content or "")
        _assert_injected_drop(excinfo)
        assert received == ["Hel"]
        assert not sb.telemetry.recovery_succeeded

        retry = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        assert "".join(c.choices[0].delta.content or "" for c in retry) == "Hello"

    assert sb.telemetry.faults_injected == 1
    assert sb.telemetry.recovery_succeeded


async def test_async_sdk_built_in_retry_recovers_from_429(base_url: str) -> None:
    with StreamChaosSandbox(_profile(FaultType.RATE_LIMIT_429)) as sb:
        client = openai.AsyncOpenAI(api_key="test-key", base_url=base_url, max_retries=2)
        resp = await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
        assert resp.choices[0].message.content == "recovered"
        await client.close()

    assert sb.telemetry.faults_injected == 1
    assert sb.telemetry.recovery_succeeded


async def test_async_sse_drop_mid_stream_reaches_caller(base_url: str) -> None:
    with StreamChaosSandbox(_profile(FaultType.SSE_MIDSTREAM_DROP)) as sb:
        client = openai.AsyncOpenAI(api_key="test-key", base_url=base_url, max_retries=2)
        stream = await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        with pytest.raises(STREAM_DROP_ERRORS) as excinfo:
            async for _ in stream:
                pass
        _assert_injected_drop(excinfo)
        await client.close()

    assert sb.telemetry.faults_injected == 1
