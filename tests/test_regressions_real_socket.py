"""Regression tests for review bugs 1-4, run against a real local HTTP socket.

MockTransport responses already hold their body in memory, which is how bug 1
slipped through. These tests use a real ThreadingHTTPServer instead.
"""

import json
import random
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from pytest_agent_chaos.exceptions import InfiniteAgentLoopError
from pytest_agent_chaos.interceptors.llm_stream import StreamChaosSandbox
from pytest_agent_chaos.interceptors.state_graph import StateGraphAuditor
from pytest_agent_chaos.interceptors.tool_corruptor import ChaosSandbox
from pytest_agent_chaos.models import ChaosProfile, FaultType
from pytest_agent_chaos.plugin import _derive_seed, chaos_inject, stream_chaos_inject

SSE_EVENTS = [
    b'data: {"token": "alpha"}\n\n',
    b'data: {"token": "beta"}\n\n',
    b"data: [DONE]\n\n",
]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/fail":
            self._json(500, {"error": "upstream broken"})
        else:
            self._json(200, {"ok": True, "path": self.path})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path != "/sse":
            self._json(200, {"ok": True})
            return
        body_len = sum(len(e) for e in SSE_EVENTS)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(body_len))
        self.end_headers()
        for event in SSE_EVENTS:
            self.wfile.write(event)
            self.wfile.flush()
            time.sleep(0.05)  # make each event arrive as its own network read

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def live_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _sse_profile(trigger_on_call: int) -> ChaosProfile:
    return ChaosProfile(
        fault_type=FaultType.SSE_MIDSTREAM_DROP,
        trigger_on_call=trigger_on_call,
        drop_after_chunks=1,
        intercept_all_hosts=True,
    )


# --- Bug 1: SSE profile must not force streaming on calls where no fault fires ---


def test_sse_profile_leaves_unfaulted_calls_readable(live_url: str) -> None:
    with StreamChaosSandbox(_sse_profile(trigger_on_call=5)), httpx.Client() as client:
        assert client.get(f"{live_url}/plain").json() == {"ok": True, "path": "/plain"}


@pytest.mark.asyncio
async def test_sse_profile_leaves_unfaulted_async_calls_readable(live_url: str) -> None:
    with StreamChaosSandbox(_sse_profile(trigger_on_call=5)):
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{live_url}/plain")
            assert resp.json()["ok"] is True


def test_sse_drop_on_non_streaming_call_raises_from_send(live_url: str) -> None:
    with StreamChaosSandbox(_sse_profile(trigger_on_call=1)) as sb, httpx.Client() as client:
        with pytest.raises(httpx.RemoteProtocolError):
            client.post(f"{live_url}/sse", json={})
        assert sb.telemetry.faults_injected == 1


def test_sse_drop_on_streaming_call_truncates_real_socket(live_url: str) -> None:
    with StreamChaosSandbox(_sse_profile(trigger_on_call=1)) as sb, httpx.Client() as client:
        received = b""
        with pytest.raises(httpx.RemoteProtocolError):  # noqa: SIM117
            with client.stream("POST", f"{live_url}/sse", json={}) as resp:
                for chunk in resp.iter_bytes():
                    received += chunk
        assert b"[DONE]" not in received
        assert sb.telemetry.faults_injected == 1

        with client.stream("POST", f"{live_url}/sse", json={}) as resp:
            full = resp.read()
        assert full.endswith(b"[DONE]\n\n")
        assert sb.telemetry.recovery_succeeded


# --- Bug 2: decorators must keep the sandbox active for async tests ---

_LIVE_URL_FOR_DECORATOR: dict[str, str] = {}


@pytest.fixture
def _remember_url(live_url: str) -> None:
    _LIVE_URL_FOR_DECORATOR["url"] = live_url


@pytest.mark.usefixtures("_remember_url")
@stream_chaos_inject(fault=FaultType.RATE_LIMIT_429, trigger_on_call=1, intercept_all_hosts=True)
async def test_stream_decorator_applies_to_async_test(
    stream_sandbox: StreamChaosSandbox,
) -> None:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{_LIVE_URL_FOR_DECORATOR['url']}/plain")
    assert resp.status_code == 429
    assert stream_sandbox.telemetry.faults_injected == 1


@chaos_inject(fault=FaultType.TRUNCATED_JSON_TOOL_CALL, trigger_on_call=1)
async def test_tool_decorator_applies_to_async_test(chaos_sandbox: ChaosSandbox) -> None:
    assert isinstance(chaos_sandbox, ChaosSandbox)
    assert chaos_sandbox.mutate_tool_payload('{"q": "x"}') != '{"q": "x"}'


# --- Bug 3: recovery requires evidence, not just "any later call" ---


def _rate_limit_profile() -> ChaosProfile:
    return ChaosProfile(
        fault_type=FaultType.RATE_LIMIT_429, trigger_on_call=1, intercept_all_hosts=True
    )


def test_unrelated_call_after_fault_is_not_recovery(live_url: str) -> None:
    with StreamChaosSandbox(_rate_limit_profile()) as sb, httpx.Client() as client:
        assert client.get(f"{live_url}/chat").status_code == 429
        assert client.get(f"{live_url}/other").status_code == 200
        assert not sb.telemetry.recovery_succeeded


def test_failed_retry_of_same_endpoint_is_not_recovery(live_url: str) -> None:
    with StreamChaosSandbox(_rate_limit_profile()) as sb, httpx.Client() as client:
        assert client.get(f"{live_url}/fail").status_code == 429
        assert client.get(f"{live_url}/fail").status_code == 500
        assert sb.telemetry.total_retries_observed == 1
        assert not sb.telemetry.recovery_succeeded


def test_successful_retry_of_same_endpoint_is_recovery(live_url: str) -> None:
    with StreamChaosSandbox(_rate_limit_profile()) as sb, httpx.Client() as client:
        assert client.get(f"{live_url}/chat").status_code == 429
        assert client.get(f"{live_url}/chat").status_code == 200
        assert sb.telemetry.recovery_succeeded


def test_tool_call_with_different_shape_is_not_recovery() -> None:
    sb = ChaosSandbox(
        ChaosProfile(fault_type=FaultType.TRUNCATED_JSON_TOOL_CALL, trigger_on_call=1)
    )
    sb.mutate_tool_payload('{"invoice_id": "A", "amount": 1.0}')
    sb.mutate_tool_payload('{"unrelated": true}')
    assert not sb.telemetry.recovery_succeeded
    sb.mutate_tool_payload('{"amount": 1.0, "invoice_id": "A"}')
    assert sb.telemetry.recovery_succeeded


# --- Bug 4: booleans must drift to a non-bool, non-number-string value ---


def test_schema_drift_mutates_booleans_distinctly() -> None:
    sb = ChaosSandbox(ChaosProfile(fault_type=FaultType.SCHEMA_TYPE_DRIFT, trigger_on_call=1))
    out = json.loads(sb.mutate_tool_payload('{"flag": true, "n": 3, "nested": {"b": false}}'))
    assert out["flag"] == ["unexpected", "array"]
    assert out["nested"]["b"] == ["unexpected", "array"]
    assert out["n"] == "CHAOS_INVALID_NUMBER_TYPE"


# --- Bug 5: probability draws must not depend on the shared global `random` state ---


def _probability_draw(seed: int) -> float:
    sb = ChaosSandbox(
        ChaosProfile(
            fault_type=FaultType.TRUNCATED_JSON_TOOL_CALL,
            trigger_on_call=1,
            probability=0.5,
            seed=seed,
        )
    )
    return sb._rng.random()


def test_tool_sandbox_probability_draw_is_isolated_from_global_random() -> None:
    first = _probability_draw(seed=123)

    # Simulate unrelated chaos tests / sandboxes consuming the global random stream
    # before this one runs. With the old global-random implementation this would
    # change the result depending on test execution order.
    for _ in range(1000):
        random.random()

    assert _probability_draw(seed=123) == first


def test_derive_seed_is_stable_and_name_sensitive() -> None:
    assert _derive_seed("test_a") == _derive_seed("test_a")
    assert _derive_seed("test_a") != _derive_seed("test_b")


# --- Bug 6: cycle detection was hardcoded to period-2 and never raised ---


def test_cycle_detection_catches_three_node_ping_pong() -> None:
    auditor = StateGraphAuditor(max_steps=20, detect_cycles=True)
    for _ in range(3):
        auditor.record_step("planner")
        auditor.record_step("tool_executor")
        auditor.record_step("synthesizer")
    assert auditor.telemetry.cycle_detected is True


def test_cycle_detection_can_raise_instead_of_only_flagging() -> None:
    auditor = StateGraphAuditor(max_steps=20, detect_cycles=True, raise_on_cycle=True)
    with pytest.raises(InfiniteAgentLoopError):
        for _ in range(10):
            auditor.record_step("agent_a")
            auditor.record_step("agent_b")


# --- Bug 7: exiting multiple StreamChaosSandboxes out of LIFO order used to
# strand the httpx.Client.send monkeypatch permanently (never restored to real_send) ---


def _rl_profile() -> ChaosProfile:
    return ChaosProfile(fault_type=FaultType.RATE_LIMIT_429, intercept_all_hosts=True)


def test_lifo_sandbox_exit_restores_real_send() -> None:
    real_send = httpx.Client.send
    real_async_send = httpx.AsyncClient.send
    sb1 = StreamChaosSandbox(_rl_profile())
    sb2 = StreamChaosSandbox(_rl_profile())
    sb1.__enter__()
    sb2.__enter__()
    try:
        sb2.__exit__(None, None, None)
        sb1.__exit__(None, None, None)
        assert httpx.Client.send is real_send
        assert httpx.AsyncClient.send is real_async_send
    finally:
        # Safety net in case an assertion above fails mid-test.
        httpx.Client.send = real_send  # type: ignore[method-assign]
        httpx.AsyncClient.send = real_async_send  # type: ignore[method-assign]


def test_non_lifo_sandbox_exit_warns_instead_of_corrupting() -> None:
    real_send = httpx.Client.send
    real_async_send = httpx.AsyncClient.send
    sb1 = StreamChaosSandbox(_rl_profile())
    sb2 = StreamChaosSandbox(_rl_profile())
    sb1.__enter__()
    sb2.__enter__()
    try:
        with pytest.warns(UserWarning, match="LIFO order"):
            sb1.__exit__(None, None, None)  # out of order: sb2 is still active
        # sb2's patch must survive sb1's misordered exit, not be silently clobbered.
        assert httpx.Client.send is sb2._patched_sync_send
        sb2.__exit__(None, None, None)
    finally:
        # Out-of-order exits can't fully recover the true original (sb1's stale
        # patch is what remains active) -- that's the whole point of the warning.
        # Force-restore here so this test doesn't leak the patch into later tests.
        httpx.Client.send = real_send  # type: ignore[method-assign]
        httpx.AsyncClient.send = real_async_send  # type: ignore[method-assign]
