"""Integration tests verifying the Out-of-Process Chaos Reverse Proxy Sidecar.

Proves that pytest-agent-chaos can operate as a standalone L7 network sidecar
to intercept real deployed containers, FastAPI microservices, and LangGraph agents
over live TCP sockets without in-process Python memory monkeypatching.
"""

import json
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from pytest_agent_chaos.sidecar.client import RemoteChaosClient
from pytest_agent_chaos.sidecar.runner import ChaosSidecarServer, find_free_port

# ==============================================================================
# Helper Mock Upstream LLM Server
# ==============================================================================


class MockUpstreamLLM:
    """Runs a lightweight local HTTP server simulating api.openai.com."""

    def __init__(self, port: int | None = None) -> None:
        self.port = port or find_free_port()
        self.app = FastAPI()
        self.call_count = 0

        @self.app.post("/v1/chat/completions")
        async def chat_completions(req: dict[str, Any]) -> Response:
            self.call_count += 1
            if req.get("stream"):

                async def sse_gen():
                    yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
                    yield b'data: {"choices": [{"delta": {"content": " from deployed container"}}]}\n\n'
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(sse_gen(), media_type="text/event-stream")

            return Response(
                status_code=200,
                headers={"Content-Type": "application/json"},
                content=json.dumps(
                    {
                        "id": f"chatcmpl-{self.call_count}",
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "Autonomous deployed agent reached synthesis through proxy.",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ).encode(),
            )

        self.config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error")
        self.server = uvicorn.Server(self.config)
        import threading

        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        import time

        start_time = time.time()
        while time.time() - start_time < 3.0:
            try:
                with httpx.Client() as c:
                    if c.get(f"http://127.0.0.1:{self.port}/docs").status_code == 200:
                        return
            except (httpx.HTTPError, OSError):
                time.sleep(0.05)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=2.0)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_upstream():
    upstream = MockUpstreamLLM()
    upstream.start()
    yield upstream
    upstream.stop()


@pytest.fixture
def proxy_sidecar(mock_upstream):
    """Spins up a ChaosSidecarServer forwarding to the mock upstream."""
    server = ChaosSidecarServer(upstream_host=f"http://127.0.0.1:{mock_upstream.port}")
    client = server.start()
    yield client
    server.stop()


# ==============================================================================
# Tests
# ==============================================================================


def test_sidecar_control_plane_health_and_rules(proxy_sidecar: RemoteChaosClient) -> None:
    """Verifies control plane endpoints: health, rule configuration, and telemetry query."""
    assert proxy_sidecar.is_healthy() is True

    # Configure a rule
    rule_res = proxy_sidecar.add_rule(fault_type="rate_limit_429", trigger_on_call=1, retry_after=2)
    assert rule_res["status"] == "rule_configured"

    # Query telemetry (initially zero)
    telem = proxy_sidecar.get_telemetry()
    assert telem["total_calls"] == 0

    # Clear rules
    clear_res = proxy_sidecar.clear_rules()
    assert clear_res["status"] == "rules_cleared"


def test_sidecar_out_of_process_429_injection_and_recovery(
    proxy_sidecar: RemoteChaosClient, mock_upstream: MockUpstreamLLM
) -> None:
    """Sends real network requests to the proxy sidecar, verifying out-of-process 429 injection."""
    proxy_sidecar.add_rule(fault_type="rate_limit_429", trigger_on_call=1, retry_after=3)

    with httpx.Client(base_url=proxy_sidecar.base_url) as client:
        # Call 1: Intercepted by proxy, returns HTTP 429 with Retry-After header
        resp1 = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert resp1.status_code == 429
        assert resp1.headers.get("Retry-After") == "3"

        # Call 2: Retry passes through cleanly to the upstream service
        resp2 = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert resp2.status_code == 200
        assert "Autonomous deployed agent" in resp2.json()["choices"][0]["message"]["content"]

    # Verify sidecar metrics across network boundaries
    telemetry = proxy_sidecar.get_telemetry()
    assert telemetry["total_calls"] == 2
    assert telemetry["faults_injected"] == 1
    assert telemetry["recovery_succeeded"] is True


def test_sidecar_out_of_process_504_gateway_timeout(proxy_sidecar: RemoteChaosClient) -> None:
    """Verifies that the proxy returns HTTP 504 on programmed triggers."""
    proxy_sidecar.add_rule(fault_type="gateway_timeout_504", trigger_on_call=1)

    with httpx.Client(base_url=proxy_sidecar.base_url) as client:
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 504
        assert "Gateway Timeout" in resp.json()["error"]["message"]


def test_sidecar_mid_stream_sse_socket_sever(
    proxy_sidecar: RemoteChaosClient, mock_upstream: MockUpstreamLLM
) -> None:
    """Verifies that the sidecar proxy slices mid-stream SSE tokens and terminates the TCP socket."""
    proxy_sidecar.add_rule(
        fault_type="sse_midstream_drop",
        trigger_on_call=1,
        drop_after_chunks=1,
        drop_mid_chunk=True,
    )

    with httpx.Client(base_url=proxy_sidecar.base_url) as client:
        # Turn 1: Stream severed mid-flight over live TCP socket. The connection is
        # aborted (not cleanly ended), so the client sees a genuine transport error.
        severed_content = b""
        with (
            pytest.raises(httpx.RemoteProtocolError),
            client.stream("POST", "/v1/chat/completions", json={"stream": True}) as resp1,
        ):
            for chunk in resp1.iter_bytes():
                severed_content += chunk

        # Proves mid-token truncation occurred over network: contains "Hello" and severed '{"conte'
        assert b"Hello" in severed_content
        assert b'{"conte' in severed_content
        assert b"from deployed container" not in severed_content
        assert not severed_content.endswith(b"\n\n")

        # Turn 2: Reconnection streams completely
        with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as resp2:
            clean_content = b"".join(resp2.iter_bytes())

        assert b"from deployed container" in clean_content
        assert clean_content.endswith(b"[DONE]\n\n")

    telemetry = proxy_sidecar.get_telemetry()
    assert telemetry["faults_injected"] == 1
    assert telemetry["recovery_succeeded"] is True


def test_deployed_langgraph_agent_over_chaos_proxy(
    proxy_sidecar: RemoteChaosClient, mock_upstream: MockUpstreamLLM
) -> None:
    """Runs a genuine LangGraph agent with its OPENAI_BASE_URL pointed directly

    at the out-of-process Chaos Proxy Sidecar, proving end-to-end container compatibility.
    """
    # Program Call 1 to return 429
    proxy_sidecar.add_rule(fault_type="rate_limit_429", trigger_on_call=1, retry_after=1)

    # Point real LangChain ChatOpenAI at the external sidecar proxy!
    # (No in-process monkeypatching; all communication goes over localhost HTTP socket)
    llm = ChatOpenAI(
        model="gpt-4o",
        base_url=f"{proxy_sidecar.base_url}/v1",
        api_key="mock-key-for-proxy",
        max_retries=2,
    )
    agent = create_react_agent(llm, tools=[])

    # Invoke the real agent across the external sidecar
    result = agent.invoke({"messages": [("user", "Hello deployed container agent")]})

    assert (
        "Autonomous deployed agent reached synthesis through proxy."
        in result["messages"][-1].content
    )

    # Assert telemetry recorded across the network boundary
    telemetry = proxy_sidecar.get_telemetry()
    assert telemetry["total_calls"] == 2
    assert telemetry["faults_injected"] == 1
    assert telemetry["recovery_succeeded"] is True


def test_sidecar_sse_drop_that_never_fires_is_not_recorded_as_a_fault(
    proxy_sidecar: RemoteChaosClient,
) -> None:
    """A configured SSE-drop rule that never actually triggers must not look faulted.

    drop_after_chunks is higher than the number of chunks the mock upstream sends
    (3), so the drop condition inside the sidecar's streaming generator is never
    satisfied -- faults_injected must stay 0, and the next call must not be
    recorded as a "recovery" from a fault that never happened.
    """
    proxy_sidecar.add_rule(fault_type="sse_midstream_drop", trigger_on_call=1, drop_after_chunks=10)

    with httpx.Client(base_url=proxy_sidecar.base_url) as client:
        with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as resp1:
            full1 = b"".join(resp1.iter_bytes())
        assert full1.endswith(b"[DONE]\n\n")

        with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as resp2:
            b"".join(resp2.iter_bytes())

    telemetry = proxy_sidecar.get_telemetry()
    assert telemetry["faults_injected"] == 0
    assert telemetry["retries_observed"] == 0
    assert telemetry["recovery_succeeded"] is False
