"""Tests attacking a live deployed FastAPI container through the Chaos Proxy.

Validates that an out-of-process autonomous agent service recovers gracefully
when the L7 chaos proxy sidecar abruptly severs TCP transport streams mid-token.
"""

import os
import socket
import threading
import time
from collections.abc import Generator
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from agent_app.app import app as agent_app
from pytest_agent_chaos.proxy.client import ChaosProxyClient


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class BackgroundServer:
    def __init__(self, app: Any, port: int) -> None:
        self.app = app
        self.port = port
        self.config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        start_time = time.time()
        while time.time() - start_time < 3.0:
            try:
                with httpx.Client() as c:
                    if c.get(f"http://127.0.0.1:{self.port}/health").status_code == 200:
                        return
            except (httpx.HTTPError, OSError):
                time.sleep(0.05)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=2.0)


@pytest.fixture(scope="module")
def mock_upstream_llm() -> Generator[str, None, None]:
    """Lightweight upstream server simulating OpenAI SSE endpoint."""
    port = _find_free_port()
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(req: dict[str, Any]) -> Response:
        async def sse_gen():
            # Send chunk 1 with payload large enough to exceed 100 bytes
            yield (
                b'data: {"choices": [{"delta": {"content": "Synthesizing '
                b"the Q3 compliance audit ledger section 104... detailed records "
                b'verification pending ledger validation"}}]}\n\n'
            )
            # Send chunk 2
            yield b'data: {"choices": [{"delta": {"content": " completed successfully."}}]}\n\n'
            # Send completion token
            yield b"data: [DONE]\n\n"

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    srv = BackgroundServer(app, port)
    srv.start()
    yield f"http://127.0.0.1:{port}"
    srv.stop()


@pytest.fixture
def running_deployed_agent(
    chaos_proxy: ChaosProxyClient, mock_upstream_llm: str
) -> Generator[str, None, None]:
    """Ensures a deployed agent service is running on port 8000, configured to route through chaos proxy."""
    # Configure the proxy to forward upstream requests to our mock upstream LLM
    chaos_proxy.set_upstream(mock_upstream_llm)

    target_port = 8000
    server: BackgroundServer | None = None

    # Check if a live container is already running on port 8000
    is_live = False
    try:
        with httpx.Client(timeout=1.0) as c:
            if c.get(f"http://127.0.0.1:{target_port}/health").status_code == 200:
                is_live = True
    except (httpx.HTTPError, OSError):
        is_live = False

    old_openai_base = os.environ.get("OPENAI_BASE_URL")
    if not is_live:
        # Route agent upstream traffic to the chaos proxy
        os.environ["OPENAI_BASE_URL"] = chaos_proxy.control_url
        server = BackgroundServer(agent_app, target_port)
        server.start()

    try:
        yield f"http://127.0.0.1:{target_port}"
    finally:
        if server:
            server.stop()
        if old_openai_base is not None:
            os.environ["OPENAI_BASE_URL"] = old_openai_base
        else:
            os.environ.pop("OPENAI_BASE_URL", None)


def test_deployed_agent_recovers_from_severed_sse_socket(
    chaos_proxy: ChaosProxyClient,
    running_deployed_agent: str,
):
    """Verifies that an out-of-process deployed agent container handles mid-stream socket drops."""
    # 1. Command the sidecar proxy to drop the TCP stream mid-generation on call 1
    chaos_proxy.inject_mid_stream_drop(match_path="/chat/completions", drop_after_bytes=100)

    # 2. Send request to the live deployed FastAPI agent service on port 8000
    agent_api_url = f"{running_deployed_agent}/api/v1/run-agent"
    payload = {
        "session_id": "prod-session-9081",
        "prompt": "Synthesize the Q3 compliance audit ledger.",
    }

    response = httpx.post(agent_api_url, json=payload, timeout=30.0)

    # 3. Assertions on the deployed system
    assert response.status_code == 200
    data = response.json()

    # The deployed agent caught the socket disconnect, resumed from its checkpoint,
    # and finished generation on attempt 2 without failing the client HTTP call.
    assert data["status"] == "COMPLETED"
    assert "compliance audit" in data["output"].lower()

    # Verify proxy telemetry confirms 2 calls were intercepted (1 failed + 1 recovery)
    telemetry = chaos_proxy.get_telemetry()
    assert any(count >= 2 for count in telemetry["call_counts"].values())


def test_deployed_agent_recovers_from_proxy_429_rate_limit(
    chaos_proxy: ChaosProxyClient,
    running_deployed_agent: str,
):
    """Verifies that the deployed agent backs off and recovers when proxy returns 429."""
    chaos_proxy.inject_upstream_429(match_path="/chat/completions", trigger_on_call=1)

    agent_api_url = f"{running_deployed_agent}/api/v1/run-agent"
    payload = {
        "session_id": "prod-session-9082",
        "prompt": "Synthesize the Q3 compliance audit ledger.",
    }

    response = httpx.post(agent_api_url, json=payload, timeout=30.0)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert data["attempts"] == 2
