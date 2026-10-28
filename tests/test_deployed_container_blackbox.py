"""Tests attacking a live deployed FastAPI container through the Chaos Proxy.

Validates that an out-of-process autonomous agent service recovers gracefully
when the L7 chaos proxy sidecar abruptly severs TCP transport streams mid-token.
"""

import os
import socket
import threading
import time
from collections.abc import Callable, Generator
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from agent_app.app import app as agent_app
from pytest_agent_chaos.sidecar.client import RemoteChaosClient


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
    chaos_sidecar: Callable[..., RemoteChaosClient], mock_upstream_llm: str
) -> Generator[tuple[RemoteChaosClient, str], None, None]:
    """Ensures a deployed agent service is reachable, configured to route through the chaos sidecar.

    If a real container (e.g. via docker-compose) is already listening on the
    conventional port 8000, attaches to that instead of starting an in-process one --
    this is the manual/docker-compose path and is unaffected by the port choice below.
    Otherwise starts one in-process on a freshly allocated port, so that running this
    test file under pytest-xdist doesn't race multiple workers for the same fixed port.
    """
    sidecar_client = chaos_sidecar(upstream_host=mock_upstream_llm)

    docker_compose_port = 8000
    target_port = docker_compose_port
    server: BackgroundServer | None = None

    # Check if a live container is already running on the conventional port
    is_live = False
    try:
        with httpx.Client(timeout=1.0) as c:
            if c.get(f"http://127.0.0.1:{docker_compose_port}/health").status_code == 200:
                is_live = True
    except (httpx.HTTPError, OSError):
        is_live = False

    old_openai_base = os.environ.get("OPENAI_BASE_URL")
    if not is_live:
        # Route agent upstream traffic to the chaos sidecar
        os.environ["OPENAI_BASE_URL"] = sidecar_client.base_url
        target_port = _find_free_port()
        server = BackgroundServer(agent_app, target_port)
        server.start()

    try:
        yield sidecar_client, f"http://127.0.0.1:{target_port}"
    finally:
        if server:
            server.stop()
        if old_openai_base is not None:
            os.environ["OPENAI_BASE_URL"] = old_openai_base
        else:
            os.environ.pop("OPENAI_BASE_URL", None)


def test_deployed_agent_recovers_from_severed_sse_socket(
    running_deployed_agent: tuple[RemoteChaosClient, str],
):
    """Verifies that an out-of-process deployed agent container handles mid-stream socket drops."""
    sidecar_client, agent_base_url = running_deployed_agent

    # 1. Command the sidecar to drop the TCP stream mid-generation on call 1
    sidecar_client.add_rule(fault_type="sse_midstream_drop", trigger_on_call=1, drop_after_chunks=1)

    # 2. Send request to the live deployed FastAPI agent service on port 8000
    agent_api_url = f"{agent_base_url}/api/v1/run-agent"
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

    # Verify sidecar telemetry confirms 2 calls were intercepted (1 failed + 1 recovery)
    telemetry = sidecar_client.get_telemetry()
    assert telemetry["total_calls"] >= 2
    assert telemetry["recovery_succeeded"] is True


def test_deployed_agent_recovers_from_proxy_429_rate_limit(
    running_deployed_agent: tuple[RemoteChaosClient, str],
):
    """Verifies that the deployed agent backs off and recovers when the sidecar returns 429."""
    sidecar_client, agent_base_url = running_deployed_agent
    sidecar_client.add_rule(fault_type="rate_limit_429", trigger_on_call=1, retry_after=1)

    agent_api_url = f"{agent_base_url}/api/v1/run-agent"
    payload = {
        "session_id": "prod-session-9082",
        "prompt": "Synthesize the Q3 compliance audit ledger.",
    }

    response = httpx.post(agent_api_url, json=payload, timeout=30.0)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert data["attempts"] == 2
