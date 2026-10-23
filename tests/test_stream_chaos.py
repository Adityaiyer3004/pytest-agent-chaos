"""Self-tests verifying HTTP transport interception, SSE midstream cutoffs, and retry backoffs."""

import asyncio
from typing import AsyncIterator, Iterator
import httpx
import pytest

from pytest_agent_chaos.exceptions import ChaosStreamDropError
from pytest_agent_chaos.interceptors.llm_stream import (
    ChaosAsyncByteStream,
    ChaosSyncByteStream,
    StreamChaosSandbox,
)
from pytest_agent_chaos.models import ChaosProfile, FaultType


def test_rate_limit_429_injection_and_recovery():
    """Verifies that 429 RateLimit with Retry-After header is injected on attempt #1 and recovered on attempt #2."""
    profile = ChaosProfile(
        fault_type=FaultType.RATE_LIMIT_429,
        trigger_on_call=1,
        retry_after_seconds=3,
        target_domains=["api.openai.com"],
    )

    def mock_backend_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "Successfully generated analysis."}}]},
        )

    transport = httpx.MockTransport(mock_backend_handler)

    with StreamChaosSandbox(profile) as sandbox:
        client = httpx.Client(transport=transport)
        request = client.build_request("POST", "https://api.openai.com/v1/chat/completions")

        # Attempt 1: Chaos intercepts and returns 429
        resp1 = client.send(request)
        assert resp1.status_code == 429
        assert resp1.headers.get("Retry-After") == "3"
        assert sandbox.telemetry.faults_injected == 1
        assert not sandbox.telemetry.recovery_succeeded

        # Simulated Agent Resilience Logic: agent catches 429, inspects Retry-After, and retries
        if resp1.status_code == 429:
            resp2 = client.send(request)
            assert resp2.status_code == 200
            assert resp2.json()["choices"][0]["message"]["content"] == "Successfully generated analysis."
            assert sandbox.telemetry.recovery_succeeded
            assert sandbox.telemetry.total_retries_observed == 1
            assert sandbox.telemetry.total_calls_intercepted == 2


def test_gateway_timeout_504_injection():
    """Verifies gateway 504 timeout injection on LLM requests."""
    profile = ChaosProfile(
        fault_type=FaultType.GATEWAY_TIMEOUT_504,
        trigger_on_call=1,
        target_domains=["api.anthropic.com"],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"text": "Anthropic Claude response"}]})

    with StreamChaosSandbox(profile) as sandbox:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        req = client.build_request("POST", "https://api.anthropic.com/v1/messages")

        resp = client.send(req)
        assert resp.status_code == 504
        assert sandbox.telemetry.faults_injected == 1


def test_connection_timeout_injection():
    """Verifies that httpx.ConnectTimeout is thrown deterministically."""
    profile = ChaosProfile(
        fault_type=FaultType.CONNECTION_TIMEOUT,
        trigger_on_call=1,
        target_domains=["generativelanguage.googleapis.com"],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": "Gemini response"}]})

    with StreamChaosSandbox(profile) as sandbox:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        req = client.build_request("POST", "https://generativelanguage.googleapis.com/v1beta/models")

        with pytest.raises(httpx.ConnectTimeout):
            client.send(req)

        assert sandbox.telemetry.faults_injected == 1


def test_sse_midstream_drop_sync():
    """Verifies that synchronous SSE stream is truncated mid-stream and triggers client recovery."""
    profile = ChaosProfile(
        fault_type=FaultType.SSE_MIDSTREAM_DROP,
        trigger_on_call=1,
        drop_after_chunks=2,
        target_domains=["api.openai.com"],
    )

    chunks_data = [
        b'data: {"token": "The"}\n\n',
        b'data: {"token": " agent"}\n\n',
        b'data: {"token": " recovered"}\n\n',
        b"data: [DONE]\n\n",
    ]

    def streaming_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=iter(chunks_data),
        )

    with StreamChaosSandbox(profile) as sandbox:
        client = httpx.Client(transport=httpx.MockTransport(streaming_handler))
        req = client.build_request("POST", "https://api.openai.com/v1/chat/completions")

        # Attempt 1: Should yield 2 chunks and then raise httpx.RemoteProtocolError
        resp1 = client.send(req)
        received_tokens = []
        with pytest.raises(httpx.RemoteProtocolError):
            for chunk in resp1.iter_bytes():
                received_tokens.append(chunk)

        assert len(received_tokens) == 2
        assert sandbox.telemetry.faults_injected == 1

        # Resilient agent fallback/retry logic (Attempt 2 passes cleanly)
        resp2 = client.send(req)
        recovered_tokens = list(resp2.iter_bytes())
        assert len(recovered_tokens) == 4
        assert sandbox.telemetry.recovery_succeeded
        assert sandbox.telemetry.total_retries_observed == 1


@pytest.mark.asyncio
async def test_sse_midstream_drop_async():
    """Verifies that asynchronous SSE stream triggers RemoteProtocolError mid-stream."""
    profile = ChaosProfile(
        fault_type=FaultType.SSE_MIDSTREAM_DROP,
        trigger_on_call=1,
        drop_after_chunks=1,
        target_domains=["api.anthropic.com"],
    )

    async def async_generator() -> AsyncIterator[bytes]:
        yield b'data: {"text": "Alpha"}\n\n'
        yield b'data: {"text": "Beta"}\n\n'
        yield b"data: [DONE]\n\n"

    def async_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=async_generator(),
        )

    with StreamChaosSandbox(profile) as sandbox:
        async with httpx.AsyncClient(transport=httpx.MockTransport(async_handler)) as client:
            req = client.build_request("POST", "https://api.anthropic.com/v1/messages")

            resp1 = await client.send(req)
            chunks = []
            with pytest.raises(httpx.RemoteProtocolError):
                async for chunk in resp1.aiter_bytes():
                    chunks.append(chunk)

            assert len(chunks) == 1
            assert sandbox.telemetry.faults_injected == 1

            # Retry
            resp2 = await client.send(req)
            success_chunks = []
            async for chunk in resp2.aiter_bytes():
                success_chunks.append(chunk)

            assert len(success_chunks) == 3
            assert sandbox.telemetry.recovery_succeeded
