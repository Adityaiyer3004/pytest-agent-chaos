"""Transport & Stream Interceptor for httpx and LLM providers.

Intercepts outbound calls to OpenAI, Anthropic, Vertex AI, or local LLM mocks
at the HTTP client level to simulate mid-stream SSE (Server-Sent Events) chunk drops,
429 Retry-After storms, gateway timeouts, and network latency jitter.
"""

import asyncio
import time
from typing import Any, AsyncIterator, Callable, Iterator, List, Optional
import httpx
from pytest_agent_chaos.exceptions import (
    ChaosStreamDropError,
    ChaosTimeoutError,
)
from pytest_agent_chaos.models import ChaosProfile, ChaosTelemetry, FaultType


class ChaosSyncByteStream(httpx.SyncByteStream):
    """Sync byte stream that injects chunk drop or latency midway through reading."""

    def __init__(
        self,
        original_stream: Iterator[bytes],
        drop_after_chunks: int = 1,
        should_drop: bool = False,
        latency_ms: int = 0,
        telemetry: Optional[ChaosTelemetry] = None,
    ) -> None:
        self._original_stream = original_stream
        self._drop_after_chunks = drop_after_chunks
        self._should_drop = should_drop
        self._latency_ms = latency_ms
        self._chunks_yielded = 0
        self._telemetry = telemetry

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._original_stream:
            if self._latency_ms > 0:
                time.sleep(self._latency_ms / 1000.0)

            self._chunks_yielded += 1
            if self._should_drop and self._chunks_yielded > self._drop_after_chunks:
                if self._telemetry:
                    self._telemetry.faults_injected += 1
                raise httpx.RemoteProtocolError(
                    "pytest-agent-chaos: Peer connection closed unexpectedly mid-stream."
                )

            yield chunk

    def close(self) -> None:
        if hasattr(self._original_stream, "close"):
            self._original_stream.close()


class ChaosAsyncByteStream(httpx.AsyncByteStream):
    """Async byte stream that injects chunk drop or latency midway through async reading."""

    def __init__(
        self,
        original_stream: AsyncIterator[bytes],
        drop_after_chunks: int = 1,
        should_drop: bool = False,
        latency_ms: int = 0,
        telemetry: Optional[ChaosTelemetry] = None,
    ) -> None:
        self._original_stream = original_stream
        self._drop_after_chunks = drop_after_chunks
        self._should_drop = should_drop
        self._latency_ms = latency_ms
        self._chunks_yielded = 0
        self._telemetry = telemetry

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._original_stream:
            if self._latency_ms > 0:
                await asyncio.sleep(self._latency_ms / 1000.0)

            self._chunks_yielded += 1
            if self._should_drop and self._chunks_yielded > self._drop_after_chunks:
                if self._telemetry:
                    self._telemetry.faults_injected += 1
                raise httpx.RemoteProtocolError(
                    "pytest-agent-chaos: Peer connection closed unexpectedly mid-stream."
                )

            yield chunk

    async def aclose(self) -> None:
        if hasattr(self._original_stream, "aclose"):
            await self._original_stream.aclose()


class StreamChaosSandbox:
    """Manages transport-level interception for httpx Client and AsyncClient."""

    def __init__(self, profile: ChaosProfile) -> None:
        self.profile = profile
        self.telemetry = ChaosTelemetry(fault_type=profile.fault_type)
        self._original_sync_send: Optional[Callable[..., Any]] = None
        self._original_async_send: Optional[Callable[..., Any]] = None
        self._is_active = False

    def is_target_host(self, host: str) -> bool:
        """Determines if the requested host should be intercepted."""
        if self.profile.intercept_all_hosts:
            return True
        return any(domain in host for domain in self.profile.target_domains)

    def _should_inject_fault(self) -> bool:
        """Evaluates trigger count and probability."""
        if self.telemetry.total_calls_intercepted == self.profile.trigger_on_call:
            return True
        return False

    def _generate_429_response(self, request: httpx.Request) -> httpx.Response:
        self.telemetry.faults_injected += 1
        headers = {
            "Content-Type": "application/json",
            "Retry-After": str(self.profile.retry_after_seconds),
            "x-ratelimit-reset-requests": str(self.profile.retry_after_seconds),
        }
        content = (
            b'{"error": {"message": "Rate limit exceeded. Quota exhausted.", '
            b'"type": "insufficient_quota", "code": "rate_limit_exceeded"}}'
        )
        return httpx.Response(
            status_code=429,
            headers=headers,
            content=content,
            request=request,
        )

    def _generate_504_response(self, request: httpx.Request) -> httpx.Response:
        self.telemetry.faults_injected += 1
        headers = {"Content-Type": "application/json"}
        content = b'{"error": {"message": "Gateway Timeout from upstream LLM provider.", "code": 504}}'
        return httpx.Response(
            status_code=504,
            headers=headers,
            content=content,
            request=request,
        )

    def intercept_sync_send(
        self,
        client: httpx.Client,
        request: httpx.Request,
        *args: Any,
        **kwargs: Any,
    ) -> httpx.Response:
        """Intercepts synchronous httpx.Client.send calls."""
        assert self._original_sync_send is not None

        if not self.is_target_host(request.url.host):
            return self._original_sync_send(client, request, *args, **kwargs)

        self.telemetry.total_calls_intercepted += 1

        send_kwargs = dict(kwargs)
        if self.profile.fault_type == FaultType.SSE_MIDSTREAM_DROP:
            send_kwargs["stream"] = True

        # Check if this call represents a recovery retry
        if self.telemetry.faults_injected > 0 and self.telemetry.total_calls_intercepted > self.profile.trigger_on_call:
            self.telemetry.total_retries_observed += 1
            self.telemetry.recovery_succeeded = True

        if not self._should_inject_fault():
            return self._original_sync_send(client, request, *args, **send_kwargs)

        # Inject Faults
        if self.profile.fault_type == FaultType.RATE_LIMIT_429:
            return self._generate_429_response(request)

        if self.profile.fault_type == FaultType.GATEWAY_TIMEOUT_504:
            return self._generate_504_response(request)

        if self.profile.fault_type == FaultType.CONNECTION_TIMEOUT:
            self.telemetry.faults_injected += 1
            raise httpx.ConnectTimeout(
                f"pytest-agent-chaos: Simulated connection timeout to {request.url.host}",
                request=request,
            )

        if self.profile.fault_type == FaultType.LATENCY_JITTER:
            if self.profile.latency_ms > 0:
                time.sleep(self.profile.latency_ms / 1000.0)

        # Execute downstream call
        response: httpx.Response = self._original_sync_send(client, request, *args, **send_kwargs)

        if self.profile.fault_type == FaultType.SSE_MIDSTREAM_DROP:
            # Wrap stream with ChaosSyncByteStream
            original_stream = response.stream
            response.stream = ChaosSyncByteStream(
                original_stream=original_stream,
                drop_after_chunks=self.profile.drop_after_chunks,
                should_drop=True,
                latency_ms=self.profile.latency_ms,
                telemetry=self.telemetry,
            )

        return response

    async def intercept_async_send(
        self,
        client: httpx.AsyncClient,
        request: httpx.Request,
        *args: Any,
        **kwargs: Any,
    ) -> httpx.Response:
        """Intercepts asynchronous httpx.AsyncClient.send calls."""
        assert self._original_async_send is not None

        if not self.is_target_host(request.url.host):
            return await self._original_async_send(client, request, *args, **kwargs)

        self.telemetry.total_calls_intercepted += 1

        send_kwargs = dict(kwargs)
        if self.profile.fault_type == FaultType.SSE_MIDSTREAM_DROP:
            send_kwargs["stream"] = True

        # Check recovery
        if self.telemetry.faults_injected > 0 and self.telemetry.total_calls_intercepted > self.profile.trigger_on_call:
            self.telemetry.total_retries_observed += 1
            self.telemetry.recovery_succeeded = True

        if not self._should_inject_fault():
            return await self._original_async_send(client, request, *args, **send_kwargs)

        if self.profile.fault_type == FaultType.RATE_LIMIT_429:
            return self._generate_429_response(request)

        if self.profile.fault_type == FaultType.GATEWAY_TIMEOUT_504:
            return self._generate_504_response(request)

        if self.profile.fault_type == FaultType.CONNECTION_TIMEOUT:
            self.telemetry.faults_injected += 1
            raise httpx.ConnectTimeout(
                f"pytest-agent-chaos: Simulated async connection timeout to {request.url.host}",
                request=request,
            )

        if self.profile.fault_type == FaultType.LATENCY_JITTER:
            if self.profile.latency_ms > 0:
                await asyncio.sleep(self.profile.latency_ms / 1000.0)

        response: httpx.Response = await self._original_async_send(client, request, *args, **send_kwargs)

        if self.profile.fault_type == FaultType.SSE_MIDSTREAM_DROP:
            original_stream = response.stream
            response.stream = ChaosAsyncByteStream(
                original_stream=original_stream,
                drop_after_chunks=self.profile.drop_after_chunks,
                should_drop=True,
                latency_ms=self.profile.latency_ms,
                telemetry=self.telemetry,
            )

        return response

    def __enter__(self) -> "StreamChaosSandbox":
        """Installs monkeypatches on httpx.Client.send and httpx.AsyncClient.send."""
        if not self._is_active:
            self._original_sync_send = httpx.Client.send
            self._original_async_send = httpx.AsyncClient.send

            sandbox = self

            def patched_sync_send(client_self: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
                return sandbox.intercept_sync_send(client_self, request, *args, **kwargs)

            async def patched_async_send(client_self: httpx.AsyncClient, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
                return await sandbox.intercept_async_send(client_self, request, *args, **kwargs)

            httpx.Client.send = patched_sync_send  # type: ignore[assignment]
            httpx.AsyncClient.send = patched_async_send  # type: ignore[assignment]
            self._is_active = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Restores original send methods on httpx.Client and httpx.AsyncClient."""
        if self._is_active:
            if self._original_sync_send:
                httpx.Client.send = self._original_sync_send  # type: ignore[assignment]
            if self._original_async_send:
                httpx.AsyncClient.send = self._original_async_send  # type: ignore[assignment]
            self._is_active = False
