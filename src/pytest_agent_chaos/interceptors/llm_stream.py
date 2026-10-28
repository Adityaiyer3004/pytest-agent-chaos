"""Transport & Stream Interceptor for httpx and LLM providers.

Intercepts outbound calls to OpenAI, Anthropic, Vertex AI, or local LLM mocks
at the HTTP client level to simulate mid-stream SSE (Server-Sent Events) chunk drops,
429 Retry-After storms, gateway timeouts, and network latency jitter.
"""

import asyncio
import random
import threading
import time
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, cast

import httpx

from pytest_agent_chaos.models import DEFAULT_SEED, ChaosProfile, ChaosTelemetry, FaultType

# Guards install/restore of the process-global httpx.Client.send / AsyncClient.send
# monkeypatch. Sandboxes must still be entered/exited in strict LIFO order (standard
# `with`-block nesting gives you this for free); this lock only prevents a torn
# read/write if two threads install or restore at the same instant.
_PATCH_LOCK = threading.Lock()


class ChaosSyncByteStream(httpx.SyncByteStream):
    """Sync byte stream that injects chunk drop or latency midway through reading."""

    def __init__(
        self,
        original_stream: Iterator[bytes],
        drop_after_chunks: int = 1,
        should_drop: bool = False,
        latency_ms: int = 0,
        telemetry: ChaosTelemetry | None = None,
        drop_mid_chunk: bool = False,
        on_drop: Callable[[], None] | None = None,
    ) -> None:
        self._original_stream = original_stream
        self._drop_after_chunks = drop_after_chunks
        self._should_drop = should_drop
        self._latency_ms = latency_ms
        self._chunks_yielded = 0
        self._telemetry = telemetry
        self._drop_mid_chunk = drop_mid_chunk
        self._on_drop = on_drop

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._original_stream:
            if self._latency_ms > 0:
                time.sleep(self._latency_ms / 1000.0)

            self._chunks_yielded += 1
            if self._should_drop and self._chunks_yielded > self._drop_after_chunks:
                if self._drop_mid_chunk and len(chunk) > 1:
                    half = max(1, len(chunk) // 2)
                    yield chunk[:half]
                if self._telemetry:
                    self._telemetry.faults_injected += 1
                if self._on_drop:
                    self._on_drop()
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
        telemetry: ChaosTelemetry | None = None,
        drop_mid_chunk: bool = False,
        on_drop: Callable[[], None] | None = None,
    ) -> None:
        self._original_stream = original_stream
        self._drop_after_chunks = drop_after_chunks
        self._should_drop = should_drop
        self._latency_ms = latency_ms
        self._chunks_yielded = 0
        self._telemetry = telemetry
        self._drop_mid_chunk = drop_mid_chunk
        self._on_drop = on_drop

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._original_stream:
            if self._latency_ms > 0:
                await asyncio.sleep(self._latency_ms / 1000.0)

            self._chunks_yielded += 1
            if self._should_drop and self._chunks_yielded > self._drop_after_chunks:
                if self._drop_mid_chunk and len(chunk) > 1:
                    half = max(1, len(chunk) // 2)
                    yield chunk[:half]
                if self._telemetry:
                    self._telemetry.faults_injected += 1
                if self._on_drop:
                    self._on_drop()
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
        # Private RNG stream, mirroring ChaosSandbox: probability draws must not
        # depend on what other sandboxes have already pulled from global random.
        # Falls back to DEFAULT_SEED (not OS entropy) if the profile wasn't seeded.
        self._rng = random.Random(profile.seed if profile.seed is not None else DEFAULT_SEED)
        self._original_sync_send: Callable[..., Any] | None = None
        self._original_async_send: Callable[..., Any] | None = None
        self._patched_sync_send: Callable[..., Any] | None = None
        self._patched_async_send: Callable[..., Any] | None = None
        self._is_active = False
        # Endpoints (method, host, path) that have received an injected fault.
        self._faulted_endpoints: set[tuple[str, str, str]] = set()

    @staticmethod
    def _endpoint(request: httpx.Request) -> tuple[str, str, str]:
        return (request.method, request.url.host, request.url.path)

    def _note_fault(self, request: httpx.Request) -> None:
        self._faulted_endpoints.add(self._endpoint(request))

    def _note_response(self, request: httpx.Request, response: httpx.Response) -> None:
        """Records recovery only when a previously faulted endpoint later returns 2xx.

        A call to an unrelated endpoint, or a non-2xx retry, is not evidence of recovery.
        """
        if self._endpoint(request) not in self._faulted_endpoints:
            return
        self.telemetry.total_retries_observed += 1
        if response.is_success:
            self.telemetry.recovery_succeeded = True

    def is_target_host(self, host: str) -> bool:
        """Determines if the requested host should be intercepted."""
        if self.profile.intercept_all_hosts:
            return True
        return any(domain in host for domain in self.profile.target_domains)

    def _get_active_fault(self) -> FaultType | None:
        """Determines if a fault should inject on the current call occurrence."""
        call_num = self.telemetry.total_calls_intercepted
        if self.profile.fault_schedule and call_num in self.profile.fault_schedule:
            candidate = self.profile.fault_schedule[call_num]
        elif call_num == self.profile.trigger_on_call:
            candidate = self.profile.fault_type
        else:
            return None
        if self._rng.random() <= self.profile.probability:
            return candidate
        return None

    def _generate_429_response(self, request: httpx.Request) -> httpx.Response:
        self.telemetry.faults_injected += 1
        self._note_fault(request)
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
        self._note_fault(request)
        headers = {"Content-Type": "application/json"}
        content = (
            b'{"error": {"message": "Gateway Timeout from upstream LLM provider.", "code": 504}}'
        )
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
            return cast(httpx.Response, self._original_sync_send(client, request, *args, **kwargs))

        self.telemetry.total_calls_intercepted += 1

        active_fault = self._get_active_fault()

        if active_fault is None:
            clean = cast(httpx.Response, self._original_sync_send(client, request, *args, **kwargs))
            self._note_response(request, clean)
            return clean

        # Inject Faults
        if active_fault == FaultType.RATE_LIMIT_429:
            return self._generate_429_response(request)

        if active_fault == FaultType.GATEWAY_TIMEOUT_504:
            return self._generate_504_response(request)

        if active_fault == FaultType.CONNECTION_TIMEOUT:
            self.telemetry.faults_injected += 1
            self._note_fault(request)
            raise httpx.ConnectTimeout(
                f"pytest-agent-chaos: Simulated connection timeout to {request.url.host}",
                request=request,
            )

        if active_fault == FaultType.LATENCY_JITTER and self.profile.latency_ms > 0:
            time.sleep(self.profile.latency_ms / 1000.0)
            self.telemetry.faults_injected += 1
            self._note_fault(request)

        if active_fault != FaultType.SSE_MIDSTREAM_DROP:
            return cast(httpx.Response, self._original_sync_send(client, request, *args, **kwargs))

        # SSE drop: only this call is forced to stream, so the body can be cut mid-read.
        caller_streams = bool(kwargs.get("stream", False))
        send_kwargs = {**kwargs, "stream": True}
        response = cast(
            httpx.Response, self._original_sync_send(client, request, *args, **send_kwargs)
        )
        # Only mark this endpoint as faulted if the drop actually fires -- if the
        # response has fewer chunks than drop_after_chunks, nothing is ever dropped,
        # and the endpoint must not look "faulted" to the next call's recovery check.
        response.stream = ChaosSyncByteStream(
            original_stream=cast(Iterator[bytes], response.stream),
            drop_after_chunks=self.profile.drop_after_chunks,
            should_drop=True,
            latency_ms=self.profile.latency_ms,
            telemetry=self.telemetry,
            drop_mid_chunk=self.profile.drop_mid_chunk,
            on_drop=lambda: self._note_fault(request),
        )
        if not caller_streams:
            # A non-streaming caller expects send() to return a fully read body.
            # Reading here makes the drop surface from send(), as on a real socket.
            try:
                response.read()
            except BaseException:
                response.close()
                raise
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
            return cast(
                httpx.Response, await self._original_async_send(client, request, *args, **kwargs)
            )

        self.telemetry.total_calls_intercepted += 1

        active_fault = self._get_active_fault()

        if active_fault is None:
            clean = cast(
                httpx.Response,
                await self._original_async_send(client, request, *args, **kwargs),
            )
            self._note_response(request, clean)
            return clean

        if active_fault == FaultType.RATE_LIMIT_429:
            return self._generate_429_response(request)

        if active_fault == FaultType.GATEWAY_TIMEOUT_504:
            return self._generate_504_response(request)

        if active_fault == FaultType.CONNECTION_TIMEOUT:
            self.telemetry.faults_injected += 1
            self._note_fault(request)
            raise httpx.ConnectTimeout(
                f"pytest-agent-chaos: Simulated async connection timeout to {request.url.host}",
                request=request,
            )

        if active_fault == FaultType.LATENCY_JITTER and self.profile.latency_ms > 0:
            await asyncio.sleep(self.profile.latency_ms / 1000.0)
            self.telemetry.faults_injected += 1
            self._note_fault(request)

        if active_fault != FaultType.SSE_MIDSTREAM_DROP:
            return cast(
                httpx.Response,
                await self._original_async_send(client, request, *args, **kwargs),
            )

        caller_streams = bool(kwargs.get("stream", False))
        send_kwargs = {**kwargs, "stream": True}
        response = cast(
            httpx.Response,
            await self._original_async_send(client, request, *args, **send_kwargs),
        )
        # See the sync path above: only mark faulted if the drop actually fires.
        response.stream = ChaosAsyncByteStream(
            original_stream=cast(AsyncIterator[bytes], response.stream),
            drop_after_chunks=self.profile.drop_after_chunks,
            should_drop=True,
            latency_ms=self.profile.latency_ms,
            telemetry=self.telemetry,
            drop_mid_chunk=self.profile.drop_mid_chunk,
            on_drop=lambda: self._note_fault(request),
        )
        if not caller_streams:
            try:
                await response.aread()
            except BaseException:
                await response.aclose()
                raise
        return response

    def __enter__(self) -> "StreamChaosSandbox":
        """Installs monkeypatches on httpx.Client.send and httpx.AsyncClient.send.

        Sandboxes nest like any other context manager: entering a second sandbox
        while the first is still active is fine, as long as they are exited in
        the reverse (LIFO) order they were entered — exactly what `with` blocks
        and pytest fixture teardown give you, provided teardown also unwinds LIFO.
        """
        with _PATCH_LOCK:
            if not self._is_active:
                self._original_sync_send = httpx.Client.send
                self._original_async_send = httpx.AsyncClient.send

                sandbox = self

                def patched_sync_send(
                    client_self: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any
                ) -> httpx.Response:
                    return sandbox.intercept_sync_send(client_self, request, *args, **kwargs)

                async def patched_async_send(
                    client_self: httpx.AsyncClient,
                    request: httpx.Request,
                    *args: Any,
                    **kwargs: Any,
                ) -> httpx.Response:
                    return await sandbox.intercept_async_send(client_self, request, *args, **kwargs)

                self._patched_sync_send = patched_sync_send
                self._patched_async_send = patched_async_send
                httpx.Client.send = patched_sync_send  # type: ignore[assignment,method-assign]
                httpx.AsyncClient.send = patched_async_send  # type: ignore[assignment,method-assign]
                self._is_active = True
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Restores original send methods on httpx.Client and httpx.AsyncClient.

        Only restores if this sandbox's patch is still the one currently installed.
        If it isn't (a sandbox was exited out of LIFO order), restoring blindly would
        clobber whichever patch is actually active now, so this warns and leaves the
        current patch in place instead of corrupting it.
        """
        with _PATCH_LOCK:
            if not self._is_active:
                return
            self._is_active = False

            if httpx.Client.send is self._patched_sync_send and self._original_sync_send:
                httpx.Client.send = self._original_sync_send  # type: ignore[method-assign]
            elif httpx.Client.send is self._patched_sync_send:
                pass
            else:
                warnings.warn(
                    "pytest_agent_chaos: StreamChaosSandbox exited out of LIFO order; "
                    "another sandbox's patch is active, so httpx.Client.send was left as-is. "
                    "Enter/exit sandboxes in strict reverse order to avoid this.",
                    stacklevel=2,
                )

            if httpx.AsyncClient.send is self._patched_async_send and self._original_async_send:
                httpx.AsyncClient.send = self._original_async_send  # type: ignore[method-assign]
            elif httpx.AsyncClient.send is self._patched_async_send:
                pass
            else:
                warnings.warn(
                    "pytest_agent_chaos: StreamChaosSandbox exited out of LIFO order; "
                    "another sandbox's patch is active, so httpx.AsyncClient.send was left as-is. "
                    "Enter/exit sandboxes in strict reverse order to avoid this.",
                    stacklevel=2,
                )
