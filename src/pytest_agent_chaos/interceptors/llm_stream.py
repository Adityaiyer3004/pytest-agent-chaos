"""Transport & Stream Interceptor for httpx and LLM providers.

Intercepts outbound calls to OpenAI, Anthropic, Vertex AI, or local LLM mocks
at the HTTP client level to simulate mid-stream SSE (Server-Sent Events) chunk drops,
429 Retry-After storms, gateway timeouts, and network latency jitter.

Patches both `httpx` and, when installed, `httpx2` -- the separate package the
openai SDK switched to in 3.x. The two share an API but no classes, so a patch
on one is invisible to clients of the other.
"""

import asyncio
import functools
import importlib
import random
import threading
import time
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from types import ModuleType
from typing import Any, NamedTuple, cast

import httpx

from pytest_agent_chaos.models import DEFAULT_SEED, ChaosProfile, ChaosTelemetry, FaultType

# Guards install/restore of the process-global httpx.Client.send / AsyncClient.send
# monkeypatch. Sandboxes must still be entered/exited in strict LIFO order (standard
# `with`-block nesting gives you this for free); this lock only prevents a torn
# read/write if two threads install or restore at the same instant.
_PATCH_LOCK = threading.Lock()

_DROP_MESSAGE = "pytest-agent-chaos: Peer connection closed unexpectedly mid-stream."


class ChaosSyncByteStream(httpx.SyncByteStream):
    """Sync byte stream that injects chunk drop or latency midway through reading."""

    _drop_error: type[Exception] = httpx.RemoteProtocolError

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
                raise self._drop_error(_DROP_MESSAGE)

            yield chunk

    def close(self) -> None:
        if hasattr(self._original_stream, "close"):
            self._original_stream.close()


class ChaosAsyncByteStream(httpx.AsyncByteStream):
    """Async byte stream that injects chunk drop or latency midway through async reading."""

    _drop_error: type[Exception] = httpx.RemoteProtocolError

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
                raise self._drop_error(_DROP_MESSAGE)

            yield chunk

    async def aclose(self) -> None:
        if hasattr(self._original_stream, "aclose"):
            await self._original_stream.aclose()


class _Transport(NamedTuple):
    """An HTTP client library whose Client/AsyncClient.send the sandbox patches."""

    lib: ModuleType
    sync_stream: type[ChaosSyncByteStream]
    async_stream: type[ChaosAsyncByteStream]


class _Patch(NamedTuple):
    """The send methods one sandbox replaced on one library, and what it replaced them with."""

    transport: _Transport
    original_sync_send: Callable[..., Any]
    original_async_send: Callable[..., Any]
    patched_sync_send: Callable[..., Any]
    patched_async_send: Callable[..., Any]


@functools.cache
def _transports() -> tuple[_Transport, ...]:
    """httpx always; httpx2 too when it's installed (openai>=3 depends on it)."""
    transports = [_Transport(httpx, ChaosSyncByteStream, ChaosAsyncByteStream)]
    try:
        httpx2 = importlib.import_module("httpx2")
    except ImportError:
        return tuple(transports)
    # httpx2 only reads a Response.stream that is an instance of its own stream base
    # classes, and its callers (e.g. the openai SDK) only catch httpx2 exceptions.
    sync_stream = type(
        "ChaosHttpx2SyncByteStream",
        (ChaosSyncByteStream, httpx2.SyncByteStream),
        {"_drop_error": httpx2.RemoteProtocolError},
    )
    async_stream = type(
        "ChaosHttpx2AsyncByteStream",
        (ChaosAsyncByteStream, httpx2.AsyncByteStream),
        {"_drop_error": httpx2.RemoteProtocolError},
    )
    transports.append(_Transport(httpx2, sync_stream, async_stream))
    return tuple(transports)


class StreamChaosSandbox:
    """Manages transport-level interception for httpx (and httpx2) Client and AsyncClient."""

    def __init__(self, profile: ChaosProfile) -> None:
        self.profile = profile
        self.telemetry = ChaosTelemetry(fault_type=profile.fault_type)
        # Private RNG stream, mirroring ChaosSandbox: probability draws must not
        # depend on what other sandboxes have already pulled from global random.
        # Falls back to DEFAULT_SEED (not OS entropy) if the profile wasn't seeded.
        self._rng = random.Random(profile.seed if profile.seed is not None else DEFAULT_SEED)
        # Keyed by library name ("httpx", "httpx2"); filled in by __enter__.
        self._patches: dict[str, _Patch] = {}
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

    def _patch_for(self, client: httpx.Client | httpx.AsyncClient) -> _Patch:
        """The patch for whichever library (httpx or httpx2) `client` belongs to."""
        for patch in self._patches.values():
            lib = patch.transport.lib
            if isinstance(client, (lib.Client, lib.AsyncClient)):
                return patch
        raise RuntimeError(
            f"pytest_agent_chaos: StreamChaosSandbox has no patch for {type(client).__name__}"
        )

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

    def _generate_429_response(
        self, request: httpx.Request, lib: ModuleType = httpx
    ) -> httpx.Response:
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
        return cast(
            httpx.Response,
            lib.Response(
                status_code=429,
                headers=headers,
                content=content,
                request=request,
            ),
        )

    def _generate_504_response(
        self, request: httpx.Request, lib: ModuleType = httpx
    ) -> httpx.Response:
        self.telemetry.faults_injected += 1
        self._note_fault(request)
        headers = {"Content-Type": "application/json"}
        content = (
            b'{"error": {"message": "Gateway Timeout from upstream LLM provider.", "code": 504}}'
        )
        return cast(
            httpx.Response,
            lib.Response(
                status_code=504,
                headers=headers,
                content=content,
                request=request,
            ),
        )

    def intercept_sync_send(
        self,
        client: httpx.Client,
        request: httpx.Request,
        *args: Any,
        **kwargs: Any,
    ) -> httpx.Response:
        """Intercepts synchronous httpx.Client.send (or httpx2.Client.send) calls."""
        patch = self._patch_for(client)
        lib = patch.transport.lib
        original_send = patch.original_sync_send

        if not self.is_target_host(request.url.host):
            return cast(httpx.Response, original_send(client, request, *args, **kwargs))

        self.telemetry.total_calls_intercepted += 1

        active_fault = self._get_active_fault()

        if active_fault is None:
            clean = cast(httpx.Response, original_send(client, request, *args, **kwargs))
            self._note_response(request, clean)
            return clean

        # Inject Faults
        if active_fault == FaultType.RATE_LIMIT_429:
            return self._generate_429_response(request, lib)

        if active_fault == FaultType.GATEWAY_TIMEOUT_504:
            return self._generate_504_response(request, lib)

        if active_fault == FaultType.CONNECTION_TIMEOUT:
            self.telemetry.faults_injected += 1
            self._note_fault(request)
            raise lib.ConnectTimeout(
                f"pytest-agent-chaos: Simulated connection timeout to {request.url.host}",
                request=request,
            )

        if active_fault == FaultType.LATENCY_JITTER and self.profile.latency_ms > 0:
            time.sleep(self.profile.latency_ms / 1000.0)
            self.telemetry.faults_injected += 1
            self._note_fault(request)

        if active_fault != FaultType.SSE_MIDSTREAM_DROP:
            return cast(httpx.Response, original_send(client, request, *args, **kwargs))

        # SSE drop: only this call is forced to stream, so the body can be cut mid-read.
        caller_streams = bool(kwargs.get("stream", False))
        send_kwargs = {**kwargs, "stream": True}
        response = cast(httpx.Response, original_send(client, request, *args, **send_kwargs))
        # Only mark this endpoint as faulted if the drop actually fires -- if the
        # response has fewer chunks than drop_after_chunks, nothing is ever dropped,
        # and the endpoint must not look "faulted" to the next call's recovery check.
        response.stream = patch.transport.sync_stream(
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
        """Intercepts asynchronous httpx.AsyncClient.send (or httpx2.AsyncClient.send) calls."""
        patch = self._patch_for(client)
        lib = patch.transport.lib
        original_send = patch.original_async_send

        if not self.is_target_host(request.url.host):
            return cast(httpx.Response, await original_send(client, request, *args, **kwargs))

        self.telemetry.total_calls_intercepted += 1

        active_fault = self._get_active_fault()

        if active_fault is None:
            clean = cast(
                httpx.Response,
                await original_send(client, request, *args, **kwargs),
            )
            self._note_response(request, clean)
            return clean

        if active_fault == FaultType.RATE_LIMIT_429:
            return self._generate_429_response(request, lib)

        if active_fault == FaultType.GATEWAY_TIMEOUT_504:
            return self._generate_504_response(request, lib)

        if active_fault == FaultType.CONNECTION_TIMEOUT:
            self.telemetry.faults_injected += 1
            self._note_fault(request)
            raise lib.ConnectTimeout(
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
                await original_send(client, request, *args, **kwargs),
            )

        caller_streams = bool(kwargs.get("stream", False))
        send_kwargs = {**kwargs, "stream": True}
        response = cast(
            httpx.Response,
            await original_send(client, request, *args, **send_kwargs),
        )
        # See the sync path above: only mark faulted if the drop actually fires.
        response.stream = patch.transport.async_stream(
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

    def _install(self, transport: _Transport) -> None:
        lib = transport.lib
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

        self._patches[lib.__name__] = _Patch(
            transport=transport,
            original_sync_send=lib.Client.send,
            original_async_send=lib.AsyncClient.send,
            patched_sync_send=patched_sync_send,
            patched_async_send=patched_async_send,
        )
        lib.Client.send = patched_sync_send
        lib.AsyncClient.send = patched_async_send

    def __enter__(self) -> "StreamChaosSandbox":
        """Installs monkeypatches on Client.send and AsyncClient.send (httpx, and httpx2 if installed).

        Sandboxes nest like any other context manager: entering a second sandbox
        while the first is still active is fine, as long as they are exited in
        the reverse (LIFO) order they were entered — exactly what `with` blocks
        and pytest fixture teardown give you, provided teardown also unwinds LIFO.
        """
        with _PATCH_LOCK:
            if not self._is_active:
                self._patches = {}
                for transport in _transports():
                    self._install(transport)
                self._is_active = True
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Restores the original send methods on every library this sandbox patched.

        Only restores if this sandbox's patch is still the one currently installed.
        If it isn't (a sandbox was exited out of LIFO order), restoring blindly would
        clobber whichever patch is actually active now, so this warns and leaves the
        current patch in place instead of corrupting it.
        """
        with _PATCH_LOCK:
            if not self._is_active:
                return
            self._is_active = False

            for name, patch in self._patches.items():
                lib = patch.transport.lib
                for cls, patched, original in (
                    (lib.Client, patch.patched_sync_send, patch.original_sync_send),
                    (lib.AsyncClient, patch.patched_async_send, patch.original_async_send),
                ):
                    if cls.send is patched:
                        cls.send = original
                    else:
                        warnings.warn(
                            "pytest_agent_chaos: StreamChaosSandbox exited out of LIFO order; "
                            f"another sandbox's patch is active, so {name}.{cls.__name__}.send "
                            "was left as-is. Enter/exit sandboxes in strict reverse order to "
                            "avoid this.",
                            stacklevel=2,
                        )
