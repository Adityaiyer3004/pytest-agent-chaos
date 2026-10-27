"""Pytest native plugin hooks and fixtures for pytest-agent-chaos.

Provides hooks (pytest_addoption, pytest_configure, pytest_terminal_summary),
fixtures (agent_chaos, stream_chaos, graph_auditor), and decorators (chaos_inject, stream_chaos_inject).
"""

import functools
import inspect
import zlib
from collections.abc import Callable, Generator
from typing import Any

import pytest
from _pytest.terminal import TerminalReporter

from pytest_agent_chaos.interceptors.llm_stream import StreamChaosSandbox
from pytest_agent_chaos.interceptors.state_graph import StateGraphAuditor
from pytest_agent_chaos.interceptors.tool_corruptor import ChaosSandbox
from pytest_agent_chaos.models import ChaosProfile, FaultType
from pytest_agent_chaos.reporting.terminal import (
    GLOBAL_CHAOS_REGISTRY,
    render_chaos_summary_table,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Adds CLI options for agent chaos engineering."""
    group = parser.getgroup("agent-chaos")
    group.addoption(
        "--chaos-seed",
        action="store",
        default=42,
        type=int,
        help="Deterministic seed for reproducible agent chaos injection.",
    )
    group.addoption(
        "--chaos-disable-report",
        action="store_true",
        default=False,
        help="Disable the end-of-session Agent Chaos Resilience Summary table.",
    )


_BASE_SEED: int = 42


def _derive_seed(name: str) -> int:
    """Derives a per-test seed from --chaos-seed and the test/node name.

    Using crc32 (not the builtin hash()) keeps this stable across interpreter
    runs regardless of PYTHONHASHSEED. Each sandbox gets its own RNG stream
    from this, so probability-based firing no longer depends on which other
    chaos tests ran first and how many random draws they made.
    """
    return _BASE_SEED ^ zlib.crc32(name.encode())


def _seeded(profile: ChaosProfile, name: str) -> ChaosProfile:
    """Returns profile with an automatic seed, unless the caller already set one."""
    if profile.seed is not None:
        return profile
    return profile.model_copy(update={"seed": _derive_seed(name)})


def pytest_configure(config: pytest.Config) -> None:
    """Initializes deterministic seeds and registers markers."""
    global _BASE_SEED
    seed = config.getoption("--chaos-seed", default=42)
    if seed is not None:
        _BASE_SEED = seed

    config.addinivalue_line(
        "markers",
        "chaos_inject(fault, trigger_on_call): Injects deterministic tool payload chaos into agent tests.",
    )
    config.addinivalue_line(
        "markers",
        "stream_chaos(fault, drop_after_chunks): Injects transport and SSE stream drops into LLM client calls.",
    )


def pytest_terminal_summary(
    terminalreporter: TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Pytest terminal hook to render the chaos resilience table."""
    if not config.getoption("--chaos-disable-report", default=False):
        render_chaos_summary_table(terminalreporter, GLOBAL_CHAOS_REGISTRY)


@pytest.fixture
def agent_chaos(
    request: pytest.FixtureRequest,
) -> Generator[Callable[[ChaosProfile], ChaosSandbox], None, None]:
    """Pytest fixture providing a configurable ChaosSandbox for tool call corruption."""
    sandboxes: list[ChaosSandbox] = []

    def _create_sandbox(profile: ChaosProfile) -> ChaosSandbox:
        sb = ChaosSandbox(_seeded(profile, request.node.name))
        sb.telemetry.test_name = request.node.name
        GLOBAL_CHAOS_REGISTRY.register(sb.telemetry)
        sandboxes.append(sb)
        return sb

    yield _create_sandbox


@pytest.fixture
def stream_chaos(
    request: pytest.FixtureRequest,
) -> Generator[Callable[[ChaosProfile], StreamChaosSandbox], None, None]:
    """Pytest fixture providing a transport-level StreamChaosSandbox for httpx interception."""
    active_sandboxes: list[StreamChaosSandbox] = []

    def _create_stream_sandbox(profile: ChaosProfile) -> StreamChaosSandbox:
        sandbox = StreamChaosSandbox(profile)
        sandbox.telemetry.test_name = request.node.name
        GLOBAL_CHAOS_REGISTRY.register(sandbox.telemetry)
        active_sandboxes.append(sandbox)
        sandbox.__enter__()
        return sandbox

    yield _create_stream_sandbox

    # Exit in reverse (LIFO) order: each sandbox's __exit__ only restores the
    # monkeypatch if it's still the one currently installed, so exiting in
    # creation order would leave the last-entered sandbox's patch stranded.
    for sb in reversed(active_sandboxes):
        sb.__exit__(None, None, None)


@pytest.fixture
def graph_auditor(
    request: pytest.FixtureRequest,
) -> Generator[Callable[..., StateGraphAuditor], None, None]:
    """Pytest fixture providing a state machine transition auditor and cycle guardrail."""
    auditors: list[StateGraphAuditor] = []

    def _create_auditor(max_steps: int = 10, detect_cycles: bool = True) -> StateGraphAuditor:
        auditor = StateGraphAuditor(max_steps=max_steps, detect_cycles=detect_cycles)
        auditors.append(auditor)
        return auditor

    yield _create_auditor


@pytest.fixture
def chaos_sandbox() -> ChaosSandbox | None:
    """Fixture placeholder allowing tests decorated with @chaos_inject to accept chaos_sandbox."""
    return None


@pytest.fixture
def stream_sandbox() -> StreamChaosSandbox | None:
    """Fixture placeholder allowing tests decorated with @stream_chaos_inject to accept stream_sandbox."""
    return None


@pytest.fixture
def chaos_sidecar(
    request: pytest.FixtureRequest,
) -> Generator[Callable[..., Any], None, None]:
    """Pytest fixture providing a running out-of-process Chaos Proxy Sidecar."""
    from pytest_agent_chaos.sidecar.runner import ChaosSidecarServer

    servers: list[ChaosSidecarServer] = []

    def _start_sidecar(
        upstream_host: str = "https://api.openai.com",
        port: int | None = None,
    ) -> Any:
        srv = ChaosSidecarServer(port=port, upstream_host=upstream_host)
        client = srv.start()
        servers.append(srv)
        return client

    yield _start_sidecar

    for srv in servers:
        srv.stop()


def chaos_inject(
    fault: FaultType,
    trigger_on_call: int = 1,
    max_graph_steps: int = 8,
    probability: float = 1.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to inject deterministic tool corruption into an agent test case."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def _make_sandbox() -> ChaosSandbox:
            profile = ChaosProfile(
                fault_type=fault,
                trigger_on_call=trigger_on_call,
                max_graph_steps=max_graph_steps,
                probability=probability,
            )
            sandbox = ChaosSandbox(_seeded(profile, func.__name__))
            sandbox.telemetry.test_name = func.__name__
            GLOBAL_CHAOS_REGISTRY.register(sandbox.telemetry)
            return sandbox

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                kwargs["chaos_sandbox"] = _make_sandbox()
                return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs["chaos_sandbox"] = _make_sandbox()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def stream_chaos_inject(
    fault: FaultType,
    trigger_on_call: int = 1,
    drop_after_chunks: int = 1,
    retry_after_seconds: int = 2,
    intercept_all_hosts: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to inject transport stream drops or rate limits into an agent test case."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def _make_sandbox() -> StreamChaosSandbox:
            profile = ChaosProfile(
                fault_type=fault,
                trigger_on_call=trigger_on_call,
                drop_after_chunks=drop_after_chunks,
                retry_after_seconds=retry_after_seconds,
                intercept_all_hosts=intercept_all_hosts,
            )
            stream_sb = StreamChaosSandbox(profile)
            stream_sb.telemetry.test_name = func.__name__
            GLOBAL_CHAOS_REGISTRY.register(stream_sb.telemetry)
            return stream_sb

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # The sandbox must stay installed while the coroutine actually runs.
                with _make_sandbox() as stream_sb:
                    kwargs["stream_sandbox"] = stream_sb
                    return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _make_sandbox() as stream_sb:
                kwargs["stream_sandbox"] = stream_sb
                return func(*args, **kwargs)

        return wrapper

    return decorator
