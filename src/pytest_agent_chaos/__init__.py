"""pytest-agent-chaos: Deterministic chaos engineering and fault-injection for multi-agent LLM systems."""

from typing import Any

from pytest_agent_chaos.exceptions import (
    AgentChaosException,
    AgentLoopExhaustedError,
    ChaosStreamDropError,
    ChaosTimeoutError,
    InfiniteAgentLoopError,
    UnhandledChaosFaultError,
)
from pytest_agent_chaos.interceptors.llm_stream import (
    ChaosAsyncByteStream,
    ChaosSyncByteStream,
    StreamChaosSandbox,
)
from pytest_agent_chaos.interceptors.state_graph import StateGraphAuditor
from pytest_agent_chaos.interceptors.tool_corruptor import ChaosSandbox
from pytest_agent_chaos.models import (
    ChaosProfile,
    ChaosTelemetry,
    FaultType,
    GraphAuditTelemetry,
)
from pytest_agent_chaos.plugin import chaos_inject, stream_chaos_inject

__version__ = "0.1.0"

# The sidecar (fastapi/uvicorn) is an optional extra (`pip install
# pytest-agent-chaos[sidecar]`), not a core dependency. Importing it eagerly here
# would break plain `pip install pytest-agent-chaos` entirely: pytest's pytest11
# entry point loads this package to reach plugin.py, so any ImportError here is
# fatal to every test run, not just to sidecar usage. Load it lazily instead.
_LAZY_SIDECAR_EXPORTS = ("ChaosSidecarServer", "RemoteChaosClient", "run_chaos_sidecar")


def __getattr__(name: str) -> Any:
    if name in _LAZY_SIDECAR_EXPORTS:
        from pytest_agent_chaos import sidecar

        return getattr(sidecar, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentChaosException",
    "AgentLoopExhaustedError",
    "ChaosAsyncByteStream",
    "ChaosProfile",
    "ChaosSandbox",
    "ChaosSidecarServer",
    "ChaosStreamDropError",
    "ChaosSyncByteStream",
    "ChaosTelemetry",
    "ChaosTimeoutError",
    "FaultType",
    "GraphAuditTelemetry",
    "InfiniteAgentLoopError",
    "RemoteChaosClient",
    "StateGraphAuditor",
    "StreamChaosSandbox",
    "UnhandledChaosFaultError",
    "chaos_inject",
    "run_chaos_sidecar",
    "stream_chaos_inject",
]
