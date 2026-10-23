"""pytest-agent-chaos: Deterministic chaos engineering and fault-injection for multi-agent LLM systems."""

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

__all__ = [
    "AgentChaosException",
    "AgentLoopExhaustedError",
    "ChaosAsyncByteStream",
    "ChaosProfile",
    "ChaosSandbox",
    "ChaosStreamDropError",
    "ChaosSyncByteStream",
    "ChaosTelemetry",
    "ChaosTimeoutError",
    "FaultType",
    "GraphAuditTelemetry",
    "InfiniteAgentLoopError",
    "StateGraphAuditor",
    "StreamChaosSandbox",
    "UnhandledChaosFaultError",
    "chaos_inject",
    "stream_chaos_inject",
]
