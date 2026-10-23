"""Data models and configuration schemas for pytest-agent-chaos.

Defines deterministic fault profiles, telemetry tracking, and fault types
for chaos testing across LLM transport streams, tool invocation payloads,
and agent graph state machines.
"""

from enum import Enum

from pydantic import BaseModel, Field


class FaultType(str, Enum):
    """Enumeration of deterministic fault types injected by agent-chaos."""

    RATE_LIMIT_429 = "rate_limit_429"
    GATEWAY_TIMEOUT_504 = "gateway_timeout_504"
    CONNECTION_TIMEOUT = "connection_timeout"
    TRUNCATED_JSON_TOOL_CALL = "truncated_json_tool_call"
    SCHEMA_TYPE_DRIFT = "schema_type_drift"
    HALLUCINATED_PARAMS = "hallucinated_params"
    SSE_MIDSTREAM_DROP = "sse_midstream_drop"
    LATENCY_JITTER = "latency_jitter"


class ChaosProfile(BaseModel):
    """Configuration profile governing deterministic fault injection parameters."""

    fault_type: FaultType
    probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of fault triggering on the target call",
    )
    trigger_on_call: int = Field(
        default=1, ge=1, description="1-indexed call occurrence when fault must inject"
    )
    latency_ms: int = Field(
        default=0,
        ge=0,
        description="Synthetic latency in milliseconds to inject before/during response",
    )
    max_allowed_retries: int = Field(
        default=3, ge=0, description="Max retry attempts expected from resilient agent"
    )
    max_graph_steps: int = Field(
        default=10, ge=1, description="Ceiling for graph state transitions before loop abort"
    )
    retry_after_seconds: int = Field(
        default=2, ge=0, description="Retry-After header value for RATE_LIMIT_429"
    )
    drop_after_chunks: int = Field(
        default=1, ge=0, description="Number of SSE chunks emitted before cutting connection"
    )
    target_domains: list[str] = Field(
        default_factory=lambda: [
            "api.openai.com",
            "api.anthropic.com",
            "generativelanguage.googleapis.com",
            "mock-llm.local",
        ],
        description="Target hosts for HTTP transport interception",
    )
    intercept_all_hosts: bool = Field(
        default=False,
        description="If True, intercepts all outbound httpx requests regardless of host",
    )


class ChaosTelemetry(BaseModel):
    """Execution telemetry captured during a test execution tick."""

    test_name: str | None = None
    fault_type: FaultType | None = None
    total_calls_intercepted: int = 0
    faults_injected: int = 0
    recovery_succeeded: bool = False
    total_retries_observed: int = 0
    peak_step_count: int = 0
    error_messages: list[str] = Field(default_factory=list)

    def mark_recovery(self) -> None:
        """Flags that the agent successfully adapted or retried after the fault."""
        self.recovery_succeeded = True


class GraphAuditTelemetry(BaseModel):
    """Telemetry capturing agent state graph transitions and execution budgets."""

    node_transitions: list[str] = Field(default_factory=list)
    step_count: int = 0
    max_steps_allowed: int = 10
    cycle_detected: bool = False
    exhausted: bool = False
