"""State & Loop Guardrail Auditor for agent workflows and LangGraph architectures.

Monitors execution ticks across agent graph state transitions, detecting infinite cycles,
tracking node visitation history, and enforcing strict step budgets to prevent runaways.
"""

import functools
from typing import Any, Callable, Dict, List, Optional
from pytest_agent_chaos.exceptions import AgentLoopExhaustedError, InfiniteAgentLoopError
from pytest_agent_chaos.models import GraphAuditTelemetry


class StateGraphAuditor:
    """Monitors state machine / LangGraph node transitions and enforces step ceilings."""

    def __init__(self, max_steps: int = 10, detect_cycles: bool = True) -> None:
        self.max_steps = max_steps
        self.detect_cycles = detect_cycles
        self.telemetry = GraphAuditTelemetry(max_steps_allowed=max_steps)

    def record_step(self, node_name: str) -> None:
        """Records a single state transition / node execution tick.

        Args:
            node_name: The name or identifier of the executed state graph node.

        Raises:
            InfiniteAgentLoopError: If the step budget is exhausted.
        """
        self.telemetry.step_count += 1
        self.telemetry.node_transitions.append(node_name)

        # Check budget
        if self.telemetry.step_count > self.max_steps:
            self.telemetry.exhausted = True
            path_str = " -> ".join(self.telemetry.node_transitions[-6:])
            raise InfiniteAgentLoopError(
                f"Agent graph exceeded maximum step budget ({self.max_steps}). "
                f"Recent transition path: ... -> {path_str}"
            )

        # Basic cycle detection (e.g., repeated node sequence [A, B, A, B, A, B])
        if self.detect_cycles and len(self.telemetry.node_transitions) >= 6:
            recent = self.telemetry.node_transitions[-6:]
            if recent[0:2] == recent[2:4] == recent[4:6]:
                self.telemetry.cycle_detected = True

    def wrap_node(
        self, node_name: str, node_fn: Optional[Callable[..., Any]] = None
    ) -> Callable[..., Any]:
        """Wraps a LangGraph node or step callable to audit every invocation."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self.record_step(node_name)
                return fn(*args, **kwargs)

            return wrapper

        if node_fn is not None:
            return decorator(node_fn)
        return decorator

    def wrap_async_node(
        self, node_name: str, node_fn: Optional[Callable[..., Any]] = None
    ) -> Callable[..., Any]:
        """Wraps an asynchronous graph node to audit invocation."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                self.record_step(node_name)
                return await fn(*args, **kwargs)

            return async_wrapper

        if node_fn is not None:
            return decorator(node_fn)
        return decorator
