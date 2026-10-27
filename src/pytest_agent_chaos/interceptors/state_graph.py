"""State & Loop Guardrail Auditor for agent workflows and LangGraph architectures.

Monitors execution ticks across agent graph state transitions, detecting infinite cycles,
tracking node visitation history, and enforcing strict step budgets to prevent runaways.
"""

import functools
from collections.abc import Callable
from typing import Any

from pytest_agent_chaos.exceptions import InfiniteAgentLoopError
from pytest_agent_chaos.models import GraphAuditTelemetry


def _has_repeating_cycle(transitions: list[str]) -> bool:
    """Detects any repeating node pattern of period 1..n//3 in the trailing transitions.

    Requires 3 full repeats of the period before flagging, matching the confidence bar
    of the original period-2-only check this replaces (e.g. [A, B, A, B, A, B]).
    """
    n = len(transitions)
    for period in range(1, n // 3 + 1):
        block = transitions[-period:]
        if transitions[-2 * period : -period] == block and (
            transitions[-3 * period : -2 * period] == block
        ):
            return True
    return False


class StateGraphAuditor:
    """Monitors state machine / LangGraph node transitions and enforces step ceilings."""

    def __init__(
        self,
        max_steps: int = 10,
        detect_cycles: bool = True,
        raise_on_cycle: bool = False,
    ) -> None:
        self.max_steps = max_steps
        self.detect_cycles = detect_cycles
        self.raise_on_cycle = raise_on_cycle
        self.telemetry = GraphAuditTelemetry(max_steps_allowed=max_steps)

    def record_step(self, node_name: str) -> None:
        """Records a single state transition / node execution tick.

        Args:
            node_name: The name or identifier of the executed state graph node.

        Raises:
            InfiniteAgentLoopError: If the step budget is exhausted, or if a cyclic
                deadlock pattern is detected and raise_on_cycle is True.
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

        if self.detect_cycles and _has_repeating_cycle(self.telemetry.node_transitions):
            self.telemetry.cycle_detected = True
            if self.raise_on_cycle:
                path_str = " -> ".join(self.telemetry.node_transitions[-6:])
                raise InfiniteAgentLoopError(
                    f"Agent graph detected a cyclic deadlock pattern. "
                    f"Recent transition path: ... -> {path_str}"
                )

    def wrap_node(
        self, node_name: str, node_fn: Callable[..., Any] | None = None
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
        self, node_name: str, node_fn: Callable[..., Any] | None = None
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
