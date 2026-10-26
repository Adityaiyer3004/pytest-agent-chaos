"""Tool & function call corruption interceptor.

Wraps agent tool nodes (compatible with LangGraph @tool, PydanticAI, or raw callables)
to inject schema drift, truncated JSON payloads, hallucinated parameter types,
and malformed tool arguments before they hit the agent's parser.
"""

import functools
import json
import random
from collections.abc import Callable
from typing import Any

from pytest_agent_chaos.exceptions import AgentLoopExhaustedError
from pytest_agent_chaos.models import ChaosProfile, ChaosTelemetry, FaultType


class ChaosSandbox:
    """Manages tool call interception, payload mutation, and recovery telemetry."""

    def __init__(self, profile: ChaosProfile) -> None:
        self.profile = profile
        self.telemetry = ChaosTelemetry(fault_type=profile.fault_type)

    def mutate_tool_payload(self, raw_arguments: str) -> str:
        """Deterministically corrupts LLM tool-call JSON payloads to test parser recovery.

        Args:
            raw_arguments: The serialized JSON arguments emitted by the LLM or agent.

        Returns:
            The mutated (or clean) arguments string.

        Raises:
            AgentLoopExhaustedError: If total intercepted calls exceed profile.max_graph_steps.
        """
        self.telemetry.total_calls_intercepted += 1

        if self.telemetry.total_calls_intercepted > self.profile.max_graph_steps:
            raise AgentLoopExhaustedError(
                f"Agent exceeded max_graph_steps ({self.profile.max_graph_steps}) without terminating."
            )

        should_inject = (
            self.telemetry.total_calls_intercepted == self.profile.trigger_on_call
            and random.random() <= self.profile.probability
        )

        if not should_inject:
            # If a fault was injected previously and we receive a subsequent call,
            # that means the agent recovered / retried!
            if self.telemetry.faults_injected > 0:
                self.telemetry.total_retries_observed += 1
                self.telemetry.recovery_succeeded = True
            return raw_arguments

        self.telemetry.faults_injected += 1

        if self.profile.fault_type == FaultType.TRUNCATED_JSON_TOOL_CALL:
            # Slice off the closing braces/quotes to simulate max_tokens cutoff
            cutoff = max(1, len(raw_arguments) // 2)
            return raw_arguments[:cutoff]

        if self.profile.fault_type == FaultType.SCHEMA_TYPE_DRIFT:

            def _drift_value(val: Any) -> Any:
                if isinstance(val, (int, float)):
                    return "CHAOS_INVALID_NUMBER_TYPE"
                elif isinstance(val, bool):
                    return ["unexpected", "array"]
                elif isinstance(val, str):
                    return None
                elif isinstance(val, list):
                    return [_drift_value(item) for item in val]
                elif isinstance(val, dict):
                    return {k: _drift_value(v) for k, v in val.items()}
                return val

            try:
                parsed: Any = json.loads(raw_arguments)
                if isinstance(parsed, dict):
                    mutated = {k: _drift_value(v) for k, v in parsed.items()}
                    return json.dumps(mutated)
                elif isinstance(parsed, list):
                    mutated_list = [_drift_value(item) for item in parsed]
                    return json.dumps(mutated_list)
                return json.dumps(parsed)
            except json.JSONDecodeError:
                return "{corrupted_payload: true"

        if self.profile.fault_type == FaultType.HALLUCINATED_PARAMS:
            # Inject unexpected rogue properties to test strict schema rejection
            try:
                parsed = json.loads(raw_arguments)
                if isinstance(parsed, dict):
                    parsed["__chaos_hallucinated_field__"] = "phantom_token_xyz"
                    parsed["confidence_score"] = "hallucinated_float_as_str"
                return json.dumps(parsed)
            except json.JSONDecodeError:
                return raw_arguments + ', "hallucinated": true}'

        return raw_arguments

    def wrap_tool(self, tool_func: Callable[..., Any]) -> Callable[..., Any]:
        """Wraps a tool callable to automatically mutate incoming arguments according to the ChaosProfile.

        Can wrap standard Python functions, LangGraph tools, or Pydantic callables.
        """

        @functools.wraps(tool_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # If arguments are passed as a JSON string or kwargs
            if args and isinstance(args[0], str):
                mutated_arg = self.mutate_tool_payload(args[0])
                return tool_func(mutated_arg, *args[1:], **kwargs)

            # If arguments passed as kwargs, convert to json, mutate, and restore
            if kwargs:
                try:
                    serialized = json.dumps(kwargs)
                    mutated = self.mutate_tool_payload(serialized)
                    try:
                        new_kwargs = json.loads(mutated)
                        return tool_func(*args, **new_kwargs)
                    except json.JSONDecodeError:
                        # Pass mutated string to test caller's error handling
                        return tool_func(mutated, *args)
                except (TypeError, ValueError):
                    pass

            return tool_func(*args, **kwargs)

        return wrapper

    def record_recovery(self) -> None:
        """Explicitly records that the agent recovered from an injected chaos fault."""
        self.telemetry.recovery_succeeded = True
