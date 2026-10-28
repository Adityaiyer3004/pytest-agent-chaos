"""Tool & function call corruption interceptor.

Wraps agent tool nodes (compatible with LangGraph @tool, PydanticAI, or raw callables)
to inject schema drift, truncated JSON payloads, hallucinated parameter types,
and malformed tool arguments before they hit the agent's parser.
"""

import functools
import json
import random
import warnings
from collections.abc import Callable
from typing import Any

from pytest_agent_chaos.exceptions import AgentLoopExhaustedError
from pytest_agent_chaos.models import DEFAULT_SEED, ChaosProfile, ChaosTelemetry, FaultType


class ChaosSandbox:
    """Manages tool call interception, payload mutation, and recovery telemetry."""

    def __init__(self, profile: ChaosProfile) -> None:
        self.profile = profile
        self.telemetry = ChaosTelemetry(fault_type=profile.fault_type)
        # Private RNG stream: probability draws must not depend on what other
        # sandboxes have already pulled from the (shared, order-dependent) global random.
        # Falls back to DEFAULT_SEED (not OS entropy) if the profile wasn't seeded.
        self._rng = random.Random(profile.seed if profile.seed is not None else DEFAULT_SEED)
        # Shape of the payload that was corrupted; None until a fault fires.
        # ("dict", keys) for JSON objects; (type_name, None) for anything else, so a
        # retry is only "the same tool call" if it matches both type and (for dicts) keys.
        self._faulted_shape: tuple[str, frozenset[str] | None] | None = None

    @staticmethod
    def _object_shape(raw: str) -> tuple[str, frozenset[str] | None] | None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return ("dict", frozenset(parsed))
        return (type(parsed).__name__, None)

    def _is_recovery(self, raw_arguments: str) -> bool:
        """A clean call is recovery only if it is well-formed JSON for the same tool call.

        Same tool call = same top-level JSON type, and for objects, the same top-level
        keys, as the payload that was corrupted. This checks structure, not schema
        validity; call record_recovery() when you have stronger evidence (e.g. the tool
        executed successfully).
        """
        if self._faulted_shape is None:
            return False
        shape = self._object_shape(raw_arguments)
        if shape is None:
            return False
        return shape == self._faulted_shape

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
            and self._rng.random() <= self.profile.probability
        )

        if not should_inject:
            if self._faulted_shape is not None:
                self.telemetry.total_retries_observed += 1
                if self._is_recovery(raw_arguments):
                    self.telemetry.recovery_succeeded = True
            return raw_arguments

        self.telemetry.faults_injected += 1
        self._faulted_shape = self._object_shape(raw_arguments) or ("unknown", None)

        if self.profile.fault_type == FaultType.TRUNCATED_JSON_TOOL_CALL:
            # Slice off the closing braces/quotes to simulate max_tokens cutoff
            cutoff = max(1, len(raw_arguments) // 2)
            return raw_arguments[:cutoff]

        if self.profile.fault_type == FaultType.SCHEMA_TYPE_DRIFT:

            def _drift_value(val: Any) -> Any:
                # bool must be checked before int: bool is a subclass of int.
                if isinstance(val, bool):
                    return ["unexpected", "array"]
                elif isinstance(val, (int, float)):
                    return "CHAOS_INVALID_NUMBER_TYPE"
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
                except (TypeError, ValueError):
                    # A non-JSON-serializable kwarg (set, bytes, custom object, ...)
                    # can't be run through mutate_tool_payload at all. Warn rather than
                    # silently skipping: an un-intercepted call here doesn't advance
                    # total_calls_intercepted, so trigger_on_call would otherwise fire
                    # on the wrong call with no indication why.
                    warnings.warn(
                        "pytest_agent_chaos: wrap_tool could not JSON-serialize kwargs "
                        f"for {tool_func.__name__!r}; this call was not intercepted and "
                        "does not count toward trigger_on_call.",
                        stacklevel=2,
                    )
                    return tool_func(*args, **kwargs)

                mutated = self.mutate_tool_payload(serialized)
                try:
                    new_kwargs = json.loads(mutated)
                    return tool_func(*args, **new_kwargs)
                except json.JSONDecodeError:
                    # Pass mutated string to test caller's error handling
                    return tool_func(mutated, *args)

            return tool_func(*args, **kwargs)

        return wrapper

    def record_recovery(self) -> None:
        """Explicitly records that the agent recovered from an injected chaos fault."""
        self.telemetry.recovery_succeeded = True
