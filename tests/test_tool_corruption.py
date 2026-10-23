"""Self-tests verifying tool corruption, schema drift, and Pydantic validation recovery."""

import json

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from pytest_agent_chaos.exceptions import AgentLoopExhaustedError
from pytest_agent_chaos.interceptors.tool_corruptor import ChaosSandbox
from pytest_agent_chaos.models import ChaosProfile, FaultType
from pytest_agent_chaos.plugin import chaos_inject


class InvoicePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invoice_id: str
    amount: float
    is_paid: bool = False


def test_truncated_json_tool_call_interception():
    """Verifies that tool call JSON is sliced mid-payload and agent recovers on call #2."""
    profile = ChaosProfile(
        fault_type=FaultType.TRUNCATED_JSON_TOOL_CALL,
        trigger_on_call=1,
        max_graph_steps=5,
    )
    sandbox = ChaosSandbox(profile)
    raw_tool_args = '{"invoice_id": "INV-101", "amount": 99.50, "is_paid": true}'

    # Attempt 1: Truncated JSON
    attempt_1 = sandbox.mutate_tool_payload(raw_tool_args)
    assert not attempt_1.endswith("}")
    assert sandbox.telemetry.faults_injected == 1
    assert not sandbox.telemetry.recovery_succeeded

    with pytest.raises(json.JSONDecodeError):
        json.loads(attempt_1)

    # Attempt 2: Self-correction retry
    attempt_2 = sandbox.mutate_tool_payload(raw_tool_args)
    assert attempt_2 == raw_tool_args
    assert sandbox.telemetry.recovery_succeeded
    assert sandbox.telemetry.total_retries_observed == 1
    assert sandbox.telemetry.total_calls_intercepted == 2


def test_schema_type_drift_with_pydantic():
    """Verifies that schema type mutation is caught by Pydantic and self-healed on retry."""
    profile = ChaosProfile(
        fault_type=FaultType.SCHEMA_TYPE_DRIFT,
        trigger_on_call=1,
        max_graph_steps=5,
    )
    sandbox = ChaosSandbox(profile)
    valid_args = '{"invoice_id": "INV-202", "amount": 1500.0, "is_paid": false}'

    # Attempt 1: Schema mutated (amount becomes string, invoice_id becomes null)
    corrupted_json = sandbox.mutate_tool_payload(valid_args)

    with pytest.raises(ValidationError):
        InvoicePayload.model_validate_json(corrupted_json)

    assert sandbox.telemetry.faults_injected == 1

    # Attempt 2: Valid payload passes through
    repaired_json = sandbox.mutate_tool_payload(valid_args)
    validated = InvoicePayload.model_validate_json(repaired_json)
    assert validated.invoice_id == "INV-202"
    assert validated.amount == 1500.0
    assert sandbox.telemetry.recovery_succeeded


def test_hallucinated_params_injection():
    """Verifies injection of rogue parameters rejected by strict schema config."""
    profile = ChaosProfile(
        fault_type=FaultType.HALLUCINATED_PARAMS,
        trigger_on_call=1,
    )
    sandbox = ChaosSandbox(profile)
    valid_args = '{"invoice_id": "INV-303", "amount": 340.0}'

    corrupted = sandbox.mutate_tool_payload(valid_args)
    data = json.loads(corrupted)
    assert "__chaos_hallucinated_field__" in data

    with pytest.raises(ValidationError) as exc_info:
        InvoicePayload.model_validate_json(corrupted)

    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_max_graph_steps_ceiling_exhaustion():
    """Verifies that runaway retry loops throw AgentLoopExhaustedError."""
    profile = ChaosProfile(
        fault_type=FaultType.TRUNCATED_JSON_TOOL_CALL,
        trigger_on_call=1,
        max_graph_steps=3,
    )
    sandbox = ChaosSandbox(profile)
    raw = '{"action": "lookup"}'

    sandbox.mutate_tool_payload(raw)  # step 1
    sandbox.mutate_tool_payload(raw)  # step 2
    sandbox.mutate_tool_payload(raw)  # step 3

    with pytest.raises(AgentLoopExhaustedError) as exc_info:
        sandbox.mutate_tool_payload(raw)  # step 4 exceeds max_graph_steps 3

    assert "Agent exceeded max_graph_steps (3)" in str(exc_info.value)


@chaos_inject(
    fault=FaultType.TRUNCATED_JSON_TOOL_CALL,
    trigger_on_call=1,
    max_graph_steps=5,
)
def test_decorator_injection_syntax(chaos_sandbox: ChaosSandbox):
    """Verifies the @chaos_inject decorator injection workflow."""
    raw = '{"target": "database"}'
    first = chaos_sandbox.mutate_tool_payload(raw)
    assert first != raw
    second = chaos_sandbox.mutate_tool_payload(raw)
    assert second == raw
    assert chaos_sandbox.telemetry.recovery_succeeded
