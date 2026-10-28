"""Production stress scenarios pushing pytest-agent-chaos through critical failure modes.

Tests:
1. Partial / Corrupted SSE Stream Mid-Token (TCP drops mid-word, leaving unclosed JSON/markdown).
2. Multi-Turn Cascading Fault Stacking (429 RateLimit -> 504 Gateway Timeout -> 200 OK).
3. Deeply Nested Pydantic v2 Schema Drift & Self-Healing (Valid JSON, but deeply nested type corruption).
4. Sub-agent Deadlock & Cyclic Ping-Pong (Agent A <-> Agent B infinite cycle halted at step budget Nmax).
"""

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError

from pytest_agent_chaos.exceptions import InfiniteAgentLoopError
from pytest_agent_chaos.interceptors.llm_stream import StreamChaosSandbox
from pytest_agent_chaos.interceptors.state_graph import StateGraphAuditor
from pytest_agent_chaos.interceptors.tool_corruptor import ChaosSandbox
from pytest_agent_chaos.models import ChaosProfile, FaultType

# ==============================================================================
# Scenario 1: Mid-Token SSE Disconnect
# ==============================================================================


def test_production_stress_mid_token_sse_disconnect() -> None:
    """Simulates an upstream LLM provider dropping the TCP connection mid-token,

    leaving an unclosed JSON payload, and verifies resilient client reconnection.
    """
    profile = ChaosProfile(
        fault_type=FaultType.SSE_MIDSTREAM_DROP,
        trigger_on_call=1,
        drop_after_chunks=1,
        drop_mid_chunk=True,  # Truncates chunk mid-word before severing socket
        target_domains=["api.openai.com"],
    )

    full_chunks = [
        b'data: {"choices": [{"delta": {"content": "Analyzing "}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "quarterly revenue metrics..."}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    def streaming_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=iter(full_chunks),
        )

    with StreamChaosSandbox(profile) as sandbox:
        client = httpx.Client(transport=httpx.MockTransport(streaming_handler))
        req = client.build_request("POST", "https://api.openai.com/v1/chat/completions")

        # Turn 1: Client streams tokens, but chunk 2 is sliced mid-word and socket drops
        resp1 = client.send(req, stream=True)
        received_bytes: list[bytes] = []

        with pytest.raises(httpx.RemoteProtocolError):
            for chunk in resp1.iter_bytes():
                received_bytes.append(chunk)  # noqa: PERF402

        # Confirm that chunk 1 passed and chunk 2 was partially received before cutoff
        assert len(received_bytes) == 2
        assert b"Analyzing " in received_bytes[0]
        # Chunk 2 was halved mid-chunk
        assert len(received_bytes[1]) < len(full_chunks[1])
        assert sandbox.telemetry.faults_injected == 1

        # Resilient agent fallback: Re-opens stream on Turn 2 and receives all 3 chunks intact
        resp2 = client.send(req, stream=True)
        clean_chunks = list(resp2.iter_bytes())

        assert len(clean_chunks) == 3
        assert b"quarterly revenue" in clean_chunks[1]
        assert sandbox.telemetry.recovery_succeeded is True
        assert sandbox.telemetry.total_retries_observed == 1


# ==============================================================================
# Scenario 2: Multi-Turn Cascading Fault Stacking (429 -> 504 -> 200)
# ==============================================================================


def test_production_stress_multi_turn_fault_stacking_429_504_200() -> None:
    """Verifies that an agent retry loop navigates a multi-turn fault cascade:

    Call 1: 429 RateLimit (with Retry-After: 1)
    Call 2: 504 Gateway Timeout from LLM backend
    Call 3: 200 OK Success
    """
    profile = ChaosProfile(
        fault_schedule={
            1: FaultType.RATE_LIMIT_429,
            2: FaultType.GATEWAY_TIMEOUT_504,
        },
        retry_after_seconds=1,
        target_domains=["api.anthropic.com"],
    )

    call_history: list[int] = []

    def mock_backend(request: httpx.Request) -> httpx.Response:
        call_history.append(200)
        return httpx.Response(
            status_code=200,
            json={"content": [{"text": "Anthropic Claude synthesis complete."}]},
        )

    with StreamChaosSandbox(profile) as sandbox:
        client = httpx.Client(transport=httpx.MockTransport(mock_backend))
        req = client.build_request("POST", "https://api.anthropic.com/v1/messages")

        # Simulated Production Agent Retry Middleware
        # Attempts execution with backoff upon 429 or 504
        max_attempts = 4
        final_response: httpx.Response | None = None

        for attempt in range(1, max_attempts + 1):
            resp = client.send(req)
            if resp.status_code == 429:
                # Inspect retry-after header and continue to next retry
                assert resp.headers.get("Retry-After") == "1"
                continue
            elif resp.status_code == 504:
                # Upstream gateway error, retry
                continue
            elif resp.status_code == 200:
                final_response = resp
                break

        assert final_response is not None
        assert final_response.status_code == 200
        assert final_response.json()["content"][0]["text"] == "Anthropic Claude synthesis complete."

        # Verify telemetry tracked all faults across turns
        assert sandbox.telemetry.faults_injected == 2
        assert sandbox.telemetry.total_calls_intercepted == 3
        assert sandbox.telemetry.recovery_succeeded is True
        assert sandbox.telemetry.total_retries_observed == 1


# ==============================================================================
# Scenario 3: Deeply Nested Pydantic v2 Schema Drift & Self-Healing
# ==============================================================================


class PricingDetails(BaseModel):
    base_price: float = Field(ge=0.0)
    tax_rate: float = Field(ge=0.0, le=1.0)
    currency: str


class OrderItem(BaseModel):
    sku: str
    quantity: int = Field(ge=1)
    pricing: PricingDetails


class CheckoutPayload(BaseModel):
    order_id: str
    customer_tier: str
    items: list[OrderItem]


def test_production_stress_nested_pydantic_schema_drift_self_healing() -> None:
    """Verifies that schema drift accurately mutates deeply nested Pydantic v2 structures,

    triggers precise field validation errors, and verifies agent recovery.
    """
    profile = ChaosProfile(
        fault_type=FaultType.SCHEMA_TYPE_DRIFT,
        trigger_on_call=1,
        max_graph_steps=5,
    )
    sandbox = ChaosSandbox(profile)

    valid_payload = {
        "order_id": "ORD-9921",
        "customer_tier": "VIP_PLATINUM",
        "items": [
            {
                "sku": "GPU-H100-80G",
                "quantity": 4,
                "pricing": {
                    "base_price": 29999.00,
                    "tax_rate": 0.19,
                    "currency": "EUR",
                },
            }
        ],
    }
    raw_json = json.dumps(valid_payload)

    # Attempt 1: Deep recursive mutation corrupts inner numeric/string types
    corrupted_json = sandbox.mutate_tool_payload(raw_json)

    # Verify Pydantic v2 catches the deeply nested field corruption
    with pytest.raises(ValidationError) as exc_info:
        CheckoutPayload.model_validate_json(corrupted_json)

    errors = exc_info.value.errors()
    # Confirm multiple inner fields in items[0].pricing failed validation
    failed_locs = [tuple(err["loc"]) for err in errors]
    assert ("items", 0, "pricing", "base_price") in failed_locs
    assert ("items", 0, "pricing", "tax_rate") in failed_locs
    assert sandbox.telemetry.faults_injected == 1

    # Attempt 2: Resilient agent self-corrects and passes clean payload
    clean_json = sandbox.mutate_tool_payload(raw_json)
    validated = CheckoutPayload.model_validate_json(clean_json)

    assert validated.order_id == "ORD-9921"
    assert validated.items[0].pricing.tax_rate == 0.19
    assert sandbox.telemetry.recovery_succeeded is True
    assert sandbox.telemetry.total_retries_observed == 1


# ==============================================================================
# Scenario 4: Sub-Agent Deadlock & Cyclic Ping-Pong Ceiling Enforcement
# ==============================================================================


def test_production_stress_subagent_deadlock_cycle_ceiling() -> None:
    """Simulates Sub-agent A and Sub-agent B caught in an infinite delegation deadlock

    (A -> B -> A -> B...) and verifies StateGraphAuditor aborts at step budget Nmax.
    """
    max_budget = 8
    auditor = StateGraphAuditor(max_steps=max_budget, detect_cycles=True)

    # Wrap sub-agents with state graph auditor
    @auditor.wrap_node("triage_agent")
    def triage_agent_node(task_state: dict[str, Any]) -> dict[str, Any]:
        task_state["unresolved"] = True
        return task_state

    @auditor.wrap_node("specialist_agent")
    def specialist_agent_node(task_state: dict[str, Any]) -> dict[str, Any]:
        task_state["needs_triage"] = True
        return task_state

    # Run cyclic sub-agent ping-pong loop until guardrail trips
    state: dict[str, Any] = {"query": "Ambiguous multi-turn user intent"}

    with pytest.raises(InfiniteAgentLoopError) as exc_info:
        while True:
            state = triage_agent_node(state)
            state = specialist_agent_node(state)

    # Assert that execution was halted before infinite runaway
    assert f"Agent graph exceeded maximum step budget ({max_budget})" in str(exc_info.value)
    assert auditor.telemetry.exhausted is True
    assert auditor.telemetry.cycle_detected is True
    assert auditor.telemetry.step_count == max_budget + 1  # 9th step triggered ceiling trip

    # Confirm transition log records exact alternating cycle
    transitions = auditor.telemetry.node_transitions
    assert transitions[0] == "triage_agent"
    assert transitions[1] == "specialist_agent"
    assert transitions[2] == "triage_agent"
    assert transitions[3] == "specialist_agent"
