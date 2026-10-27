"""Targeted tests for code paths a coverage run showed were otherwise untested.

Each test here exists because `coverage run --source=src -m pytest` showed the
line uncovered by the rest of the suite, not because of a bug report.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from pytest_agent_chaos.exceptions import InfiniteAgentLoopError
from pytest_agent_chaos.interceptors.llm_stream import StreamChaosSandbox
from pytest_agent_chaos.interceptors.state_graph import StateGraphAuditor
from pytest_agent_chaos.interceptors.tool_corruptor import ChaosSandbox
from pytest_agent_chaos.models import ChaosProfile, FaultType
from pytest_agent_chaos.plugin import stream_chaos_inject

# --- ChaosSandbox: HALLUCINATED_PARAMS, wrap_tool, record_recovery ---


def test_hallucinated_params_injects_rogue_fields_into_dict_payload() -> None:
    sb = ChaosSandbox(ChaosProfile(fault_type=FaultType.HALLUCINATED_PARAMS, trigger_on_call=1))
    out = json.loads(sb.mutate_tool_payload('{"invoice_id": "A"}'))
    assert out["invoice_id"] == "A"
    assert out["__chaos_hallucinated_field__"] == "phantom_token_xyz"
    assert out["confidence_score"] == "hallucinated_float_as_str"


def test_hallucinated_params_on_non_json_payload_appends_raw_suffix() -> None:
    sb = ChaosSandbox(ChaosProfile(fault_type=FaultType.HALLUCINATED_PARAMS, trigger_on_call=1))
    out = sb.mutate_tool_payload("not json")
    assert out == 'not json, "hallucinated": true}'


def test_wrap_tool_mutates_single_string_argument() -> None:
    sb = ChaosSandbox(
        ChaosProfile(fault_type=FaultType.TRUNCATED_JSON_TOOL_CALL, trigger_on_call=1)
    )

    def tool(raw_args: str) -> str:
        return raw_args

    wrapped = sb.wrap_tool(tool)
    result = wrapped('{"a": 1, "b": 2}')
    assert result != '{"a": 1, "b": 2}'
    assert sb.telemetry.faults_injected == 1


def test_wrap_tool_mutates_kwargs_and_restores_valid_json() -> None:
    sb = ChaosSandbox(ChaosProfile(fault_type=FaultType.SCHEMA_TYPE_DRIFT, trigger_on_call=1))
    received: dict[str, object] = {}

    def tool(**kwargs: object) -> None:
        received.update(kwargs)

    wrapped = sb.wrap_tool(tool)
    wrapped(flag=True, n=3)
    assert received["flag"] == ["unexpected", "array"]
    assert received["n"] == "CHAOS_INVALID_NUMBER_TYPE"


def test_wrap_tool_passes_mutated_string_through_when_kwargs_become_invalid_json() -> None:
    sb = ChaosSandbox(
        ChaosProfile(fault_type=FaultType.TRUNCATED_JSON_TOOL_CALL, trigger_on_call=1)
    )
    captured: list[tuple[object, ...]] = []

    def tool(*args: object, **kwargs: object) -> None:
        captured.append(args)

    wrapped = sb.wrap_tool(tool)
    wrapped(a=1, b=2, c=3, d=4, e=5)
    assert captured, "tool should have been called with the truncated string as a positional arg"
    (args,) = captured
    assert isinstance(args[0], str)


def test_wrap_tool_falls_through_when_no_args_or_kwargs() -> None:
    sb = ChaosSandbox(
        ChaosProfile(fault_type=FaultType.TRUNCATED_JSON_TOOL_CALL, trigger_on_call=1)
    )
    calls = []

    def tool() -> str:
        calls.append(1)
        return "ok"

    wrapped = sb.wrap_tool(tool)
    assert wrapped() == "ok"
    assert sb.telemetry.faults_injected == 0


def test_record_recovery_sets_flag_explicitly() -> None:
    sb = ChaosSandbox(
        ChaosProfile(fault_type=FaultType.TRUNCATED_JSON_TOOL_CALL, trigger_on_call=1)
    )
    assert sb.telemetry.recovery_succeeded is False
    sb.record_recovery()
    assert sb.telemetry.recovery_succeeded is True


def test_is_recovery_rejects_malformed_json_after_a_fault() -> None:
    sb = ChaosSandbox(
        ChaosProfile(fault_type=FaultType.TRUNCATED_JSON_TOOL_CALL, trigger_on_call=1)
    )
    sb.mutate_tool_payload('{"a": 1}')
    assert sb._is_recovery("not json at all") is False


# --- StateGraphAuditor: wrap_node/wrap_async_node direct-call form, async decorator ---


def test_wrap_node_direct_call_form() -> None:
    auditor = StateGraphAuditor(max_steps=5)

    def step(x: int) -> int:
        return x + 1

    wrapped = auditor.wrap_node("incrementer", step)
    assert wrapped(1) == 2
    assert auditor.telemetry.node_transitions == ["incrementer"]


@pytest.mark.asyncio
async def test_wrap_async_node_as_decorator() -> None:
    auditor = StateGraphAuditor(max_steps=5)

    @auditor.wrap_async_node("async_step")
    async def step(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    assert await step(3) == 6
    assert auditor.telemetry.node_transitions == ["async_step"]


@pytest.mark.asyncio
async def test_wrap_async_node_direct_call_form_and_budget_enforced() -> None:
    auditor = StateGraphAuditor(max_steps=1)

    async def step(x: int) -> int:
        return x

    wrapped = auditor.wrap_async_node("async_step", step)
    assert await wrapped(1) == 1
    with pytest.raises(InfiniteAgentLoopError):
        await wrapped(1)


# --- StreamChaosSandbox: untested faults and host-filtering paths ---


def test_connection_timeout_fault_sync() -> None:
    profile = ChaosProfile(
        fault_type=FaultType.CONNECTION_TIMEOUT, trigger_on_call=1, intercept_all_hosts=True
    )
    with (
        StreamChaosSandbox(profile) as sb,
        httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200))) as client,
    ):
        req = client.build_request("GET", "https://api.openai.com/v1/models")
        with pytest.raises(httpx.ConnectTimeout):
            client.send(req)
        assert sb.telemetry.faults_injected == 1


@pytest.mark.asyncio
async def test_connection_timeout_fault_async() -> None:
    profile = ChaosProfile(
        fault_type=FaultType.CONNECTION_TIMEOUT, trigger_on_call=1, intercept_all_hosts=True
    )
    with StreamChaosSandbox(profile) as sb:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(200))
        ) as client:
            req = client.build_request("GET", "https://api.openai.com/v1/models")
            with pytest.raises(httpx.ConnectTimeout):
                await client.send(req)
            assert sb.telemetry.faults_injected == 1


def test_latency_jitter_fault_sync_delays_but_returns_response() -> None:
    profile = ChaosProfile(
        fault_type=FaultType.LATENCY_JITTER,
        trigger_on_call=1,
        latency_ms=5,
        intercept_all_hosts=True,
    )
    with (
        StreamChaosSandbox(profile),
        httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))
        ) as client,
    ):
        req = client.build_request("GET", "https://api.openai.com/v1/models")
        resp = client.send(req)
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_latency_jitter_fault_async_delays_but_returns_response() -> None:
    profile = ChaosProfile(
        fault_type=FaultType.LATENCY_JITTER,
        trigger_on_call=1,
        latency_ms=5,
        intercept_all_hosts=True,
    )
    with StreamChaosSandbox(profile):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))
        ) as client:
            req = client.build_request("GET", "https://api.openai.com/v1/models")
            resp = await client.send(req)
            assert resp.status_code == 200


def test_non_target_host_passes_through_untouched() -> None:
    profile = ChaosProfile(
        fault_type=FaultType.RATE_LIMIT_429, trigger_on_call=1
    )  # default target_domains
    with (
        StreamChaosSandbox(profile),
        httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))
        ) as client,
    ):
        req = client.build_request("GET", "https://example.com/unrelated")
        resp = client.send(req)
        assert resp.status_code == 200


@stream_chaos_inject(fault=FaultType.RATE_LIMIT_429, trigger_on_call=1, intercept_all_hosts=True)
def test_stream_chaos_inject_decorator_applies_to_sync_test(
    stream_sandbox: StreamChaosSandbox,
) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))
    ) as client:
        resp = client.get("https://api.openai.com/v1/models")
    assert resp.status_code == 429
    assert stream_sandbox.telemetry.faults_injected == 1


# --- pytest fixtures (agent_chaos, stream_chaos, graph_auditor), used directly rather
# than through the decorators -- the README documents these as the primary API. ---


def test_agent_chaos_fixture_creates_a_registered_sandbox(agent_chaos) -> None:  # type: ignore[no-untyped-def]
    sb = agent_chaos(ChaosProfile(fault_type=FaultType.TRUNCATED_JSON_TOOL_CALL, trigger_on_call=1))
    assert isinstance(sb, ChaosSandbox)
    mutated = sb.mutate_tool_payload('{"a": 1, "b": 2}')
    assert mutated != '{"a": 1, "b": 2}'


def test_stream_chaos_fixture_creates_an_active_sandbox(stream_chaos) -> None:  # type: ignore[no-untyped-def]
    sb = stream_chaos(
        ChaosProfile(
            fault_type=FaultType.RATE_LIMIT_429, trigger_on_call=1, intercept_all_hosts=True
        )
    )
    assert isinstance(sb, StreamChaosSandbox)
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))
    ) as client:
        resp = client.get("https://api.openai.com/v1/models")
    assert resp.status_code == 429


def test_graph_auditor_fixture_creates_a_working_auditor(graph_auditor) -> None:  # type: ignore[no-untyped-def]
    auditor = graph_auditor(max_steps=3, detect_cycles=False)
    assert isinstance(auditor, StateGraphAuditor)
    auditor.record_step("a")
    auditor.record_step("b")
    auditor.record_step("c")
    with pytest.raises(InfiniteAgentLoopError):
        auditor.record_step("d")


@pytest.mark.asyncio
async def test_async_sse_drop_on_non_streaming_call_raises_from_send() -> None:
    sse_chunks = [b"data: a\n\n", b"data: b\n\n", b"data: [DONE]\n\n"]

    async def async_content() -> AsyncIterator[bytes]:
        for chunk in sse_chunks:
            yield chunk

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=async_content()
        )

    profile = ChaosProfile(
        fault_type=FaultType.SSE_MIDSTREAM_DROP,
        trigger_on_call=1,
        drop_after_chunks=1,
        intercept_all_hosts=True,
    )
    with StreamChaosSandbox(profile) as sb:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            req = client.build_request("POST", "https://api.openai.com/v1/chat/completions")
            with pytest.raises(httpx.RemoteProtocolError):
                await client.send(req)
            assert sb.telemetry.faults_injected == 1
