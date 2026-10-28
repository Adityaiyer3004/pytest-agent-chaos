"""Regression tests for the in-process fixes shipped in 0.1.2.

Each test fails against the pre-0.1.2 code. The sidecar's (method, path) fix
lives in test_sidecar_proxy.py, next to the fixtures it needs.
"""

import httpx
import pytest

from pytest_agent_chaos.interceptors.llm_stream import StreamChaosSandbox
from pytest_agent_chaos.interceptors.tool_corruptor import ChaosSandbox
from pytest_agent_chaos.models import ChaosProfile, FaultType


def _ok_transport() -> httpx.MockTransport:
    return httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))


def _jitter_profile() -> ChaosProfile:
    return ChaosProfile(
        fault_type=FaultType.LATENCY_JITTER,
        trigger_on_call=1,
        latency_ms=5,
        intercept_all_hosts=True,
    )


# --- LATENCY_JITTER faults slept but were never counted or recorded ---


def test_latency_jitter_is_counted_and_retry_is_recovery() -> None:
    with (
        StreamChaosSandbox(_jitter_profile()) as sb,
        httpx.Client(transport=_ok_transport()) as client,
    ):
        assert client.get("https://api.openai.com/v1/models").status_code == 200
        assert sb.telemetry.faults_injected == 1
        assert not sb.telemetry.recovery_succeeded

        assert client.get("https://api.openai.com/v1/models").status_code == 200
        assert sb.telemetry.total_retries_observed == 1
        assert sb.telemetry.recovery_succeeded


async def test_latency_jitter_async_is_counted_and_retry_is_recovery() -> None:
    with StreamChaosSandbox(_jitter_profile()) as sb:
        async with httpx.AsyncClient(transport=_ok_transport()) as client:
            assert (await client.get("https://api.openai.com/v1/models")).status_code == 200
            assert sb.telemetry.faults_injected == 1

            assert (await client.get("https://api.openai.com/v1/models")).status_code == 200
            assert sb.telemetry.recovery_succeeded


# --- Recovery treated all non-dict JSON payloads as interchangeable ---


def _truncation_sandbox() -> ChaosSandbox:
    return ChaosSandbox(
        ChaosProfile(fault_type=FaultType.TRUNCATED_JSON_TOOL_CALL, trigger_on_call=1)
    )


@pytest.mark.parametrize("retry", ['"a string"', "42", "null", '{"k": 1}', "{}"])
def test_list_payload_is_not_recovered_by_a_different_json_type(retry: str) -> None:
    sb = _truncation_sandbox()
    sb.mutate_tool_payload('["INV-1", "INV-2"]')
    sb.mutate_tool_payload(retry)
    assert sb.telemetry.total_retries_observed == 1
    assert not sb.telemetry.recovery_succeeded


def test_list_payload_is_recovered_by_a_list_retry() -> None:
    sb = _truncation_sandbox()
    sb.mutate_tool_payload('["INV-1", "INV-2"]')
    sb.mutate_tool_payload('["INV-1", "INV-2"]')
    assert sb.telemetry.recovery_succeeded


# --- wrap_tool silently skipped interception for non-JSON-serializable kwargs ---


def test_wrap_tool_warns_on_unserializable_kwargs_and_still_calls_tool() -> None:
    sb = ChaosSandbox(ChaosProfile(fault_type=FaultType.HALLUCINATED_PARAMS, trigger_on_call=1))
    received: list[dict[str, object]] = []

    def lookup(**kwargs: object) -> str:
        received.append(kwargs)
        return "ok"

    wrapped = sb.wrap_tool(lookup)
    with pytest.warns(UserWarning, match=r"could not JSON-serialize kwargs for 'lookup'"):
        assert wrapped(tags={"a", "b"}) == "ok"

    assert received == [{"tags": {"a", "b"}}]
    assert sb.telemetry.total_calls_intercepted == 0
    assert sb.telemetry.faults_injected == 0
