"""Self-tests verifying the terminal summary reporting table and resilience index calculations."""

from unittest.mock import MagicMock

from pytest_agent_chaos.models import ChaosTelemetry, FaultType
from pytest_agent_chaos.reporting.terminal import (
    ChaosTelemetryRegistry,
    render_chaos_summary_table,
)


def test_render_chaos_summary_table_output():
    """Verifies that terminal table renders correct header, rows, and resilience index."""
    registry = ChaosTelemetryRegistry()

    # Record 1: Recovered 429
    t1 = ChaosTelemetry(
        test_name="test_openai_429_backoff",
        fault_type=FaultType.RATE_LIMIT_429,
        total_calls_intercepted=3,
        faults_injected=1,
        total_retries_observed=2,
        recovery_succeeded=True,
    )
    # Record 2: Recovered SSE Midstream
    t2 = ChaosTelemetry(
        test_name="test_anthropic_sse_cutoff",
        fault_type=FaultType.SSE_MIDSTREAM_DROP,
        total_calls_intercepted=2,
        faults_injected=1,
        total_retries_observed=1,
        recovery_succeeded=True,
    )
    registry.register(t1)
    registry.register(t2)

    mock_reporter = MagicMock()
    render_chaos_summary_table(mock_reporter, registry=registry)

    # Check separator called
    assert mock_reporter.write_sep.called
    # Check lines written
    calls_text = "".join(
        call.args[0] for call in mock_reporter.write_line.call_args_list if call.args
    )

    assert "Test Case" in calls_text
    assert "Fault Type" in calls_text
    assert "Chaos Resilience Index: 100.0%" in calls_text
    assert "Injected Faults Handled: 2/2" in calls_text


def test_empty_registry_writes_nothing():
    """Verifies that if no chaos tests ran, no table is written."""
    registry = ChaosTelemetryRegistry()
    mock_reporter = MagicMock()
    render_chaos_summary_table(mock_reporter, registry=registry)
    assert not mock_reporter.write_sep.called
