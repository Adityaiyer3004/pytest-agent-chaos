"""Custom terminal summary reporting for pytest-agent-chaos.

Renders an enterprise-grade terminal summary table displaying injected fault types,
intercepted transport/tool calls, retry backoffs observed, and the Agent Resilience Index.
"""

from _pytest.terminal import TerminalReporter

from pytest_agent_chaos.models import ChaosTelemetry


class ChaosTelemetryRegistry:
    """Session-level registry accumulating chaos telemetry records for summary reporting."""

    def __init__(self) -> None:
        self.records: list[ChaosTelemetry] = []

    def register(self, telemetry: ChaosTelemetry) -> None:
        """Registers a telemetry instance for end-of-session reporting."""
        self.records.append(telemetry)

    def clear(self) -> None:
        """Resets all records."""
        self.records.clear()


# Global singleton registry for the active pytest session
GLOBAL_CHAOS_REGISTRY = ChaosTelemetryRegistry()


def render_chaos_summary_table(
    terminalreporter: TerminalReporter,
    registry: ChaosTelemetryRegistry | None = None,
) -> None:
    """Formats and prints the Agent Chaos Resilience Summary table to the pytest terminal.

    Args:
        terminalreporter: The pytest terminal reporter hook instance.
        registry: The telemetry registry to pull records from (defaults to GLOBAL_CHAOS_REGISTRY).
    """
    reg = registry or GLOBAL_CHAOS_REGISTRY
    records = [r for r in reg.records if r.faults_injected > 0 or r.total_calls_intercepted > 0]

    if not records:
        return

    terminalreporter.write_sep("=", "Agent Chaos Resilience Summary", bold=True, cyan=True)

    header = f"{'Test Case':<36} {'Fault Type':<25} {'Calls':<8} {'Faults':<8} {'Retries':<8} {'Resilience'}"
    terminalreporter.write_line(header, bold=True)
    terminalreporter.write_line("-" * 98)

    total_faults = 0
    total_recovered = 0
    total_calls = 0

    for item in records:
        test_title = (item.test_name or "anonymous_agent_test")[-34:]
        fault_name = item.fault_type.value if item.fault_type else "unknown"
        calls_str = str(item.total_calls_intercepted)
        faults_str = str(item.faults_injected)
        retries_str = str(item.total_retries_observed)

        total_faults += item.faults_injected
        total_calls += item.total_calls_intercepted

        if item.recovery_succeeded or (
            item.faults_injected > 0 and item.total_retries_observed > 0
        ):
            resilience_status = "RECOVERED"
            res_color = {"green": True, "bold": True}
            total_recovered += 1
        elif item.faults_injected == 0:
            resilience_status = "PASSTHROUGH"
            res_color = {"cyan": True}
        else:
            resilience_status = "FAILED / UNRECOVERED"
            res_color = {"red": True, "bold": True}

        line = f"{test_title:<36} {fault_name:<25} {calls_str:<8} {faults_str:<8} {retries_str:<8} "
        terminalreporter.write(line)
        terminalreporter.write_line(resilience_status, **res_color)

    terminalreporter.write_line("-" * 98)

    recovery_rate = (total_recovered / total_faults * 100.0) if total_faults > 0 else 100.0
    summary_metric = (
        f"Chaos Resilience Index: {recovery_rate:.1f}% | "
        f"Injected Faults Handled: {total_recovered}/{total_faults} | "
        f"Total Calls Intercepted: {total_calls}"
    )

    if recovery_rate >= 80.0:
        terminalreporter.write_line(summary_metric, green=True, bold=True)
    elif recovery_rate >= 50.0:
        terminalreporter.write_line(summary_metric, yellow=True, bold=True)
    else:
        terminalreporter.write_line(summary_metric, red=True, bold=True)

    terminalreporter.write_sep("=", "", cyan=True)
