"""Integration test verifying LangGraph / cyclic multi-agent graph state transition audits."""

import pytest

from pytest_agent_chaos.exceptions import InfiniteAgentLoopError
from pytest_agent_chaos.interceptors.state_graph import StateGraphAuditor


def test_state_graph_infinite_loop_guardrail():
    """Verifies that cyclic node loops raise InfiniteAgentLoopError when exceeding budget."""
    auditor = StateGraphAuditor(max_steps=6, detect_cycles=True)

    # Simulated cyclic multi-agent router loop
    def run_agent_loop():
        nodes = ["planner", "tool_executor", "synthesizer"]
        idx = 0
        while True:
            current_node = nodes[idx % len(nodes)]
            auditor.record_step(current_node)
            idx += 1

    with pytest.raises(InfiniteAgentLoopError) as exc_info:
        run_agent_loop()

    assert "Agent graph exceeded maximum step budget (6)" in str(exc_info.value)
    assert auditor.telemetry.step_count == 7
    assert auditor.telemetry.exhausted is True


def test_state_graph_successful_termination_within_budget():
    """Verifies that an agent completing within budget records clean execution telemetry."""
    auditor = StateGraphAuditor(max_steps=10, detect_cycles=True)

    @auditor.wrap_node("input_parser")
    def step_parser(data: str) -> str:
        return data.strip().lower()

    @auditor.wrap_node("sql_retriever")
    def step_sql(query: str) -> list[str]:
        return ["row_1", "row_2"]

    @auditor.wrap_node("output_formatter")
    def step_format(rows: list[str]) -> str:
        return f"Found {len(rows)} rows"

    # Execute workflow
    p = step_parser("  SELECT * FROM test  ")
    rows = step_sql(p)
    out = step_format(rows)

    assert out == "Found 2 rows"
    assert auditor.telemetry.step_count == 3
    assert auditor.telemetry.exhausted is False
    assert auditor.telemetry.cycle_detected is False
    assert auditor.telemetry.node_transitions == [
        "input_parser",
        "sql_retriever",
        "output_formatter",
    ]


def test_repeated_cycle_pattern_detection():
    """Verifies heuristic cycle detection on repeated alternating node executions."""
    auditor = StateGraphAuditor(max_steps=15, detect_cycles=True)

    # Alternating 2-node cycle: [agent, tool, agent, tool, agent, tool]
    for _ in range(3):
        auditor.record_step("researcher_agent")
        auditor.record_step("search_tool")

    assert auditor.telemetry.cycle_detected is True
    assert auditor.telemetry.step_count == 6
