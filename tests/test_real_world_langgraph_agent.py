"""End-to-end integration tests verifying real LangGraph ReAct agents under chaos injection.

Proves that genuine production-grade autonomous agent loops (ChatOpenAI + LangGraph create_react_agent + ToolNode)
catch transport drops and tool argument corruptions, self-heal via multi-turn feedback, and reach completion.
"""

import json

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, create_react_agent

from pytest_agent_chaos.interceptors.llm_stream import StreamChaosSandbox
from pytest_agent_chaos.interceptors.tool_corruptor import ChaosSandbox
from pytest_agent_chaos.models import ChaosProfile, FaultType


@tool
def calculate_vat(gross_amount: float, country_code: str) -> str:
    """Calculates tax based on the gross amount and country code."""
    if not isinstance(gross_amount, (int, float)):
        raise TypeError(f"gross_amount must be float/int, got {type(gross_amount).__name__}")
    rates = {"DE": 0.19, "NL": 0.21, "UK": 0.20}
    rate = rates.get(country_code.upper(), 0.15)
    tax = round(gross_amount * rate, 2)
    return json.dumps({"gross": gross_amount, "vat": tax, "total": round(gross_amount + tax, 2)})


def test_real_langgraph_react_agent_tool_schema_drift_self_healing() -> None:
    """Verifies that a real LangGraph ReAct agent catches tool argument corruption,

    observes the resulting ToolMessage error in state, self-corrects the invocation,
    and terminates successfully.
    """
    tool_profile = ChaosProfile(
        fault_type=FaultType.SCHEMA_TYPE_DRIFT,
        trigger_on_call=1,
        max_graph_steps=6,
    )
    sandbox = ChaosSandbox(tool_profile)
    original_func = calculate_vat.func

    def chaos_wrapped_vat(gross_amount: float, country_code: str) -> str:
        payload = json.dumps({"gross_amount": gross_amount, "country_code": country_code})
        mutated_payload = sandbox.mutate_tool_payload(payload)
        corrupted_args = json.loads(mutated_payload)
        return original_func(**corrupted_args)

    calculate_vat.func = chaos_wrapped_vat

    # Mock multi-turn LLM behavior for deterministic zero-token CI testing:
    # Turn 1: Model requests calculate_vat
    # Turn 2: Model observes ToolMessage error in graph state and self-corrects parameters
    # Turn 3: Model receives clean tool result and generates final synthesis
    call_index = 0

    def mock_openai_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_turn_1",
                                        "type": "function",
                                        "function": {
                                            "name": "calculate_vat",
                                            "arguments": json.dumps(
                                                {"gross_amount": 1000.0, "country_code": "DE"}
                                            ),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        elif call_index == 2:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Tool reported invalid schema type. Retrying with verified numeric float.",
                                "tool_calls": [
                                    {
                                        "id": "call_turn_2",
                                        "type": "function",
                                        "function": {
                                            "name": "calculate_vat",
                                            "arguments": json.dumps(
                                                {"gross_amount": 1000.0, "country_code": "DE"}
                                            ),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        else:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The VAT for 1000 EUR in Germany is 190.0 EUR, for a total of 1190.0 EUR.",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

    client = httpx.Client(transport=httpx.MockTransport(mock_openai_handler))
    llm = ChatOpenAI(model="gpt-4o", api_key="mock-key", http_client=client)

    # Use production resilient ToolNode that catches runtime errors into ToolMessage
    tool_node = ToolNode([calculate_vat], handle_tool_errors=True)
    agent_graph = create_react_agent(llm, tools=tool_node)

    # Execute the autonomous LangGraph state graph loop
    final_state = agent_graph.invoke(
        {"messages": [("user", "Calculate the VAT for 1000 EUR in Germany (DE).")]}
    )

    messages = final_state["messages"]

    # Verify real multi-turn progression across the graph
    # 0: HumanMessage (User prompt)
    # 1: AIMessage (Tool call 1)
    # 2: ToolMessage (Error captured from injected schema drift)
    # 3: AIMessage (Self-correction reasoning + Tool call 2)
    # 4: ToolMessage (Clean tool output)
    # 5: AIMessage (Final response)
    assert len(messages) == 6
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage) and len(messages[1].tool_calls) == 1
    assert isinstance(messages[2], ToolMessage) and "TypeError" in messages[2].content
    assert isinstance(messages[3], AIMessage) and len(messages[3].tool_calls) == 1
    assert isinstance(messages[4], ToolMessage) and "1190.0" in messages[4].content
    assert isinstance(messages[5], AIMessage) and "1190.0 EUR" in messages[5].content

    # Assert chaos telemetry observed recovery
    assert sandbox.telemetry.faults_injected == 1
    assert sandbox.telemetry.total_calls_intercepted == 2
    assert sandbox.telemetry.recovery_succeeded is True


def test_real_langgraph_agent_429_transport_recovery() -> None:
    """Verifies that ChatOpenAI driving a LangGraph agent recovers from HTTP 429 storms."""
    profile = ChaosProfile(
        fault_type=FaultType.RATE_LIMIT_429,
        trigger_on_call=1,
        retry_after_seconds=1,
        target_domains=["api.openai.com"],
    )

    def mock_backend_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Autonomous agent recovered and planned response.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(mock_backend_handler)

    with StreamChaosSandbox(profile) as stream_sandbox:
        client = httpx.Client(transport=transport)
        # Real ChatOpenAI configured with built-in tenacity retries
        llm = ChatOpenAI(model="gpt-4o", api_key="mock-key", http_client=client, max_retries=2)
        agent = create_react_agent(llm, tools=[])

        result = agent.invoke({"messages": [("user", "Summarize pending transactions.")]})

        assert result["messages"][-1].content == "Autonomous agent recovered and planned response."
        assert stream_sandbox.telemetry.faults_injected == 1
        assert stream_sandbox.telemetry.recovery_succeeded is True
        assert stream_sandbox.telemetry.total_retries_observed == 1
        assert stream_sandbox.telemetry.total_calls_intercepted == 2
