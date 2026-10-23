# pytest-agent-chaos ⚡🤖

[![CI](https://github.com/Adityaiyer3004/pytest-agent-chaos/actions/workflows/ci.yml/badge.svg)](https://github.com/Adityaiyer3004/pytest-agent-chaos/actions)
[![PyPI version](https://img.shields.io/pypi/v/pytest-agent-chaos.svg)](https://pypi.org/project/pytest-agent-chaos/)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest-agent-chaos.svg)](https://pypi.org/project/pytest-agent-chaos/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type_checked-mypy-blue.svg)](https://github.com/python/mypy)

> **Deterministic chaos engineering and fault-injection plugin for multi-agent LLM workflows, tool-calling pipelines, and state graph runtimes.**

---

## 💡 Why `pytest-agent-chaos`?

Most AI testing libraries focus on probabilistic LLM-as-a-judge assertions (like Ragas or DeepEval). While evaluating semantic accuracy is important, **real-world multi-agent systems crash at the systems and transport layer**:

* ⚡ **Mid-Stream Network Drops**: LLM providers sever SSE streams mid-generation during peak load or token limits.
* 🛑 **429 Rate-Limit Storms**: Cascading retries cause agent graphs to enter infinite loops and exhaust step budgets.
* 🧩 **Hallucinated & Truncated Tool Payloads**: LLMs omit closing brackets or mutate data types, crashing downstream Pydantic parsers.
* 🔁 **Cyclic Deadlocks**: Multi-agent router graphs get stuck ping-ponging between planner and tool nodes.

`pytest-agent-chaos` provides **deterministic, zero-token fault injection** directly inside your `pytest` suite. Verify agent self-correction, circuit breakers, and fallback routing in CI/CD without spending a penny on API calls.

---

## 🏛️ Architecture Overview

`pytest-agent-chaos` automatically hooks into pytest via the native `pytest11` entrypoint and intercepts execution across three distinct layers:

```
+-----------------------------------------------------------------------------------+
|                                  PYTEST RUNNER                                    |
+-----------------------------------------------------------------------------------+
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
   +──────────────────────────+ +───────────────────+ +─────────────────────────+
   |  Transport Interceptor   | |   Tool Corruptor  | |   State Graph Auditor   |
   |     (httpx / SSE)        | |  (Pydantic/Tools) | |    (LangGraph / Loops)  |
   +──────────────────────────+ +───────────────────+ +─────────────────────────+
                 │                       │                       │
                 ▼                       ▼                       ▼
     • Mid-stream drop (SSE)       • JSON truncation       • Cycle detection
     • 429 Retry-After storms      • Type schema drift     • Max step budget (Nmax)
     • 504 Gateway timeouts        • Hallucinated fields   • Execution tick metrics
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         ▼
                 +────────────────────────────────---------------+
                 |       Agent Resilience Summary Reporter       |
                 |  Resilience Index (%) | Injected / Recovered  |
                 +────────────────────────────────---------------+
```

---

## 🚀 Quickstart

### 1. Installation

```bash
pip install pytest-agent-chaos
```

### 2. Test SSE Stream Drops & 429 Recovery (Zero Tokens)

```python
import httpx
import pytest
from pytest_agent_chaos import FaultType, stream_chaos_inject

@stream_chaos_inject(
    fault=FaultType.SSE_MIDSTREAM_DROP,
    trigger_on_call=1,
    drop_after_chunks=2,
)
def test_agent_recovers_from_sse_midstream_drop(stream_sandbox):
    """Simulates OpenAI severing an SSE stream after 2 chunks; verifies resilient retry."""
    # When your agent calls OpenAI/Anthropic via httpx, Call #1 will trigger RemoteProtocolError.
    # On Call #2, the agent retries and completes cleanly.
    ...
```

### 3. Test Tool-Calling Parser Resilience

```python
from pydantic import BaseModel, ConfigDict
import pytest
from pytest_agent_chaos import FaultType, chaos_inject
from pytest_agent_chaos.interceptors.tool_corruptor import ChaosSandbox

class ExtractInvoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invoice_id: str
    total_amount: float

@chaos_inject(
    fault=FaultType.TRUNCATED_JSON_TOOL_CALL,
    trigger_on_call=1,
    max_graph_steps=5,
)
def test_invoice_extractor_recovers_from_truncated_tool_json(chaos_sandbox: ChaosSandbox):
    raw_tool_args = '{"invoice_id": "INV-9021", "total_amount": 450.00}'

    # Call 1: Mid-string truncation simulates max_tokens cutoff
    corrupted = chaos_sandbox.mutate_tool_payload(raw_tool_args)
    assert not corrupted.endswith("}")

    # Call 2: Agent self-correction loop emits valid JSON
    clean = chaos_sandbox.mutate_tool_payload(raw_tool_args)
    validated = ExtractInvoice.model_validate_json(clean)
    assert validated.invoice_id == "INV-9021"
    assert chaos_sandbox.telemetry.recovery_succeeded
```

### 4. Guardrail LangGraph State Machines Against Infinite Loops

```python
import pytest
from pytest_agent_chaos.exceptions import InfiniteAgentLoopError
from pytest_agent_chaos.interceptors.state_graph import StateGraphAuditor

def test_multi_agent_graph_loop_guardrail():
    auditor = StateGraphAuditor(max_steps=8, detect_cycles=True)

    @auditor.wrap_node("planner")
    def planner_node(state):
        return {"action": "call_tool"}

    @auditor.wrap_node("tool_node")
    def tool_node(state):
        return {"action": "replan"}

    # Run the cycle: auditor raises InfiniteAgentLoopError when exceeding max_steps
    with pytest.raises(InfiniteAgentLoopError):
        state = {}
        while True:
            state = planner_node(state)
            state = tool_node(state)
```

---

## 📊 Terminal Summary & Chaos Resilience Index

At the end of every test session, `pytest-agent-chaos` prints an actionable summary table:

```text
======================== Agent Chaos Resilience Summary ========================
Test Case                            Fault Type                Calls    Faults   Retries  Resilience
--------------------------------------------------------------------------------------------------
test_openai_429_backoff              rate_limit_429            3        1        2        RECOVERED
test_anthropic_sse_cutoff            sse_midstream_drop        2        1        1        RECOVERED
test_invoice_schema_drift            schema_type_drift         2        1        1        RECOVERED
test_cyclic_graph_guardrail          state_graph_budget        8        1        0        PASSTHROUGH
--------------------------------------------------------------------------------------------------
Chaos Resilience Index: 100.0% | Injected Faults Handled: 3/3 | Total Calls Intercepted: 15
=======================================  =======================================
```

---

## 🛠️ Fault Types Supported

| Fault Identifier | Target Layer | Description |
| :--- | :--- | :--- |
| `RATE_LIMIT_429` | HTTP Transport | Injects HTTP 429 with configurable `Retry-After` header. |
| `GATEWAY_TIMEOUT_504` | HTTP Transport | Injects HTTP 504 Gateway Timeout from LLM backend. |
| `CONNECTION_TIMEOUT` | HTTP Transport | Deterministically raises `httpx.ConnectTimeout`. |
| `SSE_MIDSTREAM_DROP` | SSE Stream | Drops connection after $N$ chunks with `httpx.RemoteProtocolError`. |
| `LATENCY_JITTER` | Transport / Node | Injects synthetic latency spikes in milliseconds. |
| `TRUNCATED_JSON_TOOL_CALL` | Tool Parser | Slices tool arguments mid-JSON to simulate token cutoffs. |
| `SCHEMA_TYPE_DRIFT` | Tool Parser | Flips numeric types to strings and sets required keys to `None`. |
| `HALLUCINATED_PARAMS` | Tool Parser | Injects unexpected keys to verify `extra="forbid"` guardrails. |

---

## ⚙️ CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--chaos-seed` | `42` | Random seed for deterministic, reproducible fault ordering. |
| `--chaos-disable-report` | `False` | Disables printing the Agent Chaos Resilience Summary table. |

---

## 🧪 Development & Testing

```bash
# Clone the repository
git clone https://github.com/Adityaiyer3004/pytest-agent-chaos.git
cd pytest-agent-chaos

# Install dev dependencies
pip install -e ".[dev]"

# Run test suite
pytest
```

---

## 📄 License

Distributed under the terms of the [MIT License](LICENSE).
