# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this
project adheres to [Semantic Versioning](https://semver.org/) once it reaches
1.0 — see the Versioning section in the README for what's currently covered.

## [Unreleased]

### Fixed

- **SSE drop forced every call to stream, not just the faulted one.** With
  `SSE_MIDSTREAM_DROP` active, a plain (non-streaming) request on a call where
  no fault fires used to come back unread, breaking `.json()`/`.text` on real
  sockets. Now only the call where the fault actually triggers is forced to
  stream, and a non-streaming caller still gets a fully-read response (or the
  drop raised from `send()` itself, matching a real socket).
- **`chaos_inject`/`stream_chaos_inject` decorators silently no-op'd on
  `async def` tests.** The `with`-block exited before the coroutine ran, so a
  429 or schema-drift fault that should have fired never did. Both decorators
  now detect `async def` and wrap accordingly.
- **"Recovery" was true for literally any later call.** `recovery_succeeded`
  was set whenever any call happened after a fault, even to a different
  endpoint or tool, or even when the retry itself failed. It now requires
  evidence: the same endpoint/tool called again and succeeding. This also
  fixed the out-of-process sidecar, which had the identical bug.
- **Boolean schema drift was dead code.** `_drift_value` checked
  `isinstance(val, (int, float))` before `isinstance(val, bool)`; since `bool`
  is a subclass of `int` in Python, every boolean took the numeric branch and
  the dedicated bool branch never ran. Booleans now drift correctly.
- **Probability draws depended on global, shared `random` state.** With
  `probability < 1.0`, whether a fault fired depended on which other chaos
  tests had already run and consumed the shared `random` module — a true
  "deterministic" claim requires an isolated stream. Each `ChaosSandbox` now
  owns a private `random.Random(seed)`; fixtures and decorators derive a
  stable per-test seed from `--chaos-seed` + the test name automatically.
- **`StreamChaosSandbox` exit order could strand the `httpx.Client.send`
  monkeypatch permanently.** Entering two sandboxes and exiting them in
  creation order (exactly what the `stream_chaos` fixture's teardown loop
  did) left the process-wide patch pointed at a dead closure for the rest of
  the test run, with no error. `__exit__` now only restores the patch if it's
  still the one currently installed, warns instead of corrupting it if a
  sandbox is exited out of LIFO order, and the fixture's teardown now unwinds
  in reverse order. Install/restore is also now lock-protected.
- **Cycle detection only caught a hardcoded 2-node ping-pong, and never
  raised.** `StateGraphAuditor` now detects a repeating pattern of any period
  (given enough repeats to be confident), and `raise_on_cycle=True` lets it
  raise `InfiniteAgentLoopError` instead of only flagging `cycle_detected`.
- **Sidecar's mid-stream drop ended the stream cleanly instead of severing
  it.** The fault handler used `return` inside the streaming generator, which
  let Starlette finish the HTTP response normally — downstream clients saw a
  short-but-complete response, not a severed connection. It now raises, which
  aborts the connection mid-body and surfaces as `httpx.RemoteProtocolError`
  on the client, consistent with the in-process interceptor.
- **A test left global state mutated on failure.** `calculate_vat.func` was
  swapped out in `test_real_world_langgraph_agent.py` and never restored;
  it's now wrapped in `try`/`finally`.
- **Dependency floors declared in `pyproject.toml` didn't actually resolve
  together.** `fastapi>=0.100.0` excludes `pydantic==2.0.0`/`2.0.1` (conflicts
  with the declared `pydantic>=2.0.0` floor); `langgraph>=0.2.0` pins
  `langchain-core<0.3` (conflicts with the declared `langchain-core>=0.3.0`
  floor); and `asyncio_default_fixture_loop_scope` (already in use in
  `pyproject.toml`) needs `pytest-asyncio>=0.24`, which itself needs
  `pytest>=8.2`. All floors are now bumped to versions that install and pass
  together, and CI has a dedicated job pinning to exactly these floors so a
  future bump can't silently reintroduce the conflict.

### Changed

- Consolidated the two overlapping out-of-process proxy implementations
  (`proxy/`, aiohttp-based; `sidecar/`, FastAPI-based) into one. Kept
  `sidecar/`: its fault names already matched the core `FaultType` enum, it
  forwards query strings correctly (the aiohttp version dropped them), and
  it's the one the README already documented. `Dockerfile.proxy` →
  `Dockerfile.sidecar`, `docker-compose.test.yml`, and the container
  black-box tests were ported accordingly; the `chaos_proxy` fixture and the
  `aiohttp` dependency are gone.

### Added

- Real-socket regression tests (`tests/test_regressions_real_socket.py`) for
  every fix above — the SSE/stream-forcing bug specifically slipped through
  because the existing tests only used `httpx.MockTransport`, whose responses
  already hold their body in memory.
- Hypothesis property-based tests for the tool-payload mutators
  (`tests/test_property_based.py`): truncation always shortens the payload,
  schema drift transforms every value by type recursively and preserves
  top-level keys, hallucinated-params always adds exactly the two rogue keys.
- Coverage gate (`pytest-cov`, `fail_under = 85` in `pyproject.toml`) and a
  batch of tests (`tests/test_coverage_gaps.py`) closing gaps a coverage run
  turned up: `HALLUCINATED_PARAMS`, `ChaosSandbox.wrap_tool`,
  `StateGraphAuditor.wrap_async_node`, the `CONNECTION_TIMEOUT`/
  `LATENCY_JITTER` transport faults, and the `agent_chaos`/`stream_chaos`/
  `graph_auditor` pytest fixtures themselves (previously only exercised via
  the decorators, never as fixtures).
- CI now also runs the suite under `pytest -n auto` (parallel-safety) and
  against the exact dependency floors declared in `pyproject.toml`, across
  Python 3.10–3.14.

[Unreleased]: https://github.com/Adityaiyer3004/pytest-agent-chaos/compare/v0.1.0...HEAD
