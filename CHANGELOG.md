# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this
project adheres to [Semantic Versioning](https://semver.org/) once it reaches
1.0 — see the Versioning section in the README for what's currently covered.

## [Unreleased]

## [0.1.2] - 2026-10-28

### Fixed

- **`from pytest_agent_chaos import *` crashed with
  `ModuleNotFoundError: No module named 'uvicorn'` on a plain install.**
  `__all__` listed `ChaosSidecarServer`, `RemoteChaosClient`, and
  `run_chaos_sidecar` — the lazy sidecar exports added in 0.1.1. A star
  import resolves names in `__all__` directly, bypassing the module's
  `__getattr__` lazy loader entirely, so it forced the `fastapi`/`uvicorn`
  import eagerly for every star-importer. Removed the three names from
  `__all__`; explicit imports (`from pytest_agent_chaos import
  ChaosSidecarServer`) still resolve lazily as before. Covered by a new
  `test_star_import_does_not_require_fastapi_or_uvicorn` regression test.
- **`LATENCY_JITTER` faults slept but were never counted or recorded.**
  `StreamChaosSandbox` incremented `faults_injected` and called
  `_note_fault` for every other fault type, but the latency-jitter branch
  only did `time.sleep`/`asyncio.sleep` and returned — so a test run that
  only ever hit latency jitter reported zero faults injected.
- **The sidecar tracked faulted endpoints by path alone, not by method +
  path.** `faulted_paths` was a `set[str]` keyed on the request path only,
  so a fault on one HTTP method could be "recovered" by an unrelated
  successful call to a different method on the same path. Now keyed by
  `(method, path)`, matching the in-process sandbox's endpoint key.
- **`ChaosSandbox`'s recovery check treated all non-dict JSON payloads as
  interchangeable.** `_object_keys` returned an empty `frozenset()` for any
  top-level JSON value that wasn't an object (a list, a string, a number),
  so a corrupted list payload could be marked "recovered" by a later call
  whose payload was a totally different type, as long as neither was a
  dict. Replaced with `_object_shape`, which also compares the JSON type
  name for non-dict payloads.
- **`ChaosSandbox.wrap_tool` silently skipped interception for
  non-JSON-serializable kwargs** (a `set`, `bytes`, or custom object),
  with no indication why — the call still ran, but didn't count toward
  `total_calls_intercepted`, so `trigger_on_call` could fire on an
  unexpected call. Now emits a `UserWarning` naming the tool and explains
  the call wasn't intercepted.

## [0.1.1] - 2026-10-28

### Fixed

- **The terminal "Chaos Resilience Index" ignored `recovery_succeeded` and
  recomputed its own, broken definition of "recovered."** The reporting
  layer marked a test `RECOVERED` whenever `faults_injected > 0 and
  total_retries_observed > 0` — true for any retry attempt, regardless of
  whether it actually succeeded — `or`-ed with the correct
  `recovery_succeeded`. This is the exact headline-metric bug from the
  original review (an agent that retries and fails anyway was reported as
  recovered, with a 100% Resilience Index), still present because this file
  was never touched while fixing `recovery_succeeded` itself in the
  sandboxes and sidecar. Now reports `RECOVERED` only when
  `recovery_succeeded` is actually `True`.
- **The sidecar's SSE-drop fault had the same premature-fault-marking bug
  already fixed in the in-process `StreamChaosSandbox` (see below), just
  never ported here.** `faulted_paths.add(path)` ran as soon as the active
  rule for a call was an SSE-drop rule, before the streaming generator had
  confirmed the drop would actually fire. If the upstream response had fewer
  chunks than `drop_after_chunks`, nothing dropped, but the next successful
  call to that path was still recorded as a recovery. The mark now happens
  only where `faults_injected` is incremented, inside the generator, at the
  point the drop actually occurs.
- **A plain `pip install pytest-agent-chaos` broke the plugin entirely.**
  `pytest_agent_chaos/__init__.py` unconditionally imported the sidecar
  module, which imports `fastapi`/`uvicorn` — declared only in the
  `dev`/`sidecar` extras, not core dependencies. Since pytest's `pytest11`
  entry point loads this package on every test run, that `ImportError` was
  fatal to every test, not just to sidecar usage. Found by building the
  wheel and installing it into a clean venv to actually try it. The sidecar
  exports (`ChaosSidecarServer`, `RemoteChaosClient`, `run_chaos_sidecar`)
  are now loaded lazily via a module `__getattr__`, matching how the
  `chaos_sidecar` fixture and the README's own sidecar example already
  import it.
- **SSE drop marked the endpoint "faulted" before confirming the drop fired.**
  If `drop_after_chunks` exceeded the real chunk count, nothing ever dropped
  (`faults_injected` stayed `0`), but the next successful call to that
  endpoint was still counted as a recovery. The fault is now only recorded
  via a callback invoked from the exact point the byte stream itself raises,
  for both the sync and async paths.
- **`StreamChaosSandbox` ignored `profile.probability` entirely.** Only
  `ChaosSandbox` (tool corruptor) ever consulted it; `probability=0.0` still
  injected on the trigger call for transport faults. `_get_active_fault` now
  draws from the same private `random.Random` the sandbox already has.
- **`stream_chaos`/`stream_chaos_inject` never seeded the sandbox they
  create**, unlike the tool-corruption fixture/decorator. With
  `probability < 1.0`, this meant `StreamChaosSandbox`'s RNG fell back to
  `random.Random(None)` (OS entropy) rather than a derived per-test seed, so
  results varied between otherwise-identical runs — directly undermining the
  "deterministic" part of the pitch. Both now route through `_seeded()`. As a
  second line of defense, `ChaosSandbox`/`StreamChaosSandbox` constructed
  directly (bypassing the fixtures/decorators) now fall back to
  `models.DEFAULT_SEED` instead of `None`, so even raw usage is reproducible.
- **Cycle detection still matched a single node repeated many times in a
  row.** Excluding period 1 wasn't enough: a degenerate period-2 block made
  of the same node twice (e.g. `[A, A]` repeated 3 times, from `call_llm`
  called 6 times straight) still looked like a 2-node cycle. Any block where
  every element is identical is now skipped regardless of period length.
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
- CI now also runs the suite under `pytest -n auto` (parallel-safety), enforces
  `ruff format --check`, and runs against the exact dependency floors declared
  in `pyproject.toml`, across Python 3.10–3.14.

[Unreleased]: https://github.com/Adityaiyer3004/pytest-agent-chaos/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/Adityaiyer3004/pytest-agent-chaos/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Adityaiyer3004/pytest-agent-chaos/compare/v0.1.0...v0.1.1
