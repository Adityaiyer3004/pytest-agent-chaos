"""Property-based tests for the chaos payload mutators.

These assert invariants that must hold for *any* valid input, not just the
hand-picked examples in the other test files -- e.g. "truncation always
shortens the payload" and "schema drift never leaves a bool looking numeric".
"""

import json

from hypothesis import given
from hypothesis import strategies as st

from pytest_agent_chaos.interceptors.tool_corruptor import ChaosSandbox
from pytest_agent_chaos.models import ChaosProfile, FaultType

json_scalars = st.one_of(
    st.booleans(),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=20),
    st.none(),
)

json_values = st.recursive(
    json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=4),
    ),
    max_leaves=10,
)

json_objects = st.dictionaries(
    st.text(min_size=1, max_size=10), json_values, min_size=1, max_size=6
)


def _sandbox(fault: FaultType) -> ChaosSandbox:
    return ChaosSandbox(ChaosProfile(fault_type=fault, trigger_on_call=1))


@given(payload=json_objects)
def test_truncation_always_shortens_or_leaves_minimal_payloads_alone(payload: dict) -> None:
    raw = json.dumps(payload)
    sb = _sandbox(FaultType.TRUNCATED_JSON_TOOL_CALL)
    mutated = sb.mutate_tool_payload(raw)
    assert len(mutated) <= len(raw)
    assert raw.startswith(mutated)
    if len(raw) >= 2:
        assert mutated != raw


@given(payload=json_objects)
def test_schema_drift_preserves_top_level_keys_and_is_valid_json(payload: dict) -> None:
    raw = json.dumps(payload)
    sb = _sandbox(FaultType.SCHEMA_TYPE_DRIFT)
    mutated = sb.mutate_tool_payload(raw)
    out = json.loads(mutated)  # must not raise
    assert set(out.keys()) == set(payload.keys())


def _assert_drifted(original: object, drifted: object) -> None:
    if isinstance(original, bool):
        assert drifted == ["unexpected", "array"]
    elif isinstance(original, (int, float)):
        assert drifted == "CHAOS_INVALID_NUMBER_TYPE"
    elif isinstance(original, str):
        assert drifted is None
    elif isinstance(original, list):
        assert isinstance(drifted, list) and len(drifted) == len(original)
        for o, d in zip(original, drifted, strict=True):
            _assert_drifted(o, d)
    elif isinstance(original, dict):
        assert isinstance(drifted, dict) and set(drifted.keys()) == set(original.keys())
        for k in original:
            _assert_drifted(original[k], drifted[k])
    else:
        assert drifted == original  # None stays None


@given(payload=json_objects)
def test_schema_drift_transforms_every_value_by_type_recursively(payload: dict) -> None:
    raw = json.dumps(payload)
    sb = _sandbox(FaultType.SCHEMA_TYPE_DRIFT)
    mutated = json.loads(sb.mutate_tool_payload(raw))
    for key, original_value in payload.items():
        _assert_drifted(original_value, mutated[key])


@given(payload=json_objects)
def test_hallucinated_params_adds_exactly_two_rogue_keys(payload: dict) -> None:
    raw = json.dumps(payload)
    sb = _sandbox(FaultType.HALLUCINATED_PARAMS)
    mutated = json.loads(sb.mutate_tool_payload(raw))
    assert set(mutated.keys()) == set(payload.keys()) | {
        "__chaos_hallucinated_field__",
        "confidence_score",
    }
    for key, value in payload.items():
        assert mutated[key] == value
