"""Pytest configuration and test suite fixtures for pytest-agent-chaos."""

import pytest

from pytest_agent_chaos.reporting.terminal import GLOBAL_CHAOS_REGISTRY


@pytest.fixture(scope="session", autouse=True)
def clean_chaos_registry():
    """Ensure global chaos registry is cleared before the test session starts."""
    GLOBAL_CHAOS_REGISTRY.clear()
    yield