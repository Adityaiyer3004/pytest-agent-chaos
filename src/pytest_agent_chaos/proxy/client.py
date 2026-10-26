"""Pytest control client for configuring remote out-of-process chaos sidecars."""

from collections.abc import Generator
from typing import Any, cast

import httpx
import pytest


class ChaosProxyClient:
    """HTTP client for programming dynamic fault rules into the Chaos Proxy."""

    def __init__(self, control_url: str = "http://localhost:8080"):
        self.control_url = control_url.rstrip("/")
        self.client = httpx.Client(timeout=10.0)

    def is_healthy(self) -> bool:
        """Verifies if the chaos proxy sidecar control plane is reachable."""
        try:
            resp = self.client.get(f"{self.control_url}/_chaos/telemetry")
            return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def inject_mid_stream_drop(
        self,
        match_path: str = "/chat/completions",
        drop_after_bytes: int = 128,
        trigger_on_call: int = 1,
    ) -> None:
        """Configures proxy to abruptly sever the TCP socket mid-stream on trigger turn."""
        self.client.post(
            f"{self.control_url}/_chaos/rules",
            json={
                "match_path": match_path,
                "fault_type": "mid_stream_drop",
                "drop_after_bytes": drop_after_bytes,
                "trigger_on_call": trigger_on_call,
            },
        )

    def inject_upstream_429(
        self,
        match_path: str = "/chat/completions",
        trigger_on_call: int = 1,
    ) -> None:
        """Configures proxy to inject an HTTP 429 Rate Limit error with Retry-After header."""
        self.client.post(
            f"{self.control_url}/_chaos/rules",
            json={
                "match_path": match_path,
                "fault_type": "rate_limit_429",
                "trigger_on_call": trigger_on_call,
            },
        )

    def inject_timeout_504(
        self,
        match_path: str = "/chat/completions",
        trigger_on_call: int = 1,
    ) -> None:
        """Configures proxy to inject an HTTP 504 Gateway Timeout error."""
        self.client.post(
            f"{self.control_url}/_chaos/rules",
            json={
                "match_path": match_path,
                "fault_type": "timeout_504",
                "trigger_on_call": trigger_on_call,
            },
        )

    def inject_latency(
        self,
        match_path: str = "",
        latency_ms: int = 500,
        trigger_on_call: int = 1,
    ) -> None:
        """Configures proxy to inject latency delay before forwarding request."""
        self.client.post(
            f"{self.control_url}/_chaos/rules",
            json={
                "match_path": match_path,
                "fault_type": "latency",
                "latency_ms": latency_ms,
                "trigger_on_call": trigger_on_call,
            },
        )

    def set_upstream(self, upstream_base: str) -> None:
        """Sets the upstream target base URL for the chaos proxy."""
        self.client.post(
            f"{self.control_url}/_chaos/config",
            json={"upstream_base": upstream_base},
        )

    def reset(self) -> None:
        """Resets all configured rules and clears telemetry metrics in the sidecar proxy."""
        self.client.post(f"{self.control_url}/_chaos/reset")

    def get_telemetry(self) -> dict[str, Any]:
        """Fetches telemetry metrics from the sidecar proxy."""
        return cast(dict[str, Any], self.client.get(f"{self.control_url}/_chaos/telemetry").json())

    def close(self) -> None:
        """Closes the HTTP client session."""
        self.client.close()


@pytest.fixture
def chaos_proxy() -> Generator[ChaosProxyClient, None, None]:
    """Pytest fixture providing an isolated ChaosProxyClient with pre- and post-test reset."""
    client = ChaosProxyClient()
    try:
        client.reset()
    except (httpx.HTTPError, OSError):
        pass
    yield client
    try:
        client.reset()
    except (httpx.HTTPError, OSError):
        pass
    client.close()
