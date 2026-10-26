"""Remote client for controlling the out-of-process Chaos Proxy Sidecar."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import httpx


class RemoteChaosClient:
    """Client for programming chaos rules into a running sidecar proxy."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=10.0)

    def is_healthy(self) -> bool:
        """Checks if the chaos sidecar proxy is online and responsive."""
        try:
            resp = self._http.get("/_chaos/v1/health")
            return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def add_rule(
        self,
        fault_type: str,
        trigger_on_call: int = 1,
        retry_after: int = 1,
        drop_after_chunks: int = 1,
        drop_mid_chunk: bool = False,
        latency_ms: int = 0,
        upstream_host: str | None = None,
    ) -> dict[str, Any]:
        """Programs a chaos fault rule into the sidecar proxy."""
        rule_payload: dict[str, Any] = {
            "fault_type": fault_type,
            "trigger_on_call": trigger_on_call,
            "retry_after": retry_after,
            "drop_after_chunks": drop_after_chunks,
            "drop_mid_chunk": drop_mid_chunk,
            "latency_ms": latency_ms,
        }
        if upstream_host:
            rule_payload["upstream_host"] = upstream_host

        resp = self._http.post("/_chaos/v1/rules", json=rule_payload)
        resp.raise_for_status()
        return resp.json()

    def get_telemetry(self) -> dict[str, Any]:
        """Queries recorded metrics from the sidecar proxy."""
        resp = self._http.get("/_chaos/v1/telemetry")
        resp.raise_for_status()
        return resp.json()

    def clear_rules(self) -> dict[str, Any]:
        """Resets the proxy sidecar to clean transparent forwarding."""
        resp = self._http.delete("/_chaos/v1/rules")
        resp.raise_for_status()
        return resp.json()

    @contextmanager
    def inject_fault(
        self,
        fault_type: str,
        trigger_on_call: int = 1,
        retry_after: int = 1,
        drop_after_chunks: int = 1,
        drop_mid_chunk: bool = False,
        latency_ms: int = 0,
        upstream_host: str | None = None,
    ) -> Generator["RemoteChaosClient", None, None]:
        """Context manager to inject a transient chaos rule and clear upon exit."""
        self.add_rule(
            fault_type=fault_type,
            trigger_on_call=trigger_on_call,
            retry_after=retry_after,
            drop_after_chunks=drop_after_chunks,
            drop_mid_chunk=drop_mid_chunk,
            latency_ms=latency_ms,
            upstream_host=upstream_host,
        )
        try:
            yield self
        finally:
            self.clear_rules()

    def close(self) -> None:
        """Closes the underlying HTTP client."""
        self._http.close()
