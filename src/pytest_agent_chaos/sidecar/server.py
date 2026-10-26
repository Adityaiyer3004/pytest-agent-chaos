"""Out-of-process Chaos Reverse Proxy Sidecar Server.

Runs as a standalone L7 proxy service to intercept deployed containers,
FastAPI backends, and external microservice boundaries over real network sockets.
"""

import asyncio
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse


def create_sidecar_app(default_upstream: str = "https://api.openai.com") -> FastAPI:
    """Creates a FastAPI application configured as an L7 Chaos Proxy with control plane APIs."""
    app = FastAPI(title="pytest-agent-chaos-sidecar", docs_url=None, redoc_url=None)

    active_rules: list[dict[str, Any]] = []
    telemetry: dict[str, Any] = {
        "total_calls": 0,
        "faults_injected": 0,
        "retries_observed": 0,
        "recovery_succeeded": False,
    }

    # --------------------------------------------------------------------------
    # Control Plane Endpoints (/_chaos/v1/...)
    # --------------------------------------------------------------------------

    @app.get("/_chaos/v1/health")
    async def health() -> dict[str, str]:
        """Healthcheck endpoint for orchestrators and pytest fixtures."""
        return {"status": "ok", "service": "pytest-agent-chaos-sidecar"}

    @app.get("/_chaos/v1/rules")
    async def get_rules() -> list[dict[str, Any]]:
        """Returns currently active chaos rules."""
        return active_rules

    @app.post("/_chaos/v1/rules")
    async def set_rule(rule: dict[str, Any]) -> dict[str, str]:
        """Configures a new chaos rule in the sidecar proxy."""
        active_rules.append(rule)
        return {"status": "rule_configured"}

    @app.get("/_chaos/v1/telemetry")
    async def get_telemetry() -> dict[str, Any]:
        """Returns recorded metrics across out-of-process container boundaries."""
        return telemetry

    @app.delete("/_chaos/v1/rules")
    async def clear_rules() -> dict[str, str]:
        """Resets proxy to transparent pass-through mode and clears telemetry."""
        active_rules.clear()
        telemetry["total_calls"] = 0
        telemetry["faults_injected"] = 0
        telemetry["retries_observed"] = 0
        telemetry["recovery_succeeded"] = False
        return {"status": "rules_cleared"}

    # --------------------------------------------------------------------------
    # Data Plane Reverse Proxy (Traffic Forwarding + Chaos Injection)
    # --------------------------------------------------------------------------

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def proxy_traffic(request: Request, path: str) -> Response:
        """Intercepts, mutates, and forwards live network traffic to upstream hosts."""
        # Bypass control plane requests
        if path.startswith("_chaos/v1"):
            return JSONResponse({"error": "Not Found"}, status_code=404)

        telemetry["total_calls"] += 1
        call_idx = telemetry["total_calls"]

        # Match active fault rule for current call count (or catch-all)
        active_rule = next(
            (r for r in active_rules if r.get("trigger_on_call") == call_idx),
            None,
        )

        # 1. Fault: HTTP 429 Rate Limit
        if active_rule and active_rule.get("fault_type") in (
            "rate_limit_429",
            "FaultType.RATE_LIMIT_429",
        ):
            telemetry["faults_injected"] += 1
            retry_after = str(active_rule.get("retry_after", 1))
            return Response(
                status_code=429,
                headers={
                    "Retry-After": retry_after,
                    "x-ratelimit-reset-requests": retry_after,
                    "Content-Type": "application/json",
                },
                content=(
                    b'{"error": {"message": "Rate limit exceeded by pytest-agent-chaos sidecar.", '
                    b'"type": "insufficient_quota", "code": "rate_limit_exceeded"}}'
                ),
            )

        # 2. Fault: HTTP 504 Gateway Timeout
        if active_rule and active_rule.get("fault_type") in (
            "gateway_timeout_504",
            "FaultType.GATEWAY_TIMEOUT_504",
        ):
            telemetry["faults_injected"] += 1
            return Response(
                status_code=504,
                headers={"Content-Type": "application/json"},
                content=b'{"error": {"message": "Gateway Timeout from upstream host injected by chaos sidecar.", "code": 504}}',
            )

        # 3. Fault: Latency Jitter
        if active_rule and active_rule.get("fault_type") == "latency_jitter":
            latency_ms = active_rule.get("latency_ms", 0)
            if latency_ms > 0:
                await asyncio.sleep(latency_ms / 1000.0)

        # Resolve upstream target host
        upstream_base = (
            active_rule.get("upstream_host", default_upstream) if active_rule else default_upstream
        ).rstrip("/")
        target_url = f"{upstream_base}/{path}"

        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)

        client = httpx.AsyncClient(timeout=60.0)
        try:
            upstream_req = client.build_request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                params=request.query_params,
            )
            upstream_resp = await client.send(upstream_req, stream=True)

            # Mark recovery if previous call had injected faults
            if telemetry["faults_injected"] > 0 and not active_rule:
                telemetry["retries_observed"] += 1
                telemetry["recovery_succeeded"] = True

            # 4. Fault: Mid-Stream SSE Disconnect / Slicing
            is_stream_drop = active_rule and active_rule.get("fault_type") in (
                "sse_midstream_drop",
                "drop_mid_stream",
                "FaultType.SSE_MIDSTREAM_DROP",
            )
            drop_after = active_rule.get("drop_after_chunks", 1) if active_rule else 1
            drop_mid_chunk = active_rule.get("drop_mid_chunk", False) if active_rule else False

            async def chaos_stream_generator():
                chunk_count = 0
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        chunk_count += 1
                        if is_stream_drop and chunk_count > drop_after:
                            telemetry["faults_injected"] += 1
                            if drop_mid_chunk and len(chunk) > 1:
                                yield chunk[: max(1, len(chunk) // 2)]
                            await upstream_resp.aclose()
                            await client.aclose()
                            return  # Sever connection abruptly without ending stream
                        yield chunk
                finally:
                    await upstream_resp.aclose()
                    await client.aclose()

            resp_headers = dict(upstream_resp.headers)
            resp_headers.pop("content-length", None)

            return StreamingResponse(
                chaos_stream_generator(),
                status_code=upstream_resp.status_code,
                headers=resp_headers,
            )
        except Exception:
            await client.aclose()
            raise

    return app
