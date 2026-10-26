"""Out-of-process Chaos Proxy sidecar for black-box container & microservice testing."""

import asyncio

from aiohttp import ClientSession, ClientTimeout, web
from pydantic import BaseModel


class ProxyChaosRule(BaseModel):
    """Configuration for a dynamic chaos fault rule in the proxy."""

    match_path: str = ""  # e.g. "/v1/chat/completions" or "/api/v1/tools"
    fault_type: str  # "rate_limit_429", "timeout_504", "mid_stream_drop", "latency"
    trigger_on_call: int = 1
    latency_ms: int = 0
    drop_after_bytes: int = 256
    status_code: int = 429
    active: bool = True


class ProxyState:
    """In-memory telemetry and rule registry for the chaos proxy."""

    def __init__(self, upstream_base: str = "https://api.openai.com") -> None:
        self.rules: list[ProxyChaosRule] = []
        self.call_counts: dict[str, int] = {}
        self.upstream_base: str = upstream_base

    def reset(self) -> None:
        self.rules.clear()
        self.call_counts.clear()


state = ProxyState()


# --- CONTROL PLANE API ---
async def set_rules(request: web.Request) -> web.Response:
    """Configures a chaos fault rule into the proxy."""
    current_state: ProxyState = request.app.get("proxy_state", state)
    payload = await request.json()
    rule = ProxyChaosRule(**payload)
    current_state.rules.append(rule)
    return web.json_response({"status": "configured", "total_rules": len(current_state.rules)})


async def reset_proxy(request: web.Request) -> web.Response:
    """Resets all configured rules and clears call counters."""
    current_state: ProxyState = request.app.get("proxy_state", state)
    current_state.reset()
    return web.json_response({"status": "reset", "active_rules": 0})


async def get_telemetry(request: web.Request) -> web.Response:
    """Returns telemetry of intercepted path counts."""
    current_state: ProxyState = request.app.get("proxy_state", state)
    return web.json_response({"call_counts": current_state.call_counts})


async def set_config(request: web.Request) -> web.Response:
    """Updates dynamic proxy configuration (e.g. upstream_base URL)."""
    current_state: ProxyState = request.app.get("proxy_state", state)
    payload = await request.json()
    if "upstream_base" in payload:
        current_state.upstream_base = payload["upstream_base"]
    return web.json_response({"status": "configured", "upstream_base": current_state.upstream_base})


# --- TRAFFIC INTERCEPTION ENGINE ---
async def proxy_traffic(request: web.Request) -> web.StreamResponse:
    """Intercepts, evaluates faults, and forwards live network traffic to upstream APIs."""
    current_state: ProxyState = request.app.get("proxy_state", state)
    path = request.path

    # Protect control plane endpoints if reached by catch-all
    if path.startswith("/_chaos/"):
        return web.Response(status=404, text="Not Found")

    current_state.call_counts[path] = current_state.call_counts.get(path, 0) + 1
    current_count = current_state.call_counts[path]

    # Evaluate matching rules
    active_rule = next(
        (
            r
            for r in current_state.rules
            if r.active
            and (not r.match_path or r.match_path in path)
            and r.trigger_on_call == current_count
        ),
        None,
    )

    if active_rule:
        # 1. Latency injection
        if active_rule.latency_ms > 0:
            await asyncio.sleep(active_rule.latency_ms / 1000.0)

        # 2. Synthetic Gateway Timeout (504)
        if active_rule.fault_type == "timeout_504":
            return web.json_response(
                {
                    "error": {
                        "message": "Chaos Proxy: Gateway Timeout",
                        "code": 504,
                    }
                },
                status=504,
            )

        # 3. Upstream Rate-Limit (429) with Retry-After
        if active_rule.fault_type == "rate_limit_429":
            return web.json_response(
                {
                    "error": {
                        "message": "Chaos Proxy: Rate limit exceeded",
                        "code": 429,
                    }
                },
                status=429,
                headers={"Retry-After": "2"},
            )

    # Forward to real upstream service if no blocking fault
    target_url = f"{current_state.upstream_base.rstrip('/')}{path}"
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")
    }
    body = await request.read()

    timeout = ClientTimeout(total=60)
    async with (
        ClientSession(timeout=timeout) as session,
        session.request(
            method=request.method, url=target_url, headers=headers, data=body
        ) as upstream_resp,
    ):
        clean_headers = {
            k: v
            for k, v in upstream_resp.headers.items()
            if k.lower() not in ("transfer-encoding", "content-length", "connection")
        }

        # 4. Mid-Stream SSE TCP Connection Severing
        if (
            active_rule
            and active_rule.fault_type == "mid_stream_drop"
            and upstream_resp.status == 200
        ):
            response = web.StreamResponse(
                status=200,
                headers=clean_headers,
            )
            await response.prepare(request)

            bytes_sent = 0
            async for chunk in upstream_resp.content.iter_chunked(128):
                bytes_sent += len(chunk)
                if bytes_sent > active_rule.drop_after_bytes:
                    # Abruptly sever the underlying TCP transport stream
                    transport = request.transport
                    if transport:
                        transport.close()
                    break
                await response.write(chunk)
            return response

        # Standard streaming pass-through
        response = web.StreamResponse(
            status=upstream_resp.status,
            headers=clean_headers,
        )
        await response.prepare(request)
        async for chunk in upstream_resp.content.iter_chunked(1024):
            await response.write(chunk)
        return response


def create_app(
    upstream_base: str = "https://api.openai.com",
    proxy_state: ProxyState | None = None,
) -> web.Application:
    """Builds the aiohttp Application for the Chaos Proxy."""
    app = web.Application()
    app_state = proxy_state or ProxyState(upstream_base=upstream_base)
    app["proxy_state"] = app_state

    # Control Plane Endpoints
    app.router.add_post("/_chaos/rules", set_rules)
    app.router.add_post("/_chaos/reset", reset_proxy)
    app.router.add_get("/_chaos/telemetry", get_telemetry)
    app.router.add_post("/_chaos/config", set_config)

    # Traffic Proxy (Catch-all route)
    app.router.add_route("*", "/{tail:.*}", proxy_traffic)
    return app


def main() -> None:
    """CLI entrypoint for running the chaos proxy."""
    import argparse

    parser = argparse.ArgumentParser(description="pytest-agent-chaos proxy server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default 8080)")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host interface (default 0.0.0.0)"
    )
    parser.add_argument(
        "--upstream", type=str, default="https://api.openai.com", help="Upstream API base URL"
    )
    args = parser.parse_args()

    app = create_app(upstream_base=args.upstream)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
