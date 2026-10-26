"""Background runner for the aiohttp Chaos Proxy."""

import asyncio
import socket
import threading
from collections.abc import Generator
from contextlib import contextmanager

from aiohttp import web

from pytest_agent_chaos.proxy.client import ChaosProxyClient
from pytest_agent_chaos.proxy.server import ProxyState, create_app


def find_free_port() -> int:
    """Finds an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ChaosProxyRunner:
    """Manages the lifecycle of an aiohttp Chaos Proxy server running in a background thread."""

    def __init__(
        self,
        port: int | None = None,
        upstream_base: str = "https://api.openai.com",
    ) -> None:
        self.port = port or find_free_port()
        self.upstream_base = upstream_base
        self.state = ProxyState(upstream_base=upstream_base)
        self.app = create_app(upstream_base=upstream_base, proxy_state=self.state)
        self.loop = asyncio.new_event_loop()
        self.runner: web.AppRunner | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.started = threading.Event()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)

        async def _start() -> None:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, "127.0.0.1", self.port)
            await site.start()
            self.started.set()

        self.loop.run_until_complete(_start())
        self.loop.run_forever()

    def start(self, timeout: float = 5.0) -> None:
        """Starts the background aiohttp proxy server."""
        self.thread.start()
        if not self.started.wait(timeout=timeout):
            raise TimeoutError(
                f"Chaos proxy runner failed to start within {timeout}s on port {self.port}"
            )

    def stop(self) -> None:
        """Stops the proxy and cleans up async resources."""
        if self.runner and self.loop.is_running():

            async def _cleanup() -> None:
                if self.runner:
                    await self.runner.cleanup()

            future = asyncio.run_coroutine_threadsafe(_cleanup(), self.loop)
            try:
                future.result(timeout=2.0)
            except (TimeoutError, RuntimeError, OSError):
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=2.0)

    @property
    def base_url(self) -> str:
        """Returns the base HTTP URL of the running proxy."""
        return f"http://127.0.0.1:{self.port}"


@contextmanager
def run_chaos_proxy(
    port: int | None = None,
    upstream_base: str = "https://api.openai.com",
) -> Generator[ChaosProxyClient, None, None]:
    """Context manager to spawn a Chaos Proxy server and yield a connected client."""
    runner = ChaosProxyRunner(port=port, upstream_base=upstream_base)
    runner.start()
    client = ChaosProxyClient(control_url=runner.base_url)
    try:
        yield client
    finally:
        client.close()
        runner.stop()
