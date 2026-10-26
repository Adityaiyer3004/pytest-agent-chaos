"""Server runner and thread manager for running the Chaos Proxy Sidecar."""

import socket
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

import uvicorn

from pytest_agent_chaos.sidecar.client import RemoteChaosClient
from pytest_agent_chaos.sidecar.server import create_sidecar_app


def find_free_port() -> int:
    """Finds an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return int(s.getsockname()[1])


class ChaosSidecarServer:
    """Manages starting and stopping the sidecar proxy in a background thread."""

    def __init__(
        self, port: int | None = None, upstream_host: str = "https://api.openai.com"
    ) -> None:
        self.port = port or find_free_port()
        self.upstream_host = upstream_host
        self.app = create_sidecar_app(default_upstream=self.upstream_host)
        self.config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(self.config)
        self.thread: threading.Thread | None = None
        self.client = RemoteChaosClient(base_url=f"http://127.0.0.1:{self.port}")

    def start(self, timeout: float = 5.0) -> RemoteChaosClient:
        """Starts the sidecar proxy in a background thread and waits for healthcheck."""
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.client.is_healthy():
                return self.client
            time.sleep(0.05)

        raise RuntimeError(f"Timed out waiting for Chaos Sidecar to start on port {self.port}")

    def stop(self) -> None:
        """Stops the running uvicorn server."""
        self.server.should_exit = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.client.close()


@contextmanager
def run_chaos_sidecar(
    upstream_host: str = "https://api.openai.com",
    port: int | None = None,
) -> Generator[RemoteChaosClient, None, None]:
    """Context manager that starts a chaos proxy sidecar and yields the remote control client."""
    server = ChaosSidecarServer(port=port, upstream_host=upstream_host)
    client = server.start()
    try:
        yield client
    finally:
        server.stop()
