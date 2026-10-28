"""Packaging regression: importing pytest_agent_chaos must not require fastapi/uvicorn.

Those are declared only in the dev/sidecar extras in pyproject.toml, not core
dependencies -- but pytest's pytest11 entry point imports this package on every
test run regardless of which features a user actually uses. An eager import of
the sidecar module here would break a plain `pip install pytest-agent-chaos`
entirely, not just sidecar usage. Run in subprocesses: by the time this test
file executes, other test modules in the same session have likely already
imported pytest_agent_chaos.sidecar directly, which would make an in-process
sys.modules check meaningless.
"""

import subprocess
import sys


def test_core_import_does_not_pull_in_sidecar_eagerly() -> None:
    code = (
        "import sys\n"
        "import pytest_agent_chaos\n"
        "assert 'pytest_agent_chaos.sidecar' not in sys.modules, ("
        "'pytest_agent_chaos/__init__.py must not eagerly import the sidecar module'"
        ")\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_core_import_succeeds_without_fastapi_or_uvicorn() -> None:
    # Simulates a plain `pip install pytest-agent-chaos` install, where fastapi and
    # uvicorn (sidecar-only dependencies) are not installed at all.
    code = (
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _blocked(name, *a, **k):\n"
        "    if name in ('fastapi', 'uvicorn'):\n"
        "        raise ModuleNotFoundError(f'simulated: no module named {name!r}')\n"
        "    return _real_import(name, *a, **k)\n"
        "builtins.__import__ = _blocked\n"
        "import pytest_agent_chaos\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_star_import_does_not_require_fastapi_or_uvicorn() -> None:
    # __all__ must not list the lazy sidecar exports (ChaosSidecarServer,
    # RemoteChaosClient, run_chaos_sidecar): `import *` resolves names via __all__
    # directly, bypassing __getattr__'s lazy load, so listing them there forces the
    # sidecar import eagerly for every star-importer -- defeating the lazy loader.
    code = (
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _blocked(name, *a, **k):\n"
        "    if name in ('fastapi', 'uvicorn'):\n"
        "        raise ModuleNotFoundError(f'simulated: no module named {name!r}')\n"
        "    return _real_import(name, *a, **k)\n"
        "builtins.__import__ = _blocked\n"
        "from pytest_agent_chaos import *\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_sidecar_exports_are_still_reachable_lazily() -> None:
    code = (
        "from pytest_agent_chaos import ChaosSidecarServer, RemoteChaosClient, run_chaos_sidecar\n"
        "assert ChaosSidecarServer and RemoteChaosClient and run_chaos_sidecar\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
