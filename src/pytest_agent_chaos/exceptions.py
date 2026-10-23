"""Custom exceptions for pytest-agent-chaos.

Provides distinct assertions and error hierarchies for agent loop runaway,
unhandled chaos faults, and transport-level connection failures.
"""


class AgentChaosException(Exception):
    """Base exception for all pytest-agent-chaos errors."""
    pass


class AgentLoopExhaustedError(AgentChaosException):
    """Raised when an agent state machine or cyclic graph exceeds its max_graph_steps budget."""
    pass


class InfiniteAgentLoopError(AgentLoopExhaustedError):
    """Alias for AgentLoopExhaustedError for state graph cycle audits."""
    pass


class UnhandledChaosFaultError(AgentChaosException):
    """Raised when an agent workflow ungracefully crashes upon encountering an injected fault."""
    pass


class ChaosStreamDropError(AgentChaosException):
    """Simulates an abrupt mid-stream disconnect from an LLM SSE server."""
    pass


class ChaosTimeoutError(AgentChaosException):
    """Simulates a network or gateway timeout during LLM communication."""
    pass
