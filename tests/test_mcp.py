import asyncio

import pytest

from livekit.agents.llm.mcp import MCPServer, MCPTool

try:
    import mcp.types
except ImportError:
    pytest.skip("mcp not installed", allow_module_level=True)


class FakeMCPServer(MCPServer):
    """MCPServer subclass that doesn't connect to anything real."""

    def __init__(self) -> None:
        super().__init__(client_session_timeout_seconds=5)
        self._on_tools_changed_called = asyncio.Event()

    def client_streams(self):
        raise NotImplementedError("FakeMCPServer doesn't connect")

    async def simulate_tool_list_changed(self) -> None:
        """Simulate receiving a ToolListChangedNotification."""
        notification = mcp.types.ServerNotification(
            root=mcp.types.ToolListChangedNotification()
        )
        await self._on_server_message(notification)


@pytest.mark.asyncio
async def test_mcp_server_invalidates_cache_on_tool_list_changed():
    server = FakeMCPServer()
    # Manually mark cache as clean to verify invalidation
    server._cache_dirty = False

    await server.simulate_tool_list_changed()

    assert server._cache_dirty is True


@pytest.mark.asyncio
async def test_mcp_server_calls_on_tools_changed_callback():
    callback_called = asyncio.Event()

    async def on_changed() -> None:
        callback_called.set()

    server = FakeMCPServer()
    server.on_tools_changed = on_changed

    await server.simulate_tool_list_changed()

    assert callback_called.is_set()
