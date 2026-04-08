import asyncio

import pytest

from livekit.agents.llm.mcp import MCPServer, MCPTool, MCPToolset
from livekit.agents.llm.tool_context import RawFunctionTool, function_tool, get_fnc_tool_names

try:
    import mcp.types
except ImportError:
    pytest.skip("mcp not installed", allow_module_level=True)


class FakeMCPServer(MCPServer):
    """MCPServer subclass that doesn't connect to anything real."""

    def __init__(self) -> None:
        super().__init__(client_session_timeout_seconds=5)

    def client_streams(self):
        raise NotImplementedError("FakeMCPServer doesn't connect")

    async def simulate_tool_list_changed(self) -> None:
        """Simulate receiving a ToolListChangedNotification."""
        notification = mcp.types.ServerNotification(root=mcp.types.ToolListChangedNotification())
        await self._on_server_message(notification)


def _make_raw_tool(name: str) -> RawFunctionTool:
    @function_tool(
        raw_schema={
            "name": name,
            "description": f"Tool {name}",
            "parameters": {"type": "object", "properties": {}},
        }
    )
    async def _tool(raw_arguments: dict) -> str:
        return name

    return _tool


class FakeMCPServerWithTools(MCPServer):
    """MCPServer subclass that returns configurable tools."""

    def __init__(self, tools: list[MCPTool] | None = None) -> None:
        super().__init__(client_session_timeout_seconds=5)
        self._fake_tools: list[MCPTool] = tools or []
        self._is_initialized = False

    @property
    def initialized(self) -> bool:
        return self._is_initialized

    def client_streams(self):
        raise NotImplementedError

    async def initialize(self) -> None:
        self._is_initialized = True

    async def list_tools(self) -> list[MCPTool]:
        return self._fake_tools

    def set_tools(self, tools: list[MCPTool]) -> None:
        self._fake_tools = tools


# --- MCPServer tests ---


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


# --- MCPToolset tests ---


@pytest.mark.asyncio
async def test_mcp_toolset_refreshes_on_tools_changed():
    tool_a = _make_raw_tool("tool_a")
    server = FakeMCPServerWithTools(tools=[tool_a])

    toolset = MCPToolset(id="test", mcp_server=server)
    await toolset.setup()

    assert len(toolset.tools) == 1

    # Simulate server adding a new tool
    tool_b = _make_raw_tool("tool_b")
    server.set_tools([tool_a, tool_b])

    # Trigger the on_tools_changed callback that MCPToolset registered
    assert server.on_tools_changed is not None
    await server.on_tools_changed()

    assert len(toolset.tools) == 2


@pytest.mark.asyncio
async def test_mcp_toolset_on_tools_changed_callback_called():
    tool_a = _make_raw_tool("tool_a")
    server = FakeMCPServerWithTools(tools=[tool_a])

    callback_called = asyncio.Event()

    async def on_changed() -> None:
        callback_called.set()

    toolset = MCPToolset(id="test", mcp_server=server)
    toolset.on_tools_changed = on_changed
    await toolset.setup()

    # Simulate server changing tools
    tool_b = _make_raw_tool("tool_b")
    server.set_tools([tool_b])
    await server.on_tools_changed()

    assert callback_called.is_set()


@pytest.mark.asyncio
async def test_mcp_toolset_on_tools_changed_propagates_new_tool_names():
    """Verify MCPToolset exposes the changed tools so AgentActivity.update_tools can see them."""
    tool_a = _make_raw_tool("tool_a")
    server = FakeMCPServerWithTools(tools=[tool_a])

    received_names: list[list[str]] = []

    async def on_changed() -> None:
        received_names.append(get_fnc_tool_names(list(toolset.tools)))

    toolset = MCPToolset(id="test", mcp_server=server)
    toolset.on_tools_changed = on_changed
    await toolset.setup()

    # Simulate server changing tools
    tool_b = _make_raw_tool("tool_b")
    tool_c = _make_raw_tool("tool_c")
    server.set_tools([tool_b, tool_c])
    await server.on_tools_changed()

    assert len(received_names) == 1
    assert sorted(received_names[0]) == ["tool_b", "tool_c"]
