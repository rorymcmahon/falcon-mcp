"""Structured tool output without inflating tools/list.

Background
----------
MCP's ``outputSchema`` serves two distinct purposes that FastMCP couples to a
single ``output_schema`` value on the tool:

1. It is advertised in ``tools/list`` so clients can learn a tool's return
   shape ahead of calling it.
2. When set, it makes FastMCP populate ``structuredContent`` on the
   ``CallToolResult`` at call time (``FuncMetadata.convert_result``), in
   addition to the human-readable ``content`` text block.

Registering tools with ``structured_output=False`` (falcon-mcp's default)
disables *both*: it keeps ``tools/list`` lean — the fix for issue #325, where
per-tool ``outputSchema`` bloat pushed the catalogue past VS Code Copilot's
context budget and silently dropped tools — but it also leaves
``structuredContent`` unset. Clients that read a tool result from
``structuredContent`` (rather than parsing the JSON ``content`` text block)
then see an empty result.

This module decouples the two concerns. Registering a tool with
``structured_output=True`` builds the output model/schema so
``convert_result`` emits ``structuredContent`` on responses; overriding
``list_tools`` to drop ``outputSchema`` keeps the listing lean. The result:
structured responses without regressing #325.

Why a factory rather than a fixed subclass
-------------------------------------------
The server builds its MCP server from the ``FastMCP`` symbol imported into
``falcon_mcp.server`` so that ``FastMCP`` stays the single patch point the test
suite mocks. ``structured_content_server_class`` derives the mixin subclass
from whatever ``FastMCP`` resolves to *at call time*, so a test that patches
``falcon_mcp.server.FastMCP`` with a mock still controls construction, while a
real deployment gets a genuine FastMCP subclass with the ``list_tools``
override.
"""

from mcp.types import Tool as MCPTool


class StructuredContentMixin:
    """Strip ``outputSchema`` from ``tools/list`` while allowing structured responses.

    Mixed into a FastMCP class, this emits ``structuredContent`` for tools
    registered with ``structured_output=True`` (unchanged FastMCP behaviour on
    the call path) while ``tools/list`` omits ``outputSchema`` for every tool,
    keeping the catalogue within context-budget-constrained clients' limits
    (issue #325).

    Tools registered with ``structured_output=False`` carry no output schema, so
    there is nothing to strip and no structured content to emit — making the
    mixin safe to apply unconditionally.
    """

    async def list_tools(self) -> list[MCPTool]:  # type: ignore[override]
        """Return the tool list with ``outputSchema`` stripped from every entry.

        Delegates to the stock implementation, then clears ``outputSchema`` so a
        tool built for structured *responses* does not also advertise its schema
        in the *listing*. ``inputSchema`` and every other field are left intact.
        """
        tools = await super().list_tools()  # type: ignore[misc]
        for tool in tools:
            tool.outputSchema = None
        return tools


def structured_content_server_class(base: type) -> type:
    """Return a subclass of ``base`` that applies :class:`StructuredContentMixin`.

    ``base`` is the ``FastMCP`` class (or a test double) the server currently
    references. Deriving at call time keeps ``falcon_mcp.server.FastMCP`` the
    single construction seam the test suite patches.
    """
    return type("StructuredContentFastMCP", (StructuredContentMixin, base), {})
