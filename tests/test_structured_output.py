"""Tests for the opt-in structured-output switch.

falcon-mcp registers tools with ``structured_output=False`` by default, which
keeps ``tools/list`` free of ``outputSchema`` (issue #325) but also leaves
``structuredContent`` unset on tool results — so clients that read structured
output see blank results.

The ``structured_output`` switch (``--structured-output`` /
``FALCON_MCP_STRUCTURED_OUTPUT``) decouples the two: tool results carry
``structuredContent`` while ``tools/list`` still omits ``outputSchema``. These
tests assert both halves, and that the default remains unchanged.
"""

import unittest
from unittest.mock import MagicMock, patch

from mcp.shared.memory import create_connected_server_and_client_session

from falcon_mcp.server import FalconMCPServer


def _mock_falcon_client_cls(patcher_target: str):
    """Patch FalconClient so the server initialises without real credentials."""
    patcher = patch(patcher_target)
    mock_cls = patcher.start()
    mock_client = MagicMock()
    mock_client.authenticate.return_value = True
    mock_client.is_authenticated.return_value = True
    mock_cls.return_value = mock_client
    return patcher, mock_client


class TestStructuredOutputSwitch(unittest.IsolatedAsyncioTestCase):
    """Structured-output behaviour under both settings."""

    def setUp(self):
        self.patcher, self.mock_client = _mock_falcon_client_cls(
            "falcon_mcp.server.FalconClient"
        )
        self.addCleanup(self.patcher.stop)

    async def _list_tools(self, server: FalconMCPServer):
        async with create_connected_server_and_client_session(server.server) as session:
            return (await session.list_tools()).tools

    async def _call_tool(self, server: FalconMCPServer, name: str, arguments: dict):
        async with create_connected_server_and_client_session(server.server) as session:
            return await session.call_tool(name, arguments)

    async def test_disabled_by_default_no_structured_content(self):
        """Default: no outputSchema in tools/list and no structuredContent on results."""
        server = FalconMCPServer(enabled_modules={"hosts"})

        tools = await self._list_tools(server)
        self.assertGreater(len(tools), 0)
        self.assertTrue(
            all(t.outputSchema is None for t in tools),
            "Default server must not advertise outputSchema (issue #325).",
        )

        # falcon_list_enabled_modules is a zero-argument, side-effect-free meta-tool.
        result = await self._call_tool(server, "falcon_list_enabled_modules", {})
        self.assertIsNone(
            result.structuredContent,
            "Default server must not populate structuredContent.",
        )
        # The content text block is always present regardless of the switch.
        self.assertTrue(result.content, "Result should always carry a content block.")

    async def test_enabled_populates_structured_content(self):
        """Enabled: results carry structuredContent, listing still omits outputSchema."""
        server = FalconMCPServer(enabled_modules={"hosts"}, structured_output=True)

        # Issue #325 must stay fixed: no outputSchema in the listing.
        tools = await self._list_tools(server)
        self.assertGreater(len(tools), 0)
        offenders = [t.name for t in tools if t.outputSchema is not None]
        self.assertFalse(
            offenders,
            "structured_output must not reintroduce outputSchema in tools/list "
            "(issue #325). Offenders:\n" + "\n".join(f"  - {n}" for n in offenders),
        )

        # Tool results must now carry structuredContent.
        result = await self._call_tool(server, "falcon_list_enabled_modules", {})
        self.assertIsNotNone(
            result.structuredContent,
            "structured_output=True must populate structuredContent on results.",
        )

    async def test_enabled_structured_content_matches_payload(self):
        """The structuredContent must reflect the tool's actual return value."""
        server = FalconMCPServer(
            enabled_modules={"hosts"}, structured_output=True
        )

        result = await self._call_tool(server, "falcon_list_enabled_modules", {})

        # list_enabled_modules returns {"modules": [...]}. A dict return is emitted
        # as-is under structuredContent (no {"result": ...} wrapping for object
        # returns).
        self.assertIn("modules", result.structuredContent)
        self.assertIn("hosts", result.structuredContent["modules"])


class TestStructuredOutputCLI(unittest.TestCase):
    """The switch is wired through argument parsing and the env var."""

    def test_flag_absent_defaults_false(self):
        from falcon_mcp.server import parse_args

        with patch("sys.argv", ["falcon-mcp"]), patch.dict(
            "os.environ", {}, clear=False
        ) as env:
            env.pop("FALCON_MCP_STRUCTURED_OUTPUT", None)
            args = parse_args()
        self.assertFalse(args.structured_output)

    def test_flag_enables(self):
        from falcon_mcp.server import parse_args

        with patch("sys.argv", ["falcon-mcp", "--structured-output"]):
            args = parse_args()
        self.assertTrue(args.structured_output)

    def test_env_enables(self):
        from falcon_mcp.server import parse_args

        with patch("sys.argv", ["falcon-mcp"]), patch.dict(
            "os.environ", {"FALCON_MCP_STRUCTURED_OUTPUT": "true"}
        ):
            args = parse_args()
        self.assertTrue(args.structured_output)


if __name__ == "__main__":
    unittest.main()
