"""Tests for the opt-in json-response switch.

falcon-mcp's streamable-http transport returns Server-Sent Events
(``text/event-stream``) by default. Some proxies relay MCP over a plain
request/response channel and cannot forward SSE — notably Amazon Bedrock
AgentCore Runtime, which rejects/collapses the streamed response so every call
fails. The ``json_response`` switch (``--json-response`` /
``FALCON_MCP_JSON_RESPONSE``) makes FastMCP return a single
``application/json`` body instead.

These tests assert the flag reaches FastMCP's setting and that the default is
unchanged (SSE).
"""

import unittest
from unittest.mock import MagicMock, patch

from falcon_mcp.server import FalconMCPServer, parse_args


def _mock_client():
    patcher = patch("falcon_mcp.server.FalconClient")
    mock_cls = patcher.start()
    inst = MagicMock()
    inst.authenticate.return_value = True
    inst.is_authenticated.return_value = True
    mock_cls.return_value = inst
    return patcher


class TestJsonResponseSwitch(unittest.TestCase):
    def setUp(self):
        self.patcher = _mock_client()
        self.addCleanup(self.patcher.stop)

    def test_default_is_sse(self):
        """Default: FastMCP is built with json_response=False (SSE)."""
        server = FalconMCPServer(enabled_modules={"hosts"})
        self.assertFalse(server.json_response)
        # FastMCP records the setting on its settings object.
        self.assertFalse(server.server.settings.json_response)

    def test_enabled_sets_fastmcp_json_response(self):
        """json_response=True flows through to FastMCP's setting."""
        server = FalconMCPServer(enabled_modules={"hosts"}, json_response=True)
        self.assertTrue(server.json_response)
        self.assertTrue(server.server.settings.json_response)

    def test_enabled_coexists_with_structured_output(self):
        """json_response and structured_output are independent and compose."""
        server = FalconMCPServer(
            enabled_modules={"hosts"},
            json_response=True,
            structured_output=True,
        )
        self.assertTrue(server.server.settings.json_response)
        # structured_output builds the outputSchema-stripping subclass; both on.
        self.assertTrue(server.structured_output)


class TestJsonResponseCLI(unittest.TestCase):
    def test_flag_absent_defaults_false(self):
        with patch("sys.argv", ["falcon-mcp"]), patch.dict(
            "os.environ", {}, clear=False
        ) as env:
            env.pop("FALCON_MCP_JSON_RESPONSE", None)
            args = parse_args()
        self.assertFalse(args.json_response)

    def test_flag_enables(self):
        with patch("sys.argv", ["falcon-mcp", "--json-response"]):
            args = parse_args()
        self.assertTrue(args.json_response)

    def test_env_enables(self):
        with patch("sys.argv", ["falcon-mcp"]), patch.dict(
            "os.environ", {"FALCON_MCP_JSON_RESPONSE": "true"}
        ):
            args = parse_args()
        self.assertTrue(args.json_response)


if __name__ == "__main__":
    unittest.main()
