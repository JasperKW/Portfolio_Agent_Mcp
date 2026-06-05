"""
src/mcp/client.py
------------------
MCP Client: connects to financial-data-server, wraps each tool as a BaseTool.

Two key patterns from deepbrief MCPToolAdapter (S7 nb06):

1. MCPToolAdapter: subclasses BaseTool so the agent loop sees MCP tools
   and local Python tools identically — "the agent loop has no idea
   whether a tool is local Python or remote MCP."

2. discover_mcp_tools(): connects at startup, calls tools/list,
   returns one adapter per tool. Dynamic discovery — Agent doesn't
   hardcode what tools exist.

3. Namespacing: tools are prefixed "financial__get_price_history"
   to prevent collision if we add more MCP servers later.
   (S7: "tool-name shadowing is a real attack vector")
"""

from __future__ import annotations
import json
import time
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.tools.financial_tools import BaseTool, ToolResult


# ── MCPToolAdapter (mirrors deepbrief pattern exactly) ───────────────────────
class MCPToolAdapter(BaseTool):
    """
    Wraps a single remote MCP tool as a BaseTool.

    The agent loop calls adapter.execute(**kwargs) and has no idea
    whether the tool runs locally or over HTTP — same interface either way.

    Design choice: per-call session (not persistent).
    Slightly higher overhead (~20-50ms) but simpler error isolation.
    For production: hold one persistent session per agent lifetime.
    """

    def __init__(
        self,
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: dict,
        url: str,
    ) -> None:
        # Namespace prefix prevents tool-name collisions across servers
        self.name = f"{server_name}__{tool_name}"
        self.description = description
        self._tool_name = tool_name   # bare name used in MCP call
        self._url = url
        # Patch additionalProperties: False so strict mode works
        self.parameters_schema = _strictify(input_schema)

    def to_openai_function(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        t0 = time.time()
        try:
            result = await self._call_mcp(kwargs)
            latency = int((time.time() - t0) * 1000)
            return ToolResult(
                tool_name=self.name,
                input_args=kwargs,
                success=True,
                output=result,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                input_args=kwargs,
                success=False,
                error=f"MCP transport error ({self._url}): {e}",
            )

    async def _call_mcp(self, arguments: dict) -> Any:
        """
        Open session → call tool → parse response → close session.

        Wire-level flow (Streamable HTTP, S7 §4.2):
          POST /mcp  { initialize }          → 200 OK, Mcp-Session-Id: <uuid>
          POST /mcp  { tools/call, args }    → 200 OK, { result }
          DELETE /mcp                        → session closed
        """
        async with streamablehttp_client(url=self._url) as (read, write, _):
            async with ClientSession(read, write) as session:
                # Phase 1: handshake + capability negotiation
                await session.initialize()

                # Phase 2: tool invocation
                mcp_result = await session.call_tool(self._tool_name, arguments)

                # Parse content blocks (FastMCP returns TextContent list)
                if not mcp_result.content:
                    return {}

                text = mcp_result.content[0].text
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw": text}


# ── Dynamic tool discovery ────────────────────────────────────────────────────
async def discover_mcp_tools(
    url: str,
    server_name: str,
) -> list[MCPToolAdapter]:
    """
    Connect to MCP server, call tools/list, return one adapter per tool.

    This is the key MCP advantage over static function calling:
    the agent doesn't hardcode what tools exist — it discovers them at runtime.
    Add a new tool to the server → all agents pick it up on next startup,
    no code change needed.

    Args:
        url: MCP server URL, e.g. "http://localhost:8001/mcp"
        server_name: Namespace prefix, e.g. "financial"
                     → tool names become "financial__get_price_history"
    """
    async with streamablehttp_client(url=url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # tools/list: dynamic discovery (vs static hardcoding)
            tools_response = await session.list_tools()
            tools = tools_response.tools

    print(f"  Discovered {len(tools)} tools from {server_name} ({url}):")
    adapters = []
    for t in tools:
        adapter = MCPToolAdapter(
            server_name=server_name,
            tool_name=t.name,
            description=t.description or "",
            input_schema=getattr(t, "inputSchema", {"type": "object", "properties": {}}),
            url=url,
        )
        adapters.append(adapter)
        print(f"    • {adapter.name}")

    return adapters


# ── Helper ────────────────────────────────────────────────────────────────────
def _strictify(schema: dict) -> dict:
    """
    Patch additionalProperties: False so OpenAI strict mode accepts the schema.
    MCP servers don't always declare this, but strict=True requires it.
    """
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return schema
    out = dict(schema)
    out["additionalProperties"] = False
    if "properties" in out and "required" not in out:
        out["required"] = list(out["properties"].keys())
    return out
