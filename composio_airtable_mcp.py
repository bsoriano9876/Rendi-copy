#!/usr/bin/env python3
"""Small stdio MCP bridge for Composio toolkit tools.

This avoids depending on Composio's hosted MCP endpoint while still using
Composio's normal project API, connected account, and tool schemas.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


BASE_URL = os.getenv("COMPOSIO_BASE_URL", "https://backend.composio.dev").rstrip("/")
API_KEY = os.getenv("COMPOSIO_API_KEY", "").strip()
TOOLKIT_SLUG = os.getenv("COMPOSIO_TOOLKIT_SLUG", "airtable").strip()
USER_ID = os.getenv("COMPOSIO_USER_ID", "").strip()
CONNECTED_ACCOUNT_ID = os.getenv("COMPOSIO_CONNECTED_ACCOUNT_ID", "").strip()
CACHE_TTL_SECONDS = int(os.getenv("COMPOSIO_TOOL_CACHE_TTL", "300"))

_TOOLS_CACHE: dict[str, Any] = {"loaded_at": 0.0, "tools": [], "by_name": {}}


def _require_env() -> None:
    missing = [
        name
        for name, value in {
            "COMPOSIO_API_KEY": API_KEY,
            "COMPOSIO_USER_ID": USER_ID,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")


def _request_json(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    _require_env()
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {
        "accept": "application/json",
        "x-api-key": API_KEY,
    }
    if data is not None:
        headers["content-type"] = "application/json"
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Composio HTTP {exc.code}: {text[:1200]}") from exc
    return json.loads(text) if text else {}


def _fetch_tools() -> list[dict[str, Any]]:
    now = time.monotonic()
    if _TOOLS_CACHE["tools"] and now - _TOOLS_CACHE["loaded_at"] < CACHE_TTL_SECONDS:
        return list(_TOOLS_CACHE["tools"])

    items: list[dict[str, Any]] = []
    cursor = ""
    while True:
        params = {
            "toolkit_slug": TOOLKIT_SLUG,
            "limit": "100",
        }
        if cursor:
            params["cursor"] = cursor
        query = urllib.parse.urlencode(params)
        page = _request_json("GET", f"/api/v3.1/tools?{query}")
        items.extend(page.get("items") or [])
        cursor = str(page.get("next_cursor") or "")
        if not cursor:
            break

    by_name = {item["slug"]: item for item in items if item.get("slug")}
    _TOOLS_CACHE.update({"loaded_at": now, "tools": items, "by_name": by_name})
    return list(items)


def _schema_for(item: dict[str, Any]) -> dict[str, Any]:
    schema = item.get("input_parameters") or {"type": "object", "properties": {}}
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    if schema.get("type") != "object":
        schema = {"type": "object", "properties": {"value": schema}}
    schema.setdefault("properties", {})
    return schema


def _description_for(item: dict[str, Any]) -> str:
    pieces = [
        str(item.get("name") or item.get("slug") or "").strip(),
        str(item.get("description") or "").strip(),
    ]
    if item.get("is_deprecated") or (item.get("deprecated") or {}).get("is_deprecated"):
        pieces.append("Deprecated by Composio; use a non-deprecated Airtable tool when possible.")
    pieces.append(f"Runs through the Team Hermes Composio {TOOLKIT_SLUG} connected account.")
    return "\n\n".join(piece for piece in pieces if piece)


app = Server(f"team-hermes-composio-{TOOLKIT_SLUG}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name=item["slug"],
            description=_description_for(item),
            inputSchema=_schema_for(item),
        )
        for item in _fetch_tools()
        if item.get("slug")
    ]


@app.call_tool(validate_input=False)
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    _fetch_tools()
    by_name = _TOOLS_CACHE["by_name"]
    if name not in by_name:
        known = ", ".join(sorted(by_name)[:20])
        raise RuntimeError(f"Unknown Composio tool {name!r}. Known tools include: {known}")

    body: dict[str, Any] = {
        "user_id": USER_ID,
        "arguments": arguments or {},
    }
    if CONNECTED_ACCOUNT_ID:
        body["connected_account_id"] = CONNECTED_ACCOUNT_ID
    version = by_name[name].get("version")
    if version:
        body["version"] = version

    result = _request_json("POST", f"/api/v3.1/tools/execute/{urllib.parse.quote(name)}", body)
    return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
