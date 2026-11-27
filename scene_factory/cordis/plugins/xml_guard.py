"""Reversible technical safety plugin for MuJoCo XML tool calls.

Only guards against technical hazards (path traversal in XML file attributes,
non-finite NaN/Inf tokens). Never enforces semantic/category/layout whitelists.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from ..kernel import PluginScope


_NON_FINITE_RE = re.compile(r"\b(?:nan|inf|-inf|\+inf)\b", re.IGNORECASE)
_TRAVERSAL_RE = re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)")


def plugin(scope: PluginScope, config: dict[str, Any]) -> None:
    mode = str(config.get("mode", "advise")).lower()  # "advise" | "block"

    def _before_tool(
        tool_name: str, arguments: dict[str, Any], _state: dict[str, Any]
    ) -> dict[str, Any] | None:
        if tool_name not in {"write_xml", "edit_xml"}:
            return None
        text = str(arguments.get("xml", arguments.get("new_text", "")))
        if _NON_FINITE_RE.search(text):
            msg = "XML contains non-finite numeric token (NaN/Inf); replace with finite numbers."
            return {"block": msg} if mode == "block" else {"hint": msg}

        if tool_name == "write_xml":
            try:
                root = ET.fromstring(text)
            except ET.ParseError as exc:
                msg = f"XML syntax error before write: {exc}"
                return {"block": msg} if mode == "block" else {"hint": msg}
            for elem in root.iter():
                file_attr = elem.attrib.get("file", "")
                if file_attr and (
                    _TRAVERSAL_RE.search(file_attr)
                    or file_attr.startswith(("/", "\\"))
                    or (len(file_attr) >= 2 and file_attr[1] == ":")
                ):
                    return {
                        "block": (
                            f"External/absolute XML file path {file_attr!r} is disallowed; "
                            "use inline mesh vertices or insert_asset."
                        )
                    }
        return None

    scope.on_before_tool(_before_tool)
