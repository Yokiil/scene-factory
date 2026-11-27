"""Post-training-ready multimodal Agent Trace recorder and SFT/RL JSONL exporter.

Every SceneAgent execution emits an OpenAI/ChatML-compatible multi-turn trace
recording input images/text, active Cordis plugins, reasoning/tool calls,
objective MuJoCo verification feedback, and final verifiable score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


TRACE_SCHEMA_VERSION = "scene_factory.agent_trace.v2"


@dataclass
class AgentTrace:
    """Complete self-contained trajectory ready for post-training and ablations."""

    prompt: str
    model: str
    input_images: list[dict[str, Any]] = field(default_factory=list)
    cordis_plugins: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turns: int = 0
    final_xml_sha256: str | None = None
    final_xml_path: str | None = None
    verification: dict[str, Any] | None = None
    verifier_score: float = 0.0
    variant_count: int = 0
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    fragment_cache: dict[str, int] = field(default_factory=dict)

    def add_usage(self, usage: dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens"):
            val = usage.get(key)
            if isinstance(val, int) and val > 0:
                self.usage[key] = self.usage.get(key, 0) + val

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "model": self.model,
            "input": {"prompt": self.prompt, "images": list(self.input_images)},
            "cordis_plugins": list(self.cordis_plugins),
            "messages": list(self.messages),
            "turns": self.turns,
            "final_xml_sha256": self.final_xml_sha256,
            "final_xml_path": self.final_xml_path,
            "verification": self.verification,
            "verifier_score": self.verifier_score,
            "variant_count": self.variant_count,
            "usage": dict(self.usage),
            "fragment_cache": dict(self.fragment_cache),
        }

    def save(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(dest)
        return dest

    def to_sft_record(self) -> dict[str, Any]:
        """Project into a standard multi-turn SFT / RL trajectory record."""
        return {
            "trace_id": self.trace_id,
            "model": self.model,
            "reward": self.verifier_score,
            "cordis_plugin_names": [p["name"] for p in self.cordis_plugins],
            "messages": self.messages,
            "final_xml_sha256": self.final_xml_sha256,
        }


def describe_image_file(path: str | Path, mime_type: str = "image/png") -> dict[str, Any]:
    file_path = Path(path).resolve()
    return {
        "path": str(file_path),
        "mime_type": mime_type,
        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
    }


def export_sft_jsonl(
    trace_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    min_score: float = 0.8,
) -> dict[str, Any]:
    """Filter verified traces by objective MuJoCo verifier score and export JSONL."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    exported = 0
    lines: list[str] = []
    for raw_path in trace_paths:
        total += 1
        data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        if data.get("schema_version") != TRACE_SCHEMA_VERSION:
            continue
        score = float(data.get("verifier_score", 0.0))
        if score < min_score:
            continue
        record = {
            "trace_id": data["trace_id"],
            "model": data["model"],
            "reward": score,
            "cordis_plugin_names": [p["name"] for p in data.get("cordis_plugins", [])],
            "messages": data["messages"],
            "final_xml_sha256": data.get("final_xml_sha256"),
        }
        lines.append(json.dumps(record, ensure_ascii=False))
        exported += 1
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {"total": total, "exported": exported, "min_score": min_score, "output": str(dest)}
