"""Zero-dependency OpenAI-compatible multimodal (image + text) tool-calling client."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""


class ChatModel(Protocol):
    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: float = 90.0
    temperature: float = 0.2
    ca_bundle: str = ""

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def load_llm_config(config_path: str | Path | None = None) -> LLMConfig:
    """Load config/llm.json (optional) with SCENE_FACTORY_LLM_* environment overrides."""
    raw: dict[str, Any] = {}
    path = Path(config_path) if config_path else Path(
        os.environ.get("SCENE_FACTORY_LLM_CONFIG", "config/llm.json")
    )
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"LLM config must be a JSON object: {path}")
        if "api_key" in loaded:
            raise ValueError("do not store api_key in config/llm.json; use SCENE_FACTORY_LLM_API_KEY")
        raw = loaded

    def _get(env: str, key: str, default: Any) -> Any:
        return os.environ[env] if env in os.environ else raw.get(key, default)

    base_url = str(_get("SCENE_FACTORY_LLM_BASE_URL", "base_url", "")).strip()
    model = str(_get("SCENE_FACTORY_LLM_MODEL", "model", "")).strip()
    api_key_env = str(raw.get("api_key_env", "SCENE_FACTORY_LLM_API_KEY")).strip()
    api_key = os.environ.get("SCENE_FACTORY_LLM_API_KEY") or os.environ.get(api_key_env, "")
    timeout = float(_get("SCENE_FACTORY_LLM_TIMEOUT_SECONDS", "timeout_seconds", 90.0))
    temperature = float(_get("SCENE_FACTORY_LLM_TEMPERATURE", "temperature", 0.2))
    ca_bundle = str(_get("SCENE_FACTORY_LLM_CA_BUNDLE", "ca_bundle", "")).strip()
    return LLMConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout,
        temperature=temperature,
        ca_bundle=ca_bundle,
    )


def image_to_data_url(path: str | Path) -> tuple[str, str]:
    """Encode a local image file as a base64 data URI. Returns `(data_url, mime_type)`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"image file not found: {file_path}")
    mime, _ = mimetypes.guess_type(file_path.name)
    mime_type = mime or "image/png"
    b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{b64}", mime_type


def bytes_to_data_url(data: bytes, mime_type: str = "image/png") -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


class LLMClient:
    """Minimal stdlib OpenAI-compatible client supporting images + tool calls."""

    def __init__(self, config: LLMConfig) -> None:
        if not config.base_url or not config.model:
            raise ValueError(
                "LLM base_url and model are required (set SCENE_FACTORY_LLM_BASE_URL / "
                "SCENE_FACTORY_LLM_MODEL or config/llm.json)"
            )
        self.config = config
        self.model = config.model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        req = Request(
            self.config.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        ctx = (
            ssl.create_default_context(cafile=self.config.ca_bundle)
            if self.config.ca_bundle and self.config.ca_bundle.lower() not in {"system", "certifi"}
            else None
        )
        try:
            with urlopen(req, timeout=self.config.timeout_seconds, context=ctx) as resp:
                decoded = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        choice = decoded["choices"][0]
        msg = choice["message"]
        raw_content = msg.get("content") or ""
        if isinstance(raw_content, list):
            content = "".join(
                str(part.get("text", "")) for part in raw_content if isinstance(part, dict)
            )
        else:
            content = str(raw_content)
        reasoning = str(msg.get("reasoning_content") or msg.get("reasoning") or "")

        calls: list[ToolCall] = []
        for idx, tc in enumerate(msg.get("tool_calls") or ()):
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, dict):
                parsed_args = raw_args
                raw_args_str = json.dumps(raw_args, ensure_ascii=False)
            else:
                raw_args_str = str(raw_args)
                try:
                    parsed_args = json.loads(raw_args_str) if raw_args_str.strip() else {}
                except json.JSONDecodeError:
                    parsed_args = {"_raw": raw_args_str}
            calls.append(
                ToolCall(
                    id=str(tc.get("id") or f"call_{idx}"),
                    name=str(fn.get("name") or ""),
                    arguments=parsed_args if isinstance(parsed_args, dict) else {},
                    raw_arguments=raw_args_str,
                )
            )

        return LLMResponse(
            content=content,
            tool_calls=tuple(calls),
            reasoning=reasoning,
            usage=dict(decoded.get("usage") or {}),
            model=str(decoded.get("model") or self.config.model),
            finish_reason=str(choice.get("finish_reason") or ""),
        )
