"""Dependency-free client for the local SceneFactory runtime HTTP API."""

from __future__ import annotations

import json
import hashlib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

from .session_contracts import finite_json


class RuntimeClientError(RuntimeError):
    def __init__(self, status: int, error: dict[str, Any]) -> None:
        self.status = status
        self.error = error
        super().__init__(
            f"HTTP {status}: {error.get('code', 'runtime_error')}: {error.get('message', '')}"
        )


class RuntimeClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        *,
        timeout: float = 120,
        token: str | None = None,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) URL without query or fragment")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        if not path.startswith("/v1/") or ".." in path:
            raise ValueError("runtime path must start with /v1/")
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = None
        if payload is not None:
            data = json.dumps(finite_json(payload), separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return finite_json(json.load(response))
        except urllib.error.HTTPError as exc:
            try:
                payload = json.load(exc)
                error = payload.get("error", {})
            except (ValueError, AttributeError):
                error = {"code": "invalid_error_response", "message": str(exc)}
            raise RuntimeClientError(exc.code, error) from exc

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        request = dict(request)
        request.setdefault("idempotency_key", uuid.uuid4().hex)
        return self.request("POST", "/v1/sessions", request)

    def list(self) -> dict[str, Any]:
        return self.request("GET", "/v1/sessions")

    @staticmethod
    def _path(session_id: str, suffix: str = "") -> str:
        if (
            not isinstance(session_id, str)
            or not session_id
            or not all(c.isalnum() or c in "-_" for c in session_id)
        ):
            raise ValueError("invalid session_id")
        return "/v1/sessions/" + session_id + suffix

    def inspect(self, session_id: str) -> dict[str, Any]:
        return self.request("GET", self._path(session_id))

    def reset(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", self._path(session_id, "/reset"), request)

    def step(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", self._path(session_id, "/step"), request)

    def observe(
        self, session_id: str, *, modalities: list[str] | None = None, camera: str = "default"
    ) -> dict[str, Any]:
        query = {}
        if modalities is not None:
            query["modalities"] = ",".join(modalities)
        if camera != "default":
            query["camera"] = camera
        suffix = "/observation" + ("?" + urllib.parse.urlencode(query) if query else "")
        return self.request("GET", self._path(session_id, suffix))

    def world_state(self, session_id: str) -> dict[str, Any]:
        """Read the committed world state without advancing the episode."""
        return self.request("GET", self._path(session_id, "/world-state"))

    def build(self, request: dict[str, Any]) -> dict[str, Any]:
        request = dict(request)
        request.setdefault("idempotency_key", uuid.uuid4().hex)
        return self.request("POST", "/v1/builds", request)

    def build_status(self, build_id: str) -> dict[str, Any]:
        if not isinstance(build_id, str) or not build_id.isalnum():
            raise ValueError("invalid build_id")
        return self.request("GET", "/v1/builds/" + build_id)

    def snapshot(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", self._path(session_id, "/snapshots"), request)

    def restore(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", self._path(session_id, "/restore"), request)

    def close(self, session_id: str) -> dict[str, Any]:
        return self.request("DELETE", self._path(session_id))

    def artifact(self, session_id: str, name: str, *, sha256: str | None = None) -> bytes:
        """Download a sensor artifact and optionally verify its descriptor digest."""
        if not isinstance(name, str) or not name or any(char in name for char in "/\\:") or name in {".", ".."}:
            raise ValueError("invalid artifact name")
        path = self._path(session_id, "/artifacts/" + urllib.parse.quote(name, safe=""))
        headers = {"Accept": "application/octet-stream"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(self.base_url + path, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read()
        if sha256 is not None and hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError("artifact digest does not match observation descriptor")
        return payload
