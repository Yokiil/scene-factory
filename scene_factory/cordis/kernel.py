"""Minimal Cordis-inspired reversible plugin microkernel.

Every constraint, prompt injection, or tool hook is mounted inside a disposable
PluginScope. Disposing or unmounting a plugin cleanly reverses all of its
effects, allowing teams to retire model-specific guardrails via config as base
LLMs improve — without welding interceptions into the agent loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


PromptHook = Callable[[dict[str, Any]], str | list[str] | None]
BeforeToolHook = Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any] | None]
AfterToolHook = Callable[[str, dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any] | None]
AfterVerifyHook = Callable[[str, dict[str, Any]], dict[str, Any] | None]
PluginFactory = Callable[["PluginScope", dict[str, Any]], None]


@dataclass(frozen=True)
class PluginMeta:
    """Governance metadata describing why a plugin exists and when it should exit."""

    name: str
    kind: str  # "prompt_injection" | "tool_hook" | "verifier_hook"
    purpose: str
    added_for_model: str = "general"
    exit_criteria: str = "Retire once base model passes benchmark without this plugin."
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PluginScope:
    """Isolated, reversible scope owned by one mounted plugin."""

    context: "CordisContext"
    meta: PluginMeta
    config: dict[str, Any] = field(default_factory=dict)
    _disposers: list[Callable[[], None]] = field(default_factory=list, init=False, repr=False)
    disposed: bool = field(default=False, init=False)

    def effect(self, dispose: Callable[[], None]) -> None:
        """Register a reversible teardown callback executed when this plugin exits."""
        if self.disposed:
            raise RuntimeError(f"plugin scope {self.meta.name!r} is already disposed")
        self._disposers.append(dispose)

    def on_prompt(self, fn: PromptHook) -> None:
        """Mount a prompt-injection hook (preferred way to guide LLM traces)."""
        self.context._register(self, self.context._prompt_hooks, fn)

    def on_before_tool(self, fn: BeforeToolHook) -> None:
        """Mount a pre-tool hook (may return {'arguments': ..., 'hint': ..., 'block': ...})."""
        self.context._register(self, self.context._before_tool_hooks, fn)

    def on_after_tool(self, fn: AfterToolHook) -> None:
        """Mount a post-tool hook (may annotate or add guidance hints to tool output)."""
        self.context._register(self, self.context._after_tool_hooks, fn)

    def on_after_verify(self, fn: AfterVerifyHook) -> None:
        """Mount a post-verification hook (advisory only; never falsifies physics truth)."""
        self.context._register(self, self.context._after_verify_hooks, fn)

    def dispose(self) -> None:
        """Reverse all effects registered in this scope in LIFO order."""
        if self.disposed:
            return
        self.disposed = True
        errors: list[Exception] = []
        while self._disposers:
            undo = self._disposers.pop()
            try:
                undo()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                errors.append(exc)
        if self.context._scopes.get(self.meta.name) is self:
            del self.context._scopes[self.meta.name]
        if errors:
            raise errors[0]


class CordisContext:
    """Shared reactive hook registry managing plugin admission and retirement."""

    def __init__(self) -> None:
        self._scopes: dict[str, PluginScope] = {}
        self._prompt_hooks: list[tuple[str, PromptHook]] = []
        self._before_tool_hooks: list[tuple[str, BeforeToolHook]] = []
        self._after_tool_hooks: list[tuple[str, AfterToolHook]] = []
        self._after_verify_hooks: list[tuple[str, AfterVerifyHook]] = []

    def _register(self, scope: PluginScope, bucket: list[tuple[str, Any]], fn: Any) -> None:
        entry = (scope.meta.name, fn)
        bucket.append(entry)

        def _undo() -> None:
            try:
                bucket.remove(entry)
            except ValueError:
                pass

        scope.effect(_undo)

    def mount(
        self,
        plugin: PluginFactory,
        meta: PluginMeta,
        config: dict[str, Any] | None = None,
    ) -> PluginScope:
        """Mount or hot-reload a plugin inside a fresh reversible scope."""
        if meta.name in self._scopes:
            self._scopes[meta.name].dispose()
        scope = PluginScope(context=self, meta=meta, config=dict(config or {}))
        self._scopes[meta.name] = scope
        try:
            plugin(scope, scope.config)
        except Exception:
            scope.dispose()
            raise
        return scope

    def unmount(self, name: str) -> bool:
        """Cleanly retire a mounted plugin, reversing all of its hooks."""
        scope = self._scopes.get(name)
        if scope is None:
            return False
        scope.dispose()
        return True

    def dispose_all(self) -> None:
        """Unmount every active plugin cleanly."""
        for name in list(self._scopes):
            self.unmount(name)

    def mounted_names(self) -> list[str]:
        return list(self._scopes.keys())

    def snapshot(self) -> list[dict[str, Any]]:
        """Return serializable governance metadata for trace provenance & ablations."""
        return [
            {**scope.meta.to_dict(), "config": dict(scope.config)}
            for scope in self._scopes.values()
        ]

    def collect_prompt_sections(self, state: dict[str, Any]) -> list[str]:
        sections: list[str] = []
        for _, hook in list(self._prompt_hooks):
            out = hook(state)
            if isinstance(out, str) and out.strip():
                sections.append(out.strip())
            elif isinstance(out, list):
                sections.extend(item.strip() for item in out if isinstance(item, str) and item.strip())
        return sections

    def run_before_tool(
        self, tool_name: str, arguments: dict[str, Any], state: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str], str | None]:
        """Run pre-tool hooks. Returns (arguments, advisory_hints, optional_block_reason)."""
        current = dict(arguments)
        hints: list[str] = []
        for plugin_name, hook in list(self._before_tool_hooks):
            outcome = hook(tool_name, current, state)
            if not outcome:
                continue
            if "arguments" in outcome and isinstance(outcome["arguments"], dict):
                current = dict(outcome["arguments"])
            if outcome.get("hint"):
                hints.append(f"[{plugin_name}] {outcome['hint']}")
            if outcome.get("block"):
                return current, hints, f"[{plugin_name}] {outcome['block']}"
        return current, hints, None

    def run_after_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        current = dict(result)
        for _, hook in list(self._after_tool_hooks):
            updated = hook(tool_name, arguments, current, state)
            if isinstance(updated, dict):
                current = updated
        return current

    def run_after_verify(self, xml: str, report: dict[str, Any]) -> dict[str, Any]:
        current = dict(report)
        for _, hook in list(self._after_verify_hooks):
            updated = hook(xml, current)
            if isinstance(updated, dict):
                current = updated
        return current
