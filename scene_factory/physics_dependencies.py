"""Conservative physical dependency graphs and invalidation decisions.

The graph is deliberately explicit about uncertainty. It can identify artifacts
that are safe to reuse when a declared input changes, but it never turns a
partial graph into proof that contact or task evidence remains valid.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable


GRAPH_SCHEMA = "scene_factory.physics_dependency_graph.v1"


@dataclass
class PhysicsDependencyGraph:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, node_id: str, kind: str, **metadata: Any) -> None:
        if node_id in self.nodes and self.nodes[node_id].get("kind") != kind:
            raise ValueError(f"dependency node kind changed: {node_id}")
        self.nodes[node_id] = {"kind": kind, **metadata}
        self.edges.setdefault(node_id, set())

    def add_edge(self, source: str, target: str, relation: str = "depends_on") -> None:
        if source not in self.nodes or target not in self.nodes:
            raise KeyError(f"dependency edge references unknown node: {source} -> {target}")
        self.edges.setdefault(source, set()).add(target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRAPH_SCHEMA,
            "nodes": {key: copy.deepcopy(value) for key, value in sorted(self.nodes.items())},
            "edges": [
                {"source": source, "target": target}
                for source in sorted(self.edges)
                for target in sorted(self.edges[source])
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhysicsDependencyGraph":
        if payload.get("schema_version") != GRAPH_SCHEMA:
            raise ValueError(f"unsupported dependency graph schema: {payload.get('schema_version')!r}")
        graph = cls()
        for node_id, metadata in payload.get("nodes", {}).items():
            if not isinstance(node_id, str) or not isinstance(metadata, dict):
                raise ValueError("dependency nodes must be an object")
            graph.add_node(node_id, str(metadata["kind"]), **{key: value for key, value in metadata.items() if key != "kind"})
        for edge in payload.get("edges", []):
            graph.add_edge(edge["source"], edge["target"])
        return graph

    def affected_closure(self, changed_nodes: Iterable[str]) -> set[str]:
        """Return changed nodes and every artifact that transitively uses them."""
        changed = set(changed_nodes)
        unknown = changed - self.nodes.keys()
        if unknown:
            raise KeyError(f"unknown changed dependency nodes: {sorted(unknown)}")
        reverse: dict[str, set[str]] = {node: set() for node in self.nodes}
        for source, targets in self.edges.items():
            for target in targets:
                reverse[target].add(source)
        queue = list(changed)
        while queue:
            current = queue.pop()
            for dependent in reverse[current]:
                if dependent not in changed:
                    changed.add(dependent)
                    queue.append(dependent)
        return changed

    def invalidation_report(self, changed_nodes: Iterable[str]) -> dict[str, Any]:
        changed = set(changed_nodes)
        closure = self.affected_closure(changed)
        dynamics = {node for node in closure if self.nodes[node]["kind"] in {"physics", "contact", "support"}}
        evidence = {node for node in closure if self.nodes[node]["kind"] in {"evidence", "task_validation"}}
        unknown = any(self.nodes[node].get("conservative_unknown") for node in changed)
        full_revalidation = bool(dynamics or evidence or unknown)
        reasons: list[str] = []
        if dynamics:
            reasons.append("a physics/contact/support dependency changed")
        if evidence:
            reasons.append("dependent task or physics evidence is stale")
        if unknown:
            reasons.append("changed input has no locality proof")
        reusable = {
            node for node in self.nodes
            if node not in closure and self.nodes[node]["kind"] in {"visual", "layout", "artifact"}
        }
        return {
            "schema_version": "scene_factory.physics_invalidation.v1",
            "changed_nodes": sorted(changed),
            "affected_nodes": sorted(closure),
            "reusable_nodes": sorted(reusable),
            "full_dynamics_revalidation": full_revalidation,
            "reasons": reasons,
        }


def build_dependency_graph(scene: dict[str, Any], *, backend: str = "mujoco") -> PhysicsDependencyGraph:
    if not isinstance(scene, dict) or not isinstance(scene.get("objects"), list):
        raise ValueError("scene must contain an objects array")
    graph = PhysicsDependencyGraph()
    graph.add_node("scene", "layout")
    graph.add_node("task", "task_validation")
    graph.add_node("artifact:mjcf", "artifact", backend=backend)
    graph.add_node("artifact:visual", "visual")
    graph.add_node("artifact:physics_evidence", "evidence")
    graph.add_edge("artifact:mjcf", "scene")
    graph.add_edge("artifact:visual", "scene")
    graph.add_edge("artifact:physics_evidence", "artifact:mjcf")
    graph.add_edge("artifact:physics_evidence", "task")
    for item in scene["objects"]:
        if not isinstance(item, dict) or not isinstance(item.get("object_id"), str):
            raise ValueError("scene objects require object_id")
        object_id = item["object_id"]
        instance = f"instance:{object_id}"
        asset = f"asset:{item.get('asset_id', object_id)}"
        physics = f"physics:{object_id}"
        contact = f"contact:{object_id}"
        for node, kind in ((instance, "layout"), (asset, "geometry"), (physics, "physics"), (contact, "contact")):
            graph.add_node(node, kind, object_id=object_id, conservative_unknown=kind in {"geometry", "contact"})
        graph.add_edge(instance, "scene")
        graph.add_edge(asset, instance)
        graph.add_edge(physics, instance)
        graph.add_edge(physics, asset)
        graph.add_edge(contact, physics)
        graph.add_edge(contact, asset)
        graph.add_edge("artifact:mjcf", physics)
        graph.add_edge("artifact:mjcf", contact)
        graph.add_edge("artifact:visual", asset)
        graph.add_edge("artifact:physics_evidence", contact)
        support = item.get("support")
        if isinstance(support, str) and support:
            support_node = f"support:{support.partition(':')[0]}"
            graph.add_node(support_node, "support", object_id=support.partition(':')[0])
            graph.add_edge(physics, support_node)
            graph.add_edge(contact, support_node)
    for item in scene["objects"]:
        support_node = f"support:{item['object_id']}"
        if support_node in graph.nodes:
            graph.add_edge(support_node, f"instance:{item['object_id']}")
    return graph


def plan_scene_update(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Classify a scene diff without assuming the old contact graph is complete."""
    graph = build_dependency_graph(before)
    old = {item["object_id"]: item for item in before["objects"]}
    new = {item["object_id"]: item for item in after["objects"]}
    changes = set()
    if old.keys() != new.keys():
        changes.add("scene")
    if before.get("task") != after.get("task"):
        changes.add("task")
    for object_id in old.keys() & new.keys():
        left, right = old[object_id], new[object_id]
        if left.get("physics") != right.get("physics"):
            changes.add(f"physics:{object_id}")
        if any(left.get(key) != right.get(key) for key in ("pose", "support", "dynamic", "asset_id", "bbox_m")):
            changes.add(f"instance:{object_id}")
        other_keys = (left.keys() | right.keys()) - {"physics", "pose", "support", "dynamic", "asset_id", "bbox_m"}
        if any(left.get(key) != right.get(key) for key in other_keys):
            changes.add(f"instance:{object_id}")
    if any(before.get(key) != after.get(key) for key in before.keys() | after.keys() if key not in {"objects", "task", "randomization"}):
        changes.add("scene")
    report = graph.invalidation_report(changes)
    report["local_contact_proof"] = "not_available"
    if changes:
        report["full_dynamics_revalidation"] = True
        report["reasons"].append("new contacts cannot be excluded from the old graph; full dynamics/task replay required")
    report["graph"] = graph.to_dict()
    return report

