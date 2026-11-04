from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from a3opt.contracts import CandidateSpec

DEFAULT_EXPLORATION = 0.7
DEFAULT_MAX_CHILDREN = 3


@dataclass
class TreeNode:
    node_id: str
    candidate: CandidateSpec | None = None
    parent_id: str | None = None
    visits: int = 0
    total_value: float = 0.0
    children: list[str] = field(default_factory=list)
    created_index: int = 0
    stage: str = ""
    status: str = ""
    error: str = ""

    @property
    def mean_value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "candidate": self.candidate.to_dict() if self.candidate is not None else None,
            "parent_id": self.parent_id,
            "visits": int(self.visits),
            "total_value": float(self.total_value),
            "children": list(self.children),
            "created_index": int(self.created_index),
            "stage": self.stage,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeNode":
        candidate = data.get("candidate")
        return cls(
            node_id=str(data["node_id"]),
            candidate=CandidateSpec.from_dict(candidate) if isinstance(candidate, dict) else None,
            parent_id=data.get("parent_id"),
            visits=int(data.get("visits", 0)),
            total_value=float(data.get("total_value", 0.0)),
            children=[str(child) for child in data.get("children", [])],
            created_index=int(data.get("created_index", 0)),
            stage=str(data.get("stage", "")),
            status=str(data.get("status", "")),
            error=str(data.get("error", "")),
        )


class ExperimentTree:
    def __init__(
        self,
        root_candidate: CandidateSpec | None = None,
        c: float = DEFAULT_EXPLORATION,
        max_children: int = DEFAULT_MAX_CHILDREN,
        root_id: str = "root",
    ) -> None:
        self.c = float(c)
        self.max_children = int(max_children)
        self.root_id = str(root_id)
        self.nodes: dict[str, TreeNode] = {}
        root = TreeNode(node_id=self.root_id, candidate=root_candidate, created_index=0)
        self.nodes[root.node_id] = root

    @property
    def exploration(self) -> float:
        return self.c

    @property
    def root(self) -> TreeNode:
        return self.nodes[self.root_id]

    def __len__(self) -> int:
        return len(self.nodes)

    def all_nodes(self) -> list[TreeNode]:
        return list(self.nodes.values())

    def get(self, node_id: str) -> TreeNode:
        return self.nodes[str(node_id)]

    def resolve(self, node: TreeNode | str) -> TreeNode:
        if isinstance(node, TreeNode):
            resolved = self.nodes.get(node.node_id)
            if resolved is None:
                raise KeyError(f"unknown node: {node.node_id}")
            return resolved
        key = str(node)
        if key not in self.nodes:
            raise KeyError(f"unknown node: {key}")
        return self.nodes[key]

    def expandable(self, node: TreeNode | str) -> bool:
        return len(self.resolve(node).children) < self.max_children

    def _next_created_index(self) -> int:
        if not self.nodes:
            return 0
        return max(node.created_index for node in self.nodes.values()) + 1

    def _next_node_id(self) -> str:
        index = len(self.nodes)
        while f"n{index:03d}" in self.nodes:
            index += 1
        return f"n{index:03d}"

    def add_child(
        self,
        parent: TreeNode | CandidateSpec | str | None = None,
        candidate: CandidateSpec | None = None,
        node_id: str | None = None,
        stage: str = "",
        status: str = "",
        error: str = "",
    ) -> TreeNode:
        if isinstance(parent, CandidateSpec):
            if candidate is None:
                candidate = parent
            parent = None
        parent_node = self.root if parent is None else self.resolve(parent)
        if candidate is None:
            raise ValueError("candidate is required to add a tree node")
        if node_id is None:
            node_id = self._next_node_id()
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node_id}")
        child = TreeNode(
            node_id=str(node_id),
            candidate=candidate,
            parent_id=parent_node.node_id,
            created_index=self._next_created_index(),
            stage=stage,
            status=status,
            error=error,
        )
        self.nodes[child.node_id] = child
        parent_node.children.append(child.node_id)
        return child

    def uct(self, node: TreeNode | str) -> float:
        current = self.resolve(node)
        if current.visits <= 0:
            return float("inf")
        if current.parent_id is None:
            return current.mean_value
        parent = self.nodes.get(current.parent_id)
        if parent is None or parent.visits <= 0:
            return float("inf")
        return current.mean_value + self.c * math.sqrt(math.log(parent.visits) / current.visits)

    def select(self) -> TreeNode:
        unvisited = [
            node
            for node in self.nodes.values()
            if node.visits == 0 and self.expandable(node)
        ]
        if unvisited:
            return min(unvisited, key=lambda node: node.created_index)
        expandable = [
            node
            for node in self.nodes.values()
            if node.visits > 0 and self.expandable(node)
        ]
        if not expandable:
            expandable = [node for node in self.nodes.values() if node.visits > 0]
        if not expandable:
            return self.root
        return max(expandable, key=lambda node: (self.uct(node), -node.created_index))

    def backpropagate(self, node: TreeNode | str, value: float) -> None:
        current: TreeNode | None = self.resolve(node)
        while current is not None:
            current.visits += 1
            current.total_value += float(value)
            if current.parent_id is None:
                break
            current = self.nodes.get(current.parent_id)

    def best(self) -> TreeNode:
        visited = [node for node in self.nodes.values() if node.visits > 0]
        with_candidate = [node for node in visited if node.candidate is not None]
        pool = with_candidate or visited
        if not pool:
            return self.root
        return max(pool, key=lambda node: (node.mean_value, node.visits, -node.created_index))

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "c": self.c,
            "max_children": self.max_children,
            "nodes": [node.to_dict() for node in self.nodes.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentTree":
        raw_nodes = data.get("nodes", [])
        nodes = [TreeNode.from_dict(item) for item in raw_nodes]
        if not nodes:
            raise ValueError("tree data must contain at least one node")
        root_node = nodes[0]
        tree = cls(
            root_candidate=root_node.candidate,
            c=float(data.get("c", DEFAULT_EXPLORATION)),
            max_children=int(data.get("max_children", DEFAULT_MAX_CHILDREN)),
            root_id=root_node.node_id,
        )
        tree.nodes = {node.node_id: node for node in nodes}
        return tree
