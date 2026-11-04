from __future__ import annotations

import json
import math

import pytest

from a3opt.contracts import CandidateSpec
from a3opt.tree import (
    DEFAULT_EXPLORATION,
    DEFAULT_MAX_CHILDREN,
    ExperimentTree,
    TreeNode,
)


def make_candidate(candidate_id: str, seed: int = 0) -> CandidateSpec:
    return CandidateSpec(candidate_id=candidate_id, seed=seed)


def test_default_constants_and_root():
    assert DEFAULT_EXPLORATION == 0.7
    assert DEFAULT_MAX_CHILDREN == 3
    tree = ExperimentTree(make_candidate("root"))
    assert tree.c == 0.7
    assert tree.exploration == 0.7
    assert tree.max_children == 3
    assert tree.root.candidate == make_candidate("root")
    assert tree.root.parent_id is None
    assert tree.root.visits == 0
    assert tree.root.total_value == 0.0
    assert len(tree) == 1
    assert isinstance(tree.root, TreeNode)


def test_uct_formula_matches_definition():
    tree = ExperimentTree(make_candidate("root"), c=0.7)
    root = tree.root
    tree.backpropagate(root, 0.0)
    first = tree.add_child(root, make_candidate("a"))
    tree.backpropagate(first, 0.8)
    second = tree.add_child(root, make_candidate("b"))
    tree.backpropagate(second, 0.2)
    assert root.visits == 3
    expected_first = 0.8 + 0.7 * math.sqrt(math.log(3) / 1)
    expected_second = 0.2 + 0.7 * math.sqrt(math.log(3) / 1)
    assert tree.uct(first) == pytest.approx(expected_first)
    assert tree.uct(second) == pytest.approx(expected_second)
    assert tree.uct(first.node_id) == pytest.approx(expected_first)


def test_uct_uses_parent_visits_and_node_mean():
    tree = ExperimentTree(make_candidate("root"), c=0.5)
    root = tree.root
    tree.backpropagate(root, 0.0)
    child = tree.add_child(root, make_candidate("a"))
    tree.backpropagate(child, 1.0)
    grandchild = tree.add_child(child, make_candidate("a1"))
    tree.backpropagate(grandchild, 1.0)
    tree.backpropagate(grandchild, 0.0)
    assert child.visits == 3
    assert grandchild.visits == 2
    expected = grandchild.mean_value + 0.5 * math.sqrt(
        math.log(child.visits) / grandchild.visits
    )
    assert grandchild.mean_value == pytest.approx(0.5)
    assert tree.uct(grandchild) == pytest.approx(expected)


def test_uct_unvisited_child_is_infinite():
    tree = ExperimentTree(make_candidate("root"))
    tree.backpropagate(tree.root, 0.0)
    child = tree.add_child(tree.root, make_candidate("a"))
    assert tree.uct(child) == float("inf")


def test_select_returns_unvisited_nodes_in_creation_order():
    tree = ExperimentTree(make_candidate("root"))
    assert tree.select().node_id == tree.root.node_id
    tree.backpropagate(tree.root, 0.0)
    first = tree.add_child(tree.root, make_candidate("a"))
    second = tree.add_child(tree.root, make_candidate("b"))
    assert tree.select().node_id == first.node_id
    tree.backpropagate(first, 0.5)
    assert tree.select().node_id == second.node_id
    tree.backpropagate(second, 0.5)
    assert tree.select().node_id == first.node_id


def test_select_uses_uct_when_all_nodes_are_visited():
    tree = ExperimentTree(make_candidate("root"), c=0.7)
    root = tree.root
    tree.backpropagate(root, 0.0)
    first = tree.add_child(root, make_candidate("a"))
    second = tree.add_child(root, make_candidate("b"))
    tree.backpropagate(first, 0.1)
    tree.backpropagate(second, 0.9)
    assert tree.select().node_id == second.node_id
    assert tree.uct(second) > tree.uct(first)


def test_select_skips_nodes_that_reached_child_limit():
    tree = ExperimentTree(make_candidate("root"), max_children=1)
    root = tree.root
    tree.backpropagate(root, 0.0)
    only = tree.add_child(root, make_candidate("a"))
    tree.backpropagate(only, 0.1)
    assert tree.select().node_id == only.node_id
    grandchild = tree.add_child(only, make_candidate("a1"))
    tree.backpropagate(grandchild, 0.5)
    assert tree.select().node_id == grandchild.node_id


def test_add_child_links_parent_and_children():
    tree = ExperimentTree(make_candidate("root"))
    child = tree.add_child(tree.root, make_candidate("a"))
    assert child.parent_id == tree.root.node_id
    assert tree.root.children == [child.node_id]
    assert tree.get(child.node_id) is child
    explicit = tree.add_child(
        tree.root.node_id,
        make_candidate("b"),
        node_id="custom",
        stage="encoder_fusion",
    )
    assert explicit.node_id == "custom"
    assert explicit.stage == "encoder_fusion"
    assert tree.root.children == [child.node_id, "custom"]
    assert tree.resolve(child) is child
    child_from_candidate = tree.add_child(make_candidate("c"))
    assert child_from_candidate.parent_id == tree.root.node_id
    with pytest.raises(ValueError):
        tree.add_child(tree.root, make_candidate("dup"), node_id=child.node_id)
    with pytest.raises(KeyError):
        tree.resolve("missing")


def test_backpropagate_accumulates_across_ancestors():
    tree = ExperimentTree(make_candidate("root"))
    root = tree.root
    child = tree.add_child(root, make_candidate("a"))
    grandchild = tree.add_child(child, make_candidate("a1"))
    tree.backpropagate(grandchild, 1.0)
    tree.backpropagate(grandchild, 0.5)
    tree.backpropagate(child, 0.25)
    assert grandchild.visits == 2
    assert grandchild.total_value == pytest.approx(1.5)
    assert child.visits == 3
    assert child.total_value == pytest.approx(1.75)
    assert root.visits == 3
    assert root.total_value == pytest.approx(1.75)
    assert child.mean_value == pytest.approx(1.75 / 3)


def test_best_returns_highest_mean_value():
    tree = ExperimentTree(make_candidate("root"))
    root = tree.root
    tree.backpropagate(root, 0.0)
    first = tree.add_child(root, make_candidate("a"))
    second = tree.add_child(root, make_candidate("b"))
    tree.backpropagate(first, 0.6)
    tree.backpropagate(second, 0.6)
    assert tree.best().node_id == first.node_id
    tree.backpropagate(second, 0.6)
    assert tree.best().node_id == second.node_id


def test_best_skips_nodes_without_candidate():
    tree = ExperimentTree()
    assert tree.root.candidate is None
    tree.backpropagate(tree.root, 0.99)
    child = tree.add_child(tree.root, make_candidate("a"))
    tree.backpropagate(child, 0.1)
    assert tree.best().node_id == child.node_id


def test_to_dict_from_dict_roundtrip():
    tree = ExperimentTree(make_candidate("root"), c=1.2, max_children=2)
    root = tree.root
    tree.backpropagate(root, 0.3)
    child = tree.add_child(
        root, make_candidate("a", seed=1), stage="encoder_fusion", status="ok"
    )
    tree.backpropagate(child, 0.7)
    grandchild = tree.add_child(child, make_candidate("a1", seed=1))
    tree.backpropagate(grandchild, 0.9)
    data = json.loads(json.dumps(tree.to_dict()))
    restored = ExperimentTree.from_dict(data)
    assert restored.c == pytest.approx(1.2)
    assert restored.max_children == 2
    assert len(restored) == len(tree)
    assert restored.root.node_id == root.node_id
    assert restored.root.visits == 3
    assert restored.root.candidate == root.candidate
    assert restored.get(child.node_id).candidate == child.candidate
    assert restored.get(child.node_id).visits == 2
    assert restored.get(child.node_id).total_value == pytest.approx(1.6)
    assert restored.get(child.node_id).stage == "encoder_fusion"
    assert restored.get(grandchild.node_id).visits == 1
    assert restored.get(grandchild.node_id).total_value == pytest.approx(0.9)
    assert restored.select().node_id == tree.select().node_id
    assert restored.best().node_id == tree.best().node_id


def test_from_dict_rejects_empty_nodes():
    with pytest.raises(ValueError):
        ExperimentTree.from_dict({"nodes": []})
