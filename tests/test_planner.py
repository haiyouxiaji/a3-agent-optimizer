from __future__ import annotations

from a3opt.contracts import CandidateSpec
from a3opt.planner import StagedPlanner
from a3opt.space import CandidateSpace
from a3opt.tree import ExperimentTree


def test_propose_returns_first_unused_candidate_in_staged_order():
    space = CandidateSpace()
    planner = StagedPlanner(space)
    baseline = space.candidates()[0]
    first = planner.propose(None, set(), space)
    assert first == baseline
    used = {baseline.candidate_id}
    second = planner.propose(None, used, space)
    assert second == space.candidates()[1]
    used.add(second.candidate_id)
    assert planner.propose(None, used, space) == space.candidates()[2]


def test_propose_yields_all_candidates_in_exact_staged_order():
    space = CandidateSpace()
    planner = StagedPlanner(space)
    expected = [
        ("tiny_cnn", "concat", "chunk", 20, 1e-3, 64),
        ("tiny_cnn", "film", "chunk", 20, 1e-3, 64),
        ("state_mlp", "concat", "chunk", 20, 1e-3, 0),
        ("tiny_cnn", "concat", "mlp", 1, 1e-3, 64),
        ("tiny_cnn", "concat", "chunk", 50, 1e-3, 64),
        ("tiny_cnn", "concat", "chunk", 20, 3e-3, 64),
    ]
    used: set[str] = set()
    produced: list[tuple[str, str, str, int, float, int]] = []
    while True:
        spec = planner.propose(None, used, space)
        if spec is None:
            break
        assert spec.candidate_id not in used
        assert space.validate(spec) == []
        produced.append(
            (
                spec.encoder,
                spec.fusion,
                spec.action_head,
                spec.chunk_size,
                spec.lr,
                spec.image_size,
            )
        )
        used.add(spec.candidate_id)
    assert produced == expected
    assert planner.propose(None, used, space) is None


def test_propose_returns_none_when_exhausted():
    space = CandidateSpace()
    planner = StagedPlanner(space)
    used = {spec.candidate_id for spec in space.candidates()}
    assert planner.propose(None, used, space) is None
    assert planner.propose_stage(None, used, space) is None


def test_propose_uses_stage_labels():
    space = CandidateSpace()
    planner = StagedPlanner(space)
    stage, spec = planner.propose_stage(None, set(), space)
    assert stage == "baseline"
    assert spec == space.baseline()
    stage, spec = planner.propose_stage(None, {spec.candidate_id}, space)
    assert stage == "encoder_fusion"


def test_propose_uses_planner_space_by_default():
    space = CandidateSpace()
    planner = StagedPlanner(space)
    assert planner.propose(None).candidate_id == space.candidates()[0].candidate_id
    assert (
        planner.propose(None, set()).candidate_id
        == space.candidates()[0].candidate_id
    )
    assert (
        StagedPlanner().propose(None, set(), space).candidate_id
        == space.candidates()[0].candidate_id
    )


def test_propose_does_not_mutate_used_candidate_ids():
    space = CandidateSpace()
    planner = StagedPlanner(space)
    used = {space.candidates()[0].candidate_id}
    before = set(used)
    planner.propose(None, used, space)
    assert used == before


def test_propose_derives_seed_from_tree_parent():
    space = CandidateSpace()
    planner = StagedPlanner(space)
    tree = ExperimentTree(CandidateSpec(candidate_id="root", seed=7))
    proposed = planner.propose(tree.root, set(), space)
    assert proposed is not None
    assert proposed.seed == 7
    assert proposed == space.candidates(seed=7)[0]
    proposed_from_spec = planner.propose(CandidateSpec(candidate_id="root", seed=2), set(), space)
    assert proposed_from_spec is not None
    assert proposed_from_spec.seed == 2


def test_tree_planner_integration_walks_staged_order():
    space = CandidateSpace()
    planner = StagedPlanner(space)
    tree = ExperimentTree(space.baseline())
    used = {tree.root.candidate.candidate_id}
    spec = planner.propose(tree.root, used, space)
    assert spec is not None
    assert spec.candidate_id == space.candidates()[1].candidate_id
    child = tree.add_child(tree.root, spec, stage=space.stage_of(spec))
    assert child.stage == "encoder_fusion"
    used.add(spec.candidate_id)
    next_spec = planner.propose(child, used, space)
    assert next_spec is not None
    assert next_spec.candidate_id == space.candidates()[2].candidate_id
