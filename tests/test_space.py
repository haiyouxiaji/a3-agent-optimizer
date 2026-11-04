from __future__ import annotations

import pytest

from a3opt.contracts import (
    ACTION_HEADS,
    CHUNK_SIZES,
    ENCODERS,
    FUSIONS,
    IMAGE_SIZES,
    LEARNING_RATES,
    CandidateSpec,
)
from a3opt.space import (
    SPACE_NAME,
    STAGE_BASELINE,
    STAGE_ENCODER_FUSION,
    STAGE_HEAD_CHUNK,
    STAGE_LR,
    STAGE_ORDER,
    CandidateSpace,
)


def test_space_dimensions_match_contracts():
    space = CandidateSpace()
    assert space.encoders == tuple(ENCODERS)
    assert space.fusions == tuple(FUSIONS)
    assert space.action_heads == tuple(ACTION_HEADS)
    assert space.chunk_sizes == tuple(CHUNK_SIZES)
    assert space.image_sizes == tuple(IMAGE_SIZES)
    assert space.learning_rates == tuple(LEARNING_RATES)
    assert space.family == "bc_policy"
    assert space.state_dim == 16
    assert space.name == SPACE_NAME
    assert space.grid_size() == 96


def test_baseline_matches_contract_defaults():
    space = CandidateSpace()
    baseline = space.baseline()
    assert baseline.encoder == "tiny_cnn"
    assert baseline.fusion == "concat"
    assert baseline.action_head == "chunk"
    assert baseline.chunk_size == 20
    assert baseline.lr == 1e-3
    assert baseline.image_size == 64
    assert baseline.state_dim == 16
    assert baseline.seed == 0
    assert baseline.family == "bc_policy"
    assert baseline.candidate_id
    assert space.validate(baseline) == []


def test_validate_reports_unknown_dimension_values():
    space = CandidateSpace()
    assert any(
        "encoder" in error
        for error in space.validate(CandidateSpec(candidate_id="x", encoder="resnet"))
    )
    assert any(
        "fusion" in error
        for error in space.validate(CandidateSpec(candidate_id="x", fusion="sum"))
    )
    assert any(
        "action_head" in error
        for error in space.validate(CandidateSpec(candidate_id="x", action_head="diffusion"))
    )
    assert any(
        "chunk_size" in error
        for error in space.validate(CandidateSpec(candidate_id="x", chunk_size=7))
    )
    assert any(
        "image_size" in error
        for error in space.validate(CandidateSpec(candidate_id="x", image_size=128))
    )
    assert any(
        "lr" in error
        for error in space.validate(CandidateSpec(candidate_id="x", lr=5e-3))
    )
    assert any(
        "state_dim" in error
        for error in space.validate(CandidateSpec(candidate_id="x", state_dim=0))
    )
    assert any(
        "seed" in error
        for error in space.validate(CandidateSpec(candidate_id="x", seed=-1))
    )
    assert any(
        "candidate_id" in error
        for error in space.validate(CandidateSpec(candidate_id=""))
    )
    assert any(
        "family" in error
        for error in space.validate(CandidateSpec(candidate_id="x", family=""))
    )


def test_validate_enforces_encoder_image_compatibility():
    space = CandidateSpace()
    assert any(
        "image_size" in error
        for error in space.validate(CandidateSpec(candidate_id="x", image_size=0))
    )
    assert any(
        "image_size" in error
        for error in space.validate(
            CandidateSpec(candidate_id="x", encoder="state_mlp", image_size=64)
        )
    )
    assert (
        space.validate(
            CandidateSpec(candidate_id="x", encoder="state_mlp", image_size=0)
        )
        == []
    )


def test_validate_enforces_module_compatibility():
    space = CandidateSpace()
    assert any(
        "chunk_size" in error
        for error in space.validate(
            CandidateSpec(candidate_id="x", action_head="chunk", chunk_size=1)
        )
    )
    assert any(
        "chunk_size" in error
        for error in space.validate(
            CandidateSpec(candidate_id="x", action_head="mlp", chunk_size=20)
        )
    )
    assert (
        space.validate(
            CandidateSpec(candidate_id="x", action_head="mlp", chunk_size=1)
        )
        == []
    )
    assert any(
        "image_size" in error
        for error in space.validate(
            CandidateSpec(
                candidate_id="x",
                encoder="state_mlp",
                fusion="film",
                image_size=0,
            )
        )
    )
    assert (
        space.validate(
            CandidateSpec(candidate_id="x", fusion="film", image_size=64)
        )
        == []
    )


def test_slug_is_deterministic_and_field_sensitive():
    space = CandidateSpace()
    baseline = space.baseline(seed=3)
    assert space.slug(baseline) == baseline.candidate_id
    assert space.slug(space.baseline(seed=3)) == baseline.candidate_id
    variants = [
        space.make(encoder="state_mlp", image_size=0, seed=3),
        space.make(fusion="film", seed=3),
        space.make(action_head="mlp", chunk_size=1, seed=3),
        space.make(chunk_size=1, seed=3),
        space.make(lr=3e-3, seed=3),
        space.make(seed=4),
    ]
    identifiers = [baseline.candidate_id] + [variant.candidate_id for variant in variants]
    assert len(identifiers) == len(set(identifiers))
    assert baseline.candidate_id.startswith("bc_policy-tiny_cnn-concat-chunk")
    assert baseline.candidate_id.endswith("-s3")


def test_custom_family_and_state_dim_flow_into_candidates():
    space = CandidateSpace(family="bc_policy_v2", state_dim=32)
    spec = space.baseline(seed=5)
    assert spec.family == "bc_policy_v2"
    assert spec.state_dim == 32
    assert spec.candidate_id.startswith("bc_policy_v2-")
    assert space.validate(spec) == []
    assert all(candidate.state_dim == 32 for candidate in space.candidates(seed=5))


def test_candidates_follow_staged_order():
    space = CandidateSpace()
    staged = space.staged_candidates()
    specs = space.candidates()
    expected_stages = (
        [STAGE_BASELINE]
        + [STAGE_ENCODER_FUSION] * 2
        + [STAGE_HEAD_CHUNK] * 2
        + [STAGE_LR]
    )
    assert [stage for stage, _ in staged] == expected_stages
    assert [spec for _, spec in staged] == specs
    assert tuple(dict.fromkeys(stage for stage, _ in staged)) == STAGE_ORDER
    assert len(specs) == 6
    assert len({spec.candidate_id for spec in specs}) == 6
    assert specs[0] == space.baseline()
    assert specs[1].encoder == "tiny_cnn" and specs[1].fusion == "film"
    assert specs[1].chunk_size == 20 and specs[1].image_size == 64
    assert (
        specs[2].encoder == "state_mlp"
        and specs[2].fusion == "concat"
        and specs[2].image_size == 0
        and specs[2].chunk_size == 20
    )
    assert specs[3].action_head == "mlp" and specs[3].chunk_size == 1
    assert specs[4].action_head == "chunk" and specs[4].chunk_size == 50
    assert specs[5].lr == 3e-3 and specs[5].chunk_size == 20
    assert all(spec.lr == 1e-3 for spec in specs[:5])
    assert all(spec.seed == 0 for spec in specs)
    assert all(space.validate(spec) == [] for spec in specs)
    assert all(spec.image_size > 0 if spec.fusion == "film" else True for spec in specs)
    assert all(spec.chunk_size > 1 for spec in specs if spec.action_head == "chunk")
    assert all(spec.chunk_size == 1 for spec in specs if spec.action_head == "mlp")


def test_candidates_are_deterministic_and_carry_seed():
    space = CandidateSpace()
    first = [spec.candidate_id for spec in space.candidates(seed=0)]
    second = [spec.candidate_id for spec in CandidateSpace().candidates(seed=0)]
    assert first == second
    seeded = space.candidates(seed=42)
    assert all(spec.seed == 42 for spec in seeded)
    assert [spec.candidate_id for spec in seeded] != first


def test_stage_of_maps_candidates_to_stage_labels():
    space = CandidateSpace()
    specs = space.candidates()
    assert space.stage_of(specs[0]) == STAGE_BASELINE
    assert space.stage_of(specs[1]) == STAGE_ENCODER_FUSION
    assert space.stage_of(specs[2]) == STAGE_ENCODER_FUSION
    assert space.stage_of(specs[3]) == STAGE_HEAD_CHUNK
    assert space.stage_of(specs[4]) == STAGE_HEAD_CHUNK
    assert space.stage_of(specs[5]) == STAGE_LR
    assert space.stage_of(space.make(chunk_size=50, lr=3e-3)) == ""
