from __future__ import annotations

import math
from dataclasses import replace

from a3opt.contracts import (
    ACTION_HEADS,
    CHUNK_SIZES,
    ENCODERS,
    FUSIONS,
    IMAGE_SIZES,
    LEARNING_RATES,
    CandidateSpec,
)

STAGE_BASELINE = "baseline"
STAGE_ENCODER_FUSION = "encoder_fusion"
STAGE_HEAD_CHUNK = "head_chunk"
STAGE_LR = "lr"
STAGE_ORDER = (STAGE_BASELINE, STAGE_ENCODER_FUSION, STAGE_HEAD_CHUNK, STAGE_LR)
SPACE_NAME = "a3_policy_space"


class CandidateSpace:
    name = SPACE_NAME

    def __init__(self, family: str = "bc_policy", state_dim: int = 16) -> None:
        self.family = str(family)
        self.state_dim = int(state_dim)
        self.encoders = tuple(ENCODERS)
        self.fusions = tuple(FUSIONS)
        self.action_heads = tuple(ACTION_HEADS)
        self.chunk_sizes = tuple(CHUNK_SIZES)
        self.image_sizes = tuple(IMAGE_SIZES)
        self.learning_rates = tuple(LEARNING_RATES)

    def slug(self, spec: CandidateSpec) -> str:
        return (
            f"{spec.family}-{spec.encoder}-{spec.fusion}-{spec.action_head}"
            f"-k{int(spec.chunk_size)}-lr{float(spec.lr):g}"
            f"-img{int(spec.image_size)}-sd{int(spec.state_dim)}-s{int(spec.seed)}"
        )

    def make(
        self,
        encoder: str = "tiny_cnn",
        fusion: str = "concat",
        action_head: str = "chunk",
        chunk_size: int = 20,
        lr: float = 1e-3,
        image_size: int = 64,
        seed: int = 0,
    ) -> CandidateSpec:
        draft = CandidateSpec(
            candidate_id="",
            encoder=str(encoder),
            fusion=str(fusion),
            action_head=str(action_head),
            chunk_size=int(chunk_size),
            lr=float(lr),
            image_size=int(image_size),
            state_dim=int(self.state_dim),
            seed=int(seed),
            family=self.family,
        )
        return replace(draft, candidate_id=self.slug(draft))

    def baseline(self, seed: int = 0) -> CandidateSpec:
        return self.make(seed=seed)

    def validate(self, spec: CandidateSpec) -> list[str]:
        errors: list[str] = []
        if spec.encoder not in self.encoders:
            errors.append(f"encoder {spec.encoder!r} not in {self.encoders}")
        if spec.fusion not in self.fusions:
            errors.append(f"fusion {spec.fusion!r} not in {self.fusions}")
        if spec.action_head not in self.action_heads:
            errors.append(f"action_head {spec.action_head!r} not in {self.action_heads}")
        if int(spec.chunk_size) not in self.chunk_sizes:
            errors.append(f"chunk_size {spec.chunk_size} not in {self.chunk_sizes}")
        if int(spec.image_size) not in self.image_sizes:
            errors.append(f"image_size {spec.image_size} not in {self.image_sizes}")
        if not any(
            math.isclose(float(spec.lr), float(rate), rel_tol=1e-9, abs_tol=1e-15)
            for rate in self.learning_rates
        ):
            errors.append(f"lr {spec.lr} not in {self.learning_rates}")
        if int(spec.state_dim) <= 0:
            errors.append(f"state_dim {spec.state_dim} must be positive")
        if int(spec.seed) < 0:
            errors.append(f"seed {spec.seed} must be non-negative")
        if not str(spec.candidate_id):
            errors.append("candidate_id must be non-empty")
        if not str(spec.family):
            errors.append("family must be non-empty")
        if spec.encoder == "tiny_cnn" and int(spec.image_size) <= 0:
            errors.append("tiny_cnn requires image_size > 0")
        if spec.encoder == "state_mlp" and int(spec.image_size) != 0:
            errors.append("state_mlp requires image_size == 0")
        if spec.action_head == "chunk" and int(spec.chunk_size) <= 1:
            errors.append("action_head 'chunk' requires chunk_size > 1")
        if spec.action_head == "mlp" and int(spec.chunk_size) != 1:
            errors.append("action_head 'mlp' requires chunk_size == 1")
        if spec.fusion == "film" and int(spec.image_size) <= 0:
            errors.append("fusion 'film' requires image_size > 0")
        return errors

    def staged_candidates(self, seed: int = 0) -> list[tuple[str, CandidateSpec]]:
        staged: list[tuple[str, CandidateSpec]] = [
            (STAGE_BASELINE, self.baseline(seed=seed)),
            (STAGE_ENCODER_FUSION, self.make(encoder="tiny_cnn", fusion="film", seed=seed)),
            (STAGE_ENCODER_FUSION, self.make(encoder="state_mlp", fusion="concat", image_size=0, seed=seed)),
            (STAGE_HEAD_CHUNK, self.make(action_head="mlp", chunk_size=1, seed=seed)),
            (STAGE_HEAD_CHUNK, self.make(action_head="chunk", chunk_size=50, seed=seed)),
            (STAGE_LR, self.make(lr=3e-3, seed=seed)),
        ]
        return staged

    def candidates(self, seed: int = 0) -> list[CandidateSpec]:
        return [spec for _, spec in self.staged_candidates(seed)]

    def stage_of(self, spec: CandidateSpec) -> str:
        for stage, candidate in self.staged_candidates(seed=int(spec.seed)):
            if candidate.candidate_id == spec.candidate_id:
                return stage
        return ""

    def grid_size(self) -> int:
        return int(
            len(self.encoders)
            * len(self.fusions)
            * len(self.action_heads)
            * len(self.chunk_sizes)
            * len(self.image_sizes)
            * len(self.learning_rates)
        )
