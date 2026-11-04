from __future__ import annotations

from typing import Iterable

from a3opt.contracts import CandidateSpec
from a3opt.space import CandidateSpace


class StagedPlanner:
    def __init__(self, space: CandidateSpace | None = None) -> None:
        self.space = space if space is not None else CandidateSpace()

    @staticmethod
    def _seed_for(parent: object) -> int:
        if parent is None:
            return 0
        candidate = getattr(parent, "candidate", parent)
        if candidate is None:
            return 0
        seed = getattr(candidate, "seed", 0)
        try:
            return int(seed)
        except (TypeError, ValueError):
            return 0

    def propose_stage(
        self,
        parent: object = None,
        used_candidate_ids: Iterable[str] | None = None,
        space: CandidateSpace | None = None,
    ) -> tuple[str, CandidateSpec] | None:
        active_space = space if space is not None else self.space
        used = {str(item) for item in (used_candidate_ids or ())}
        seed = self._seed_for(parent)
        for stage, spec in active_space.staged_candidates(seed=seed):
            if spec.candidate_id in used:
                continue
            if active_space.validate(spec):
                continue
            return stage, spec
        return None

    def propose(
        self,
        parent: object = None,
        used_candidate_ids: Iterable[str] | None = None,
        space: CandidateSpace | None = None,
    ) -> CandidateSpec | None:
        staged = self.propose_stage(parent, used_candidate_ids, space)
        if staged is None:
            return None
        return staged[1]
