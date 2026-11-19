from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from a3opt.backends.adaptive_phase import CLOSE, LIFT, OPEN, RETRACT, TRANSFER, AdaptivePhasePolicy, Recipe, load_checkpoint, task_column
from a3opt.difficulty_eval import EpisodeTrace, check_row, run_episode
from a3opt.refinement_eval import sha256


@lru_cache(maxsize=4)
def phase_model(checkpoint: Path) -> dict:
    with np.load(checkpoint, allow_pickle=False) as archive:
        return dict(archive)


class AdaptiveTrace(EpisodeTrace):
    def __init__(self, policy: AdaptivePhasePolicy, first_batch_only: bool):
        super().__init__(policy)
        self.first_batch_only = first_batch_only

    def bind_env(self, env) -> None:
        super().bind_env(env)
        self.initial_z = env.cookie_positions[:, 2]

    def reset(self, context=None) -> None:
        super().reset(context)
        self.finished = False
        self.events: list[dict] = [{}, {}]
        self.first_requested_five = None
        self.column = None
        self.lifted_ids = [np.empty(0, dtype=int), np.empty(0, dtype=int)]

    def act(self, observation: dict, task: str) -> np.ndarray:
        if self.column is None:
            self.column = task_column(task)
        return super().act(observation, task)

    def add_frame(self, observation, action, info, **context) -> None:
        super().add_frame(observation, action, info, **context)
        requested = self.columns == self.column
        current = int(np.count_nonzero(self.target & requested))
        if self.first_requested_five is None and current >= 5:
            self.first_requested_five = self.step
        policy = self.policy
        event = self.events[policy.batch]
        if policy.pending_features is not None:
            event["grip_features"] = policy.pending_features.tolist()
            event["grip_step"] = self.step
        if policy.started and policy.stage == CLOSE:
            event["close_start"] = self.step
        if policy.ended and policy.stage == LIFT:
            positions = context["env"].cookie_positions
            previous_eef = observation["observation.eef_pose"][:3]
            lifted = (positions[:, 2] - self.initial_z > .04) & (np.linalg.norm(positions - previous_eef, axis=1) < .15)
            event["lifted_requested"] = int(np.count_nonzero(lifted & requested))
            event["lifted_other"] = int(np.count_nonzero(lifted & ~requested))
            event["lift_end"] = self.step
            self.lifted_ids[policy.batch] = np.flatnonzero(lifted & requested)
        if policy.ended and policy.stage == TRANSFER:
            env = context["env"]
            moved = self.lifted_ids[policy.batch]
            next_eef = observation["observation.eef_pose"][:3]
            positions = env.cookie_positions[moved]
            near = np.linalg.norm(positions - next_eef, axis=1) < .15
            raised = positions[:, 2] - self.initial_z[moved] > .04
            event["retained_through_transfer"] = int(np.count_nonzero(near & raised))
        if policy.started and policy.stage == OPEN:
            event["open_start"] = self.step
            event["requested_before_open"] = current
        if policy.ended and policy.stage == RETRACT:
            event.setdefault("requested_after_release", current)
            event.setdefault("retract_end", self.step)
            env = context["env"]
            event.setdefault("released_in_target_region", sum(bool(env.privileged_cookie_in_target_region(int(index)))
                             for index in self.lifted_ids[policy.batch]))
            if self.first_batch_only:
                self.finished = True


def adaptive_episode(checkpoint: Path, profile: str, suite: str, seed: int,
                     recipe: Recipe | None = None) -> dict:
    policy = load_checkpoint(checkpoint) if recipe is None else AdaptivePhasePolicy(phase_model(checkpoint), (recipe,) * 4)
    trace = AdaptiveTrace(policy, first_batch_only=recipe is not None)
    row = run_episode(checkpoint, profile, suite, seed, trace=trace)
    row.update(batches=trace.events, retries=policy.retries, first_requested_five=trace.first_requested_five,
               checkpoint_sha256=sha256(checkpoint))
    return row


def check_adaptive_row(job: tuple, row: dict) -> None:
    checkpoint, profile, suite, seed = job[:4]
    check_row((checkpoint, profile, suite, seed), row)
    if row["checkpoint_sha256"] != sha256(checkpoint) or len(row["batches"]) != 2:
        raise ValueError("episode evidence does not match the adaptive checkpoint")
