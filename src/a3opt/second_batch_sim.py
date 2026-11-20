from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from a3opt.adaptive_sim import AdaptiveTrace, check_adaptive_row
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import LIFT, OPEN, PLACE, RETRACT, TRANSFER
from a3opt.backends.second_batch import load_checkpoint
from a3opt.difficulty_eval import RULES, run_episode
from a3opt.refinement_eval import sha256


PHASE_EVENTS = {LIFT: "lift_end", TRANSFER: "before_descent", PLACE: "before_open", OPEN: "after_open", RETRACT: "after_retract"}


class SecondBatchTrace(AdaptiveTrace):
    def __init__(self, checkpoint: Path):
        super().__init__(load_checkpoint(checkpoint), first_batch_only=False)

    def bind_env(self, env) -> None:
        super().bind_env(env)
        from a3_dual_arm_sim.controllers.placement_order import front_camera_column_order

        slots = np.asarray(env.TARGET_SLOTS_LOCAL)
        far_column = np.unique(slots[:, 0])[front_camera_column_order(env)[1]]
        self.goal = np.r_[slots[slots[:, 0] == far_column].mean(axis=0), env._target_cookie_center_z + .002].tolist()

    def reset(self, context=None) -> None:
        super().reset(context)
        self.prefix = hashlib.sha256()
        self.prefix_steps = 0
        self.placement_events = {}

    def _snapshot(self, env, info) -> dict:
        slots = {int(index): slot for slot, index in enumerate(info["target_slot_occupancy"]) if index >= 0}
        batches = []
        for indices in self.lifted_ids:
            cookies = []
            for index in map(int, indices):
                cookies.append({"id": index, "target_xyz_m": env.privileged_cookie_target_position(index).tolist(),
                                "pad_contacts": list(env.privileged_left_finger_contacts(index)),
                                "in_region": env.privileged_cookie_in_target_region(index),
                                "scored": bool(self.target[index]), "slot": slots.get(index)})
            batches.append(cookies)
        return {"step": self.step, "batches": batches}

    def add_frame(self, observation, action, info, *, env, **context) -> None:
        super().add_frame(observation, action, info, env=env, **context)
        policy = self.policy
        if policy.batch == 0 or policy.stage < PLACE:
            for command in (action, info["applied_action"]):
                self.prefix.update(np.asarray(command, dtype="<f8").tobytes())
            self.prefix_steps = self.step
        if policy.batch == 1 and policy.ended and policy.stage in PHASE_EVENTS:
            self.placement_events[PHASE_EVENTS[policy.stage]] = self._snapshot(env, info)
        if info["success"] or self.safety or self.step == RULES["max_steps"]:
            self.placement_events["terminal"] = self._snapshot(env, info)


def evaluate(checkpoint: Path, profile: str, suite: str, seed: int) -> dict:
    trace = SecondBatchTrace(checkpoint)
    row = run_episode(checkpoint, profile, suite, seed, trace=trace)
    row.update(checkpoint_sha256=sha256(checkpoint), batches=trace.events, retries=trace.policy.retries,
               first_requested_five=trace.first_requested_five, prefix_sha256=trace.prefix.hexdigest(),
               prefix_steps=trace.prefix_steps, placement_events=trace.placement_events, second_batch_goal_m=trace.goal)
    return row


def check_episode(job: tuple, row: dict) -> None:
    check_adaptive_row(job, row)
    events = row["placement_events"]
    names = [name for name in PHASE_EVENTS.values() if name in events] + ["terminal"]
    steps = [events[name]["step"] for name in names]
    if (set(names) != set(events) or steps != sorted(steps) or steps[-1] != row["steps"]
            or not 0 < row["prefix_steps"] <= row["steps"] or len(row["prefix_sha256"]) != 64):
        raise ValueError("invalid second-batch event timeline")
    if "before_descent" in events and row["prefix_steps"] != events["before_descent"]["step"]:
        raise ValueError("action prefix includes placement commands")
    for cookies in events["terminal"]["batches"]:
        if any(cookie["scored"] != (cookie["id"] in row["target_cookie_ids"]) for cookie in cookies):
            raise ValueError("terminal Cookie IDs disagree with score provenance")


def evaluate_schedule(pool, checkpoint: Path, folder: Path, schedule: dict) -> list[dict]:
    _record_once(folder / "request.json", {"checkpoint_sha256": sha256(checkpoint), "seeds": schedule, "profile": "advanced"})
    jobs = [(checkpoint, "advanced", suite, seed) for suite, seeds in schedule.items() for seed in seeds]
    rows = _jobs(pool, evaluate, jobs, folder / "episodes", lambda job: f"{job[2]}/{job[3]}", check_episode)
    _record_once(folder / "results.json", {"rows": rows})
    print(f"{folder.parent.name}/{folder.name}: {len(rows)} episodes", flush=True)
    return rows
