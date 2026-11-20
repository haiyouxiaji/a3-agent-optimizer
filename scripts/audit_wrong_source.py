from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import task_column
from a3opt.cli import _run_lock
from a3opt.placement_sim import PlacementTrace
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES

from replay_placement import decode_model, verify_policy_code


class SourceTrace(PlacementTrace):
    def __init__(self, policy, cookie_ids):
        super().__init__(policy)
        self.cookie_ids = cookie_ids

    def reset(self, context=None):
        super().reset(context)
        self.samples = []

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        contacts = {env._cookie_geoms[index]: [] for index in self.cookie_ids}
        for contact in env.data.contact:
            if contact.geom1 in contacts:
                contacts[contact.geom1].append(env.model.geom(contact.geom2).name)
            if contact.geom2 in contacts:
                contacts[contact.geom2].append(env.model.geom(contact.geom1).name)
        self.samples.append({"step": self.step, "batch": self.policy.batch, "stage": self.policy.stage,
                             "frame": self.policy.frame, "eef_m": observation["observation.eef_pose"][:3].tolist(),
                             "cookies": [{"id": index, "xyz_m": env.cookie_positions[index].tolist(),
                                          "in_source": bool(info["cookies_in_source_mask"][index]),
                                          "in_target": bool(info["cookies_in_target_mask"][index]),
                                          "contact_geometries": sorted(set(contacts[env._cookie_geoms[index]]))}
                                         for index in self.cookie_ids]})


def reproduce(checkpoint, timing, constructor, model, expected):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        column = task_column(expected["task_prompt"])
        source_x = np.asarray(env.SOURCE_POSITIONS)[:, 0]
        cookie_ids = [index for index in expected["target_cookie_ids"] if source_x[index] != np.unique(source_x)[column - 1]]
        trace = SourceTrace(constructor(checkpoint, *timing, model), cookie_ids)
        result = engine.run_episode(trace, seed=expected["seed"], env=env, diagnostic=trace)
        measured = {"steps": result.steps, "score": result.score, "success": result.success,
                    "target_cookie_ids": np.flatnonzero(trace.target).tolist(), "source_remaining": result.cookies_in_source,
                    "action_sha256": trace.digest.hexdigest(), "applied_action_sha256": trace.applied_digest.hexdigest(),
                    "terminal": trace.terminal}
        if measured != {key: expected[key] for key in measured} or len(trace.samples) != result.steps:
            raise ValueError("source-motion replay differs from the frozen terminal or action trace")
        history = {}
        for index in cookie_ids:
            sequence = [(sample, item) for sample in trace.samples for item in sample["cookies"] if item["id"] == index]
            history[str(index)] = {"source_exit": next((sample["step"] for sample, row in sequence if not row["in_source"]), None),
                                   "first_target_step": next((sample["step"] for sample, row in sequence if row["in_target"]), None),
                                   "robot_contacts": [sample["step"] for sample, row in sequence
                                                      if any(name.startswith(("L_", "R_")) for name in row["contact_geometries"])]}
        return {"seed": result.seed, "column": column, "wrong_source_ids": cookie_ids, "terminal": measured,
                "history": history, "samples": trace.samples}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Replay off-column terminal cases and locate their first source displacement.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    lock_path, protocol_path = source / "confirmation-lock.json", source / "protocol.json"
    rows_path = source / "confirmation" / "rollouts" / "results.json"
    lock, protocol, paired = map(read_json, (lock_path, protocol_path, rows_path))
    checkpoint, timing = Path(lock["checkpoint"]), lock["timing"]
    constructor, model = decode_model(lock["corrections"], "expert-landing", incumbent=lock["incumbent"],
                                      late_recovery=lock["late_recovery"])
    verify_policy_code(protocol, lock, "expert-landing", constructor)
    if (protocol["environment"] != simulator_identity() or lock["protocol_sha256"] != sha256(protocol_path)
            or lock["checkpoint_sha256"] != sha256(checkpoint)):
        raise ValueError("source-motion replay must use the frozen policy and simulator")
    cases = [row["strict-500"] for row in paired["selected"] if row["strict-500"]["score"] > row["strict-500"]["requested_score"]]
    root = args.out.resolve()
    with _run_lock(root):
        _record_once(root / "protocol.json", {"source_lock_sha256": sha256(lock_path), "source_rollouts_sha256": sha256(rows_path),
                     "code_sha256": sha256(Path(__file__)), "seeds": [row["seed"] for row in cases]})
        results = []
        for case in cases:
            path = root / f"{case['seed']}.json"
            if not path.exists():
                _record_once(path, reproduce(checkpoint, timing, constructor, model, case))
            row = read_json(path)
            results.append({key: row[key] for key in ("seed", "column", "wrong_source_ids", "history")})
        _record_once(root / "result.json", {"cases": results})
    print(f"Verified the physical traces of {len(cases)} wrong-source episodes", flush=True)


if __name__ == "__main__":
    main()
