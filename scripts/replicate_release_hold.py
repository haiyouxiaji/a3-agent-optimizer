from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from copy import copy, deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path

from a3opt import placement_benchmark, placement_first_hover, placement_hover_agent
from a3opt.placement_release_recovery import Recovery
from a3opt import placement_routed_agent, placement_sim, placement_search
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import PLACE
from a3opt.backends.advanced_trajectory import pose_label
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, render_frames, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES

from audit_placement_confirmation import audit_confirmation, report_rule, scheduled_seeds
from replay_placement import decode_model, policy_source


NATIVE_FIELDS = ("score", "requested_score", "target_cookie_ids", "terminal", "success", "steps", "safety_reason",
                 "action_sha256", "applied_action_sha256", "source_remaining")


class PairedTrace(placement_benchmark.CombinedTrace):
    def __init__(self, policy, model):
        super().__init__(policy)
        self.model = model
        self.forked = None

    def bind_env(self, env):
        super().bind_env(env)
        self.env = env

    def act(self, observation, task):
        if self.forked is None and self.column == self.model.column and not self.policy.retry_remaining:
            if isinstance(self.model, Recovery):
                signals = self.policy.signal(observation, task)
                diverges = signals is not None and self.model.activates(signals)
            else:
                diverges = self.policy.actions is not None and self.policy.cursor == (self.model.batch, PLACE, 0)
            if diverges:
                self.forked = self.continue_candidate(observation, task)
        return super().act(observation, task)

    def continue_candidate(self, observation, task):
        env = copy(self.env)
        env.data = copy(self.env.data)
        controller = copy(self.policy)
        controller.actions = self.policy.actions.copy()
        controller.actions[self.model.batch] = [phase.copy() for phase in self.policy.actions[self.model.batch]]
        if isinstance(self.model, Recovery):
            controller.recovery = self.model
        else:
            controller.release_hold = self.model
        trace = placement_benchmark.CombinedTrace(controller)
        excluded = {"policy", "env", "cameras", "digest", "applied_digest", "model", "forked"}
        trace.__dict__.update({key: deepcopy(value) for key, value in vars(self).items() if key not in excluded})
        trace.digest, trace.applied_digest = self.digest.copy(), self.applied_digest.copy()
        trace.cameras = lambda names: render_frames(env, names)
        while trace.step < RULES["max_steps"]:
            action = trace.act(observation, task)
            next_observation, _, terminated, truncated, info = env.step(action)
            trace.add_frame(observation, action, info, env=env)
            observation = next_observation
            if terminated or truncated:
                break
        if trace.course is None or trace.released is None:
            raise ValueError("candidate suffix did not reach both fixed score endpoints")
        return trace


def paired_episode(checkpoint, timing, policy_type, model, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        reference_options = {"probe": True} if isinstance(model, Recovery) else {}
        trace = PairedTrace(policy_type(checkpoint, *timing, **reference_options), model)
        started = time.monotonic()
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        if trace.course is None or trace.released is None or trace.step != result.steps:
            raise ValueError("reference did not reach both fixed score endpoints")
        shared = trace.forked is None
        meta = result.randomization
        common = {"checkpoint_sha256": sha256(checkpoint), "focus": timing[0], "scale": timing[1],
                  "seed": seed, "suite": "random", "column": result.source_column, "profile": meta["profile"],
                  "color": meta["cookie_color"], "cameras": meta["cameras"], "light_diffuse": meta["light_diffuse"],
                  "initial_pose": pose_label(meta).tolist(), "estimated_pose": trace.policy.pose.tolist(),
                  "task_prompt": engine._episode_task(result.source_column, seed), "wall_seconds": time.monotonic() - started}
        pairs = {}
        for label, arm, corrections in (("reference", trace, []), ("selected", trace.forked or trace, [model.record()])):
            pairs[label] = {}
            for rule, terminal in (("default", arm.course), ("strict-500", arm.released)):
                stages = [{key: event for key, event in batch.items() if event["step"] <= terminal["steps"]} for batch in arm.stages]
                pairs[label][rule] = {**common, **terminal, "phase_geometry": stages,
                                      "strict": rule != "default", "corrections": corrections}
        return {**pairs, "shared_complete_rollout": shared}
    finally:
        env.close()


def check_pair(job, row):
    checkpoint, timing, _, model, seed = job
    for name, correction in (("reference", ()), ("selected", (model,))):
        placement_benchmark.check((checkpoint, *timing, seed, correction), row[name])
    if row["reference"]["default"]["column"] != model.column and not row["shared_complete_rollout"]:
        raise ValueError("unexpected candidate branch outside the trained source column")


def equivalent(actual, expected):
    for rule, row in expected.items():
        if {key: value for key, value in actual[rule].items() if key != "wall_seconds"} != {
                key: value for key, value in row.items() if key != "wall_seconds"}:
            raise ValueError(f"shared rollout differs from the full {rule} rollout of scene {row['seed']}")


def run(source, output, seed_start, episodes, workers, *, policy_kind="release-hold"):
    origin = audit_confirmation(source, policy_kind)
    initial = origin["confirmation"]
    if not all(placement_search.passes(values) and values["paired_count_ci95"][0] > 0 for values in initial.values()):
        raise ValueError("replication requires a previously verified positive paired gain")
    if seed_start < 0 or seed_start % 4 or episodes <= len(origin["seeds"]) or episodes % 4 or workers < 1:
        raise ValueError("replication requires a larger balanced seed block and a positive worker budget")
    lock_path = source / "confirmation-lock.json"
    lock = read_json(lock_path)
    checkpoint, timing = Path(lock["checkpoint"]), tuple(lock["timing"])
    release = checkpoint.parent
    policy_type, models = decode_model(lock["corrections"], policy_kind, incumbent=lock["incumbent"])
    model, = models
    seeds = list(range(seed_start, seed_start + episodes))
    history = [source / "protocol.json", *output.parent.glob("*/protocol.json"),
               *release.glob("*-protocol.json"), *Path("evidence/routed-20261120").glob("*-protocol.json")]
    for path in history:
        if path.parent != output and set(seeds) & scheduled_seeds(read_json(path).get("schedule", {})):
            raise ValueError(f"replication overlaps prior data in {path.parent}")
    changes = initial["default"]["changes"]
    parity = list(dict.fromkeys([*origin["seeds"][:4], min(changes, key=lambda row: row["delta"])["seed"],
                                max(changes, key=lambda row: row["delta"])["seed"]]))
    protocol_path = output / "protocol.json"
    _record_once(protocol_path, {"environment": simulator_identity(), "rules": RULES, "strict_rules": STRICT_RULES,
                 "policy_kind": policy_kind,
                 "schedule": {"parity": parity, "replication": seeds}, "source_lock_sha256": sha256(lock_path),
                 "source_results_sha256": origin["evaluation_sha256"], "checkpoint_sha256": sha256(checkpoint),
                 "code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "policy_code_sha256": sha256(policy_source(policy_type)), "corrections": lock["corrections"],
                 "confirmation_requires": placement_routed_agent.REQUIREMENTS,
                 "evaluation": "shared_unchanged_prefix_then_separate_physics_when_policy_can_diverge"})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def evaluate(label, cases):
            jobs = [(checkpoint, timing, policy_type, model, seed) for seed in cases]
            return _jobs(pool, paired_episode, jobs, output / label / "episodes", lambda job: str(job[-1]), check_pair)

        print(f"Checking shared-prefix evaluation against {len(parity)} archived full pairs", flush=True)
        replayed = evaluate("parity", parity)
        for seed, pair in zip(parity, replayed, strict=True):
            for arm in ("reference", "selected"):
                expected = read_json(source / "confirmation" / "rollouts" / "episodes" / arm / f"{seed}.json")
                equivalent(pair[arm], expected)
        _record_once(output / "parity" / "audit.json", {"seeds": parity, "matched_full_rollouts": 2 * len(parity)})
        print(f"Agent started {episodes} new four-column pairs", flush=True)
        reports = evaluate("replication", seeds)
        paired = {arm: [pair[arm] for pair in reports] for arm in ("reference", "selected")}
        rollouts = output / "replication" / "rollouts.json"
        _record_once(rollouts, paired)
        audit_seeds = seeds[:4]
        expected = {row["default"]["seed"]: row for row in paired["selected"]}
        native_jobs = [(checkpoint, *timing, "random", seed, models, strict) for seed in audit_seeds for strict in (False, True)]
        native = _jobs(pool, partial(placement_sim.episode, policy_type=policy_type), native_jobs, output / "native-audit",
                       lambda job: f"{int(job[-1])}/{job[4]}", placement_sim.audit)
        for row in native:
            rule = "strict-500" if row["strict"] else "default"
            if any(row[key] != expected[row["seed"]][rule][key] for key in NATIVE_FIELDS):
                raise ValueError(f"native replay differs from scene {row['seed']}")
        _record_once(output / "native-audit" / "audit.json", {"seeds": audit_seeds, "matched_native_replays": len(native),
                     "matched_fields": list(NATIVE_FIELDS)})

    prior = read_json(source / "confirmation" / "rollouts" / "results.json")
    combined = {arm: prior[arm] + paired[arm] for arm in paired}
    replication = placement_hover_agent.pair_stats(paired, models)
    if any(stats["scored_pairs_by_column"] != {str(column): episodes // 4 for column in range(1, 5)}
           for stats in replication.values()):
        raise ValueError("replication requires four-column balance")
    pooled = placement_hover_agent.pair_stats(combined, models)
    decided = placement_first_hover.supports_confirmation(replication)
    for rule in replication:
        replication[rule]["by_column"] = report_rule(paired, rule)
        pooled[rule]["by_column"] = report_rule(combined, rule)
    return {"status": "replication_passed" if decided else "reference_retained_after_replication",
            "replication": replication, "pooled": pooled, "replication_sha256": sha256(rollouts),
            "target_lower_bound": replication["default"]["mean_ci95"][0] > placement_search.TARGET,
            "candidate_suffixes": sum(not pair["shared_complete_rollout"] for pair in reports)}


def main():
    parser = argparse.ArgumentParser(description="Replicate a frozen release policy on a new four-column seed block.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy-kind", choices=("release-hold", "release-recovery"), default="release-hold")
    parser.add_argument("--seed-start", type=int, default=160000)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    output = args.out.resolve()
    with _run_lock(output):
        result = run(args.source.resolve(), output, args.seed_start, args.episodes, args.workers, policy_kind=args.policy_kind)
        _record_once(output / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
