from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from copy import copy, deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path

from a3opt import placement_benchmark, placement_first_hover, placement_hover_agent
from a3opt.placement_expert_grasp import Grasp
from a3opt.placement_expert_landing import Landing
from a3opt.placement_release_recovery import Recovery
from a3opt.placement_source_guard import SourceGuard
from a3opt import placement_routed_agent, placement_sim, placement_search
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import CLOSE, PLACE, PRE_CLOSE
from a3opt.backends.advanced_trajectory import pose_label
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, render_frames, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES

from audit_placement_confirmation import audit_confirmation, audit_expert_landing, report_rule, scheduled_seeds
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
        if self.model is not None and self.forked is None and self.column == self.model.column and not self.policy.retry_remaining:
            if isinstance(self.model, Recovery):
                signals = self.policy.signal(observation, task)
                diverges = signals is not None and self.model.activates(signals)
            elif isinstance(self.model, Landing):
                diverges = self.policy.actions is not None and self.policy.cursor == (
                    self.model.batch, PLACE, max(1, len(self.policy.actions[self.model.batch][PLACE]) // 4))
            elif isinstance(self.model, (Grasp, SourceGuard)):
                stage = PRE_CLOSE if isinstance(self.model, Grasp) else CLOSE
                diverges = self.policy.actions is not None and self.policy.cursor == (self.model.batch, stage, 0)
            else:
                diverges = self.policy.actions is not None and self.policy.cursor == (self.model.batch, PLACE, 0)
            if diverges:
                self.forked = self.continue_candidate(observation, task)
        return super().act(observation, task)

    def continue_candidate(self, observation, task):
        env = copy(self.env)
        env.data = copy(self.env.data)
        controller = copy(self.policy)
        controller.actions = [[phase.copy() for phase in batch] for batch in self.policy.actions]
        controller.retries = self.policy.retries.copy()
        if isinstance(self.model, Recovery):
            controller.recovery = self.model
        elif isinstance(self.model, Landing):
            controller.landing = self.model
        elif isinstance(self.model, Grasp):
            controller.grasp = self.model
        elif isinstance(self.model, SourceGuard):
            controller.guard = self.model
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
    from a3_dual_arm_sim.sim.randomization import choose_column
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        models = model if isinstance(model, tuple) else (model,)
        column = choose_column("random", seed)
        active = [item for item in models if item.column == column]
        if len(active) > 1:
            return {"reference": placement_benchmark.episode(checkpoint, *timing, seed, (), policy_type=policy_type),
                    "selected": placement_benchmark.episode(checkpoint, *timing, seed, models, policy_type=policy_type),
                    "shared_complete_rollout": False}
        current = active[0] if active else None
        reference_options = {"probe": True} if isinstance(current, Recovery) else {}
        trace = PairedTrace(policy_type(checkpoint, *timing, **reference_options), current)
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
        for label, arm, corrections in (("reference", trace, []), ("selected", trace.forked or trace, [item.record() for item in models])):
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
    models = model if isinstance(model, tuple) else (model,)
    for name, correction in (("reference", ()), ("selected", models)):
        placement_benchmark.check((checkpoint, *timing, seed, correction), row[name])
    if row["reference"]["default"]["column"] not in {item.column for item in models} and not row["shared_complete_rollout"]:
        raise ValueError("unexpected candidate branch outside the trained source column")


def equivalent(actual, expected):
    for rule, row in expected.items():
        if {key: value for key, value in actual[rule].items() if key != "wall_seconds"} != {
                key: value for key, value in row.items() if key != "wall_seconds"}:
            raise ValueError(f"shared rollout differs from the full {rule} rollout of scene {row['seed']}")


def resume_expert_confirmation(source, workers):
    result_path = source / "result.json"
    if result_path.is_file():
        return audit_confirmation(source, "expert-landing")["status"]
    protocol_path = source / "protocol.json"
    lock_path = source / "confirmation-lock.json"
    fit_path = source / "fit.json"
    selection_path = source / "development-selection.json"
    protocol, lock, fit = map(read_json, (protocol_path, lock_path, fit_path))
    checkpoint, timing = Path(lock["checkpoint"]), tuple(lock["timing"])
    seeds = lock["seeds"]
    corrections, scores = audit_expert_landing(source, checkpoint, protocol, fit)
    if (scores is None or not all(placement_search.passes(stats) for stats in scores.values())
            or any(lock[key] != protocol[key] for key in ("incumbent", "late_recovery", "timing", "checkpoint_sha256"))
            or lock["corrections"] != corrections or seeds != protocol["schedule"]["confirmation"]
            or any(lock[key] != sha256(path) for key, path in (("fit_sha256", fit_path), ("protocol_sha256", protocol_path),
                                                               ("selection_sha256", selection_path)))):
        raise ValueError("the landing candidate differs from the pre-evaluation lock")
    policy_type, models = decode_model(corrections, "expert-landing", incumbent=lock["incumbent"],
                                       late_recovery=lock["late_recovery"])
    model, = models
    folder = source / "confirmation"
    receipts = folder / "rollouts" / "episodes"
    parity = list(dict.fromkeys([*protocol["schedule"]["development"][:4],
                                *[item["seed"] for item in scores["default"]["changes"]]]))[:6]
    _record_once(folder / "shared-prefix-protocol.json", {"lock_sha256": sha256(lock_path),
                 "code_sha256": sha256(Path(__file__)), "parity_seeds": parity, "confirmation_seeds": seeds})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def evaluate(label, cases):
            jobs = [(checkpoint, timing, policy_type, model, seed) for seed in cases]
            return _jobs(pool, paired_episode, jobs, folder / label, lambda job: str(job[-1]), check_pair)

        print(f"Verifying shared-prefix results against {len(parity)} full development pairs", flush=True)
        for seed, pair in zip(parity, evaluate("shared-prefix-parity", parity), strict=True):
            for arm in ("reference", "selected"):
                equivalent(pair[arm], read_json(source / "development" / "episodes" / arm / f"{seed}.json"))
        _record_once(folder / "shared-prefix-audit.json", {"seeds": parity, "matched_full_rollouts": 2 * len(parity)})
        pending = [seed for seed in seeds if any(not (receipts / arm / f"{seed}.json").is_file()
                                                for arm in ("reference", "selected"))]
        print(f"Completing the frozen four-column panel, {len(pending)} pairs remaining", flush=True)
        for seed, pair in zip(pending, evaluate("shared-prefix-rollouts", pending), strict=True):
            for arm in ("reference", "selected"):
                path = receipts / arm / f"{seed}.json"
                if path.is_file():
                    equivalent(pair[arm], read_json(path))
                else:
                    _record_once(path, pair[arm])
        final = placement_search.confirm_combined(pool, folder, checkpoint, timing, seeds, models, seeds[:4],
                                                   policy_type=policy_type)
    if any(stats["scored_pairs_by_column"] != dict.fromkeys(map(str, range(1, 5)), len(seeds) // 4) for stats in final.values()):
        raise ValueError("confirmation requires four equally represented columns")
    promoted = placement_first_hover.supports_confirmation(final)
    status = "candidate_promoted" if promoted else "reference_retained_after_expert_landing_confirmation"
    _record_once(result_path, {"status": status, "confirmation": final, "development": scores, "corrections": corrections,
                              "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET})
    return audit_confirmation(source, "expert-landing")["status"]


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
    policy_type, models = decode_model(lock["corrections"], policy_kind, incumbent=lock["incumbent"],
                                       late_recovery=lock.get("late_recovery", ()))
    model, = models
    seeds = list(range(seed_start, seed_start + episodes))
    history = [source / "protocol.json", *output.parent.glob("*/protocol.json"),
               *release.glob("*-protocol.json"), *Path("evidence/routed-20261120").glob("*-protocol.json")]
    if policy_kind == "expert-landing":
        history.extend(release.parent.glob("*/*protocol.json"))
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
    parser.add_argument("--out", type=Path)
    parser.add_argument("--policy-kind", choices=("release-hold", "release-recovery", "expert-landing"), default="release-hold")
    parser.add_argument("--resume-confirmation", action="store_true")
    parser.add_argument("--seed-start", type=int, default=160000)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.resume_confirmation:
        if args.policy_kind != "expert-landing" or args.out is not None or args.workers < 1:
            parser.error("resuming a confirmation requires expert-landing and a positive worker budget")
        source = args.source.resolve()
        with _run_lock(source):
            result = resume_expert_confirmation(source, args.workers)
        print(result, flush=True)
        return
    if args.out is None:
        parser.error("replication requires --out")
    output = args.out.resolve()
    with _run_lock(output):
        result = run(args.source.resolve(), output, args.seed_start, args.episodes, args.workers, policy_kind=args.policy_kind)
        _record_once(output / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
