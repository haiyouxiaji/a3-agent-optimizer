from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark as benchmark, placement_clearance_refine as refine
from a3opt import placement_descent_agent as descent, placement_early_clearance as early
from a3opt import placement_first_hover as first, placement_first_recovery as history, placement_hover_agent as hover
from a3opt import placement_contact_feedback as contact, placement_routed_agent as routed, placement_search
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, task_column
from a3opt.backends.release_residual import nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256


SECOND = 1
WINDOW = .50
RECIPES = ("raise8", "raise14", "raise20", "inset4-raise14")
SIGNALS = ("vertical_tracking_error_m", "wall_axis_tracking_error_m", "force_change_n", "planned_row_gap_m")
ENDPOINTS = ("score", "requested_score", "target_cookie_ids", "terminal", "success", "steps", "safety_reason",
             "action_sha256", "applied_action_sha256", "source_remaining")


@dataclass(frozen=True)
class Interbatch:
    column: int
    first_recipe: str
    second_recipe: str | None = None
    condition: tuple[int, float, bool] | None = None
    batch = early.BATCH

    def __post_init__(self):
        early.Clearance(self.column, self.first_recipe)
        if self.second_recipe is not None and self.second_recipe not in RECIPES:
            raise ValueError("unknown second-batch clearance recipe")
        if self.condition is not None:
            index, cutoff, above = self.condition
            if self.second_recipe is None or index not in range(len(SIGNALS)) or not np.isfinite(cutoff) or not isinstance(above, bool):
                raise ValueError("invalid second-batch clearance condition")

    def record(self):
        return [self.column, self.batch, self.first_recipe, self.second_recipe,
                list(self.condition) if self.condition is not None else None]

    def activates(self, values):
        if self.second_recipe is None or not np.isfinite(values).all():
            return False
        if self.condition is None:
            return True
        index, cutoff, above = self.condition
        return (values[index] >= cutoff) == above


class InterbatchPolicy(early.EarlyClearancePolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent, late_recovery, probe=False):
        if len(corrections) > 1:
            raise ValueError("expected one coordinated placement correction")
        self.interbatch = corrections[0] if corrections else None
        self.probe_second = probe
        primary = (early.Clearance(self.interbatch.column, self.interbatch.first_recipe),) if self.interbatch else ()
        super().__init__(checkpoint, focus, scale, primary, incumbent=incumbent, late_recovery=late_recovery)

    def reset(self, context=None):
        super().reset(context)
        self.second_force = None

    def second_features(self, observation, task):
        item = self.interbatch
        if (item is None or (item.second_recipe is None and not self.probe_second) or self.actions is None
                or self.cursor is None or self.cursor[:2] != (SECOND, PLACE) or self.retry_remaining
                or task_column(task) != item.column):
            return None
        frame = self.cursor[2]
        lowering = self.actions[SECOND][PLACE]
        if frame == 0:
            self.second_force = contact.force_observation(observation)[:3]
        if frame != max(1, int(WINDOW * len(lowering))) or self.second_force is None:
            return None
        solver = nominal_kinematics()
        measured = np.asarray(observation["observation.eef_pose"][:3])
        error = measured - descent.tool_xyz(solver, lowering[frame - 1, :7])
        first_point = descent.tool_xyz(solver, self.actions[early.BATCH][OPEN][0, :7])
        second_point = descent.tool_xyz(solver, lowering[-1, :7])
        axis = self.wall_axis()
        signed_gap = float((second_point[:2] - first_point[:2]) @ axis)
        outward = axis if signed_gap >= 0 else -axis
        force = contact.force_observation(observation)[:3] - self.second_force
        return np.array([error[2], error[:2] @ outward, np.linalg.norm(force), abs(signed_gap)]), outward

    def act(self, observation, task=""):
        measurement = self.second_features(observation, task)
        if measurement is not None:
            features, outward = measurement
            if self.interbatch.activates(features):
                measured = np.asarray(observation["observation.eef_pose"][:3])
                early.replan(self.actions[SECOND], self.cursor[2], measured, -outward, self.interbatch.second_recipe,
                             nominal_kinematics())
        return super().act(observation, task)


def endpoints(trace):
    if trace.course is None or trace.released is None:
        raise ValueError("a second-batch intervention did not reach both scoring endpoints")
    return {rule: {key: row[key] for key in ENDPOINTS}
            for rule, row in (("default", trace.course), ("strict-500", trace.released))}


class InterbatchTrace(benchmark.CombinedTrace):
    def bind_env(self, env):
        super().bind_env(env)
        self.env = env

    def reset(self, context=None):
        super().reset(context)
        self.trials = None

    def branch(self, observation, task, recipe):
        env = copy(self.env)
        env.data = copy(self.env.data)
        policy = copy(self.policy)
        policy.actions = self.policy.actions.copy()
        policy.actions[SECOND] = [stage.copy() for stage in policy.actions[SECOND]]
        policy.interbatch = replace(policy.interbatch, second_recipe=recipe)
        trace = benchmark.CombinedTrace(policy)
        excluded = {"policy", "env", "cameras", "digest", "applied_digest", "trials"}
        trace.__dict__.update({key: deepcopy(value) for key, value in vars(self).items() if key not in excluded})
        trace.digest, trace.applied_digest = self.digest.copy(), self.applied_digest.copy()
        while trace.step < RULES["max_steps"]:
            action = trace.act(observation, task)
            obs_next, _, terminated, truncated, info = env.step(action)
            trace.add_frame(observation, action, info, env=env)
            observation = obs_next
            if terminated or truncated:
                break
        return endpoints(trace)

    def act(self, observation, task):
        measurement = self.policy.second_features(observation, task)
        if measurement is not None:
            values, _ = measurement
            self.trials = {"features": values.tolist(), "step": self.step,
                           "outcomes": {recipe: self.branch(observation, task, recipe) for recipe in (None, *RECIPES)}}
            self.trials["outcomes"]["first-only"] = self.trials["outcomes"].pop(None)
        return super().act(observation, task)


def collect(checkpoint, timing, policy_type, base, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = InterbatchTrace(policy_type(checkpoint, *timing, (base,), probe=True))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "second_transfer": trace.stages[SECOND].get("transfer"),
                "trials": trace.trials, "continuation": endpoints(trace)}
    finally:
        env.close()


def audit_trial(job, row, *, archived):
    checkpoint, _, _, base, seed = job
    prior = archived[seed]
    data = row["trials"]
    expected = {rule: {key: prior[rule][key] for key in ENDPOINTS} for rule in ("default", "strict-500")}
    if (row["seed"] != seed or row["column"] != base.column or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint)
            or row["second_transfer"] != prior["strict-500"]["phase_geometry"][SECOND]["transfer"]
            or data is None or len(data["features"]) != len(SIGNALS) or not np.isfinite(data["features"]).all()
            or set(data["outcomes"]) != {"first-only", *RECIPES}
            or row["continuation"] != data["outcomes"]["first-only"] or row["continuation"] != expected):
        raise ValueError(f"second-batch counterfactual failed its historical parity audit for seed {seed}")


def effect(reference, candidate):
    paired = [(reference[rule], candidate[rule]) for rule in ("default", "strict-500")]
    return {"gain": min(new["requested_score"] - old["requested_score"] for old, new in paired),
            "complete_delta": min(int(new["success"]) - int(old["success"]) for old, new in paired),
            "unsafe": any(bool(new["safety_reason"]) or new["requested_score"] != new["score"] for _, new in paired)}


def measure(group, model):
    outcomes = [row["effects"][model.second_recipe if model.activates(row["features"]) else "first-only"] for row in group]
    return {"gain": sum(row["gain"] for row in outcomes), "improved": sum(row["gain"] > 0 for row in outcomes),
            "harmed": sum(row["gain"] < 0 for row in outcomes), "unsafe": sum(row["unsafe"] for row in outcomes),
            "complete_delta": sum(row["complete_delta"] for row in outcomes)}


def select(panels, base):
    rows = [row for group in panels.values() for row in group]
    alone = measure(rows, base)
    conditions = [None]
    for index in range(len(SIGNALS)):
        values = sorted({row["features"][index] for row in rows})
        conditions.extend((index, (low + high) / 2., direction) for low, high in zip(values, values[1:]) for direction in (True, False))
    candidates = []
    for recipe in RECIPES:
        for condition in conditions:
            trial = replace(base, second_recipe=recipe, condition=condition)
            values = measure(rows, trial)
            if (values["gain"] > alone["gain"] and values["complete_delta"] >= 0 and not (values["harmed"] or values["unsafe"])
                    and sum(measure(group, trial)["gain"] > 0 for group in panels.values()) >= 2):
                candidates.append(((values["gain"], values["complete_delta"], condition is None), trial))
    return max(candidates, key=lambda item: item[0], default=(None, None))[1]


def fit(panels, base):
    heldout = {}
    for name, rows in panels.items():
        candidate = select({key: values for key, values in panels.items() if key != name}, base)
        heldout[name] = {"correction": candidate.record() if candidate else None,
                         "first_only": measure(rows, base),
                         "combined": measure(rows, candidate) if candidate else {"gain": 0, "improved": 0, "harmed": 0,
                                                                               "unsafe": 0, "complete_delta": 0}}
    choice = select(panels, base)
    admitted = (choice is not None and all(not (row["combined"]["harmed"] or row["combined"]["unsafe"]) for row in heldout.values())
                and sum(row["combined"]["complete_delta"] for row in heldout.values()) >= 0
                and sum(row["combined"]["gain"] for row in heldout.values()) > sum(row["first_only"]["gain"] for row in heldout.values())
                and sum(row["combined"]["gain"] > 0 for row in heldout.values()) >= 2)
    rows = [row for group in panels.values() for row in group]
    return ((choice,) if admitted else ()), {"admitted": admitted, "heldout": heldout,
             "unconditional": {recipe: measure(rows, replace(base, second_recipe=recipe)) for recipe in RECIPES},
             "full_fit": choice.record() if choice else None}


def training_scenes(panels, outcomes):
    groups = {}
    for name, rows in panels.items():
        eligible = [row for row in rows if row["seed"] in outcomes]
        losses = sorted((row for row in eligible if row["gain"] < 0), key=lambda row: (row["gain"], row["seed"]))[:2]
        gains = sorted((row for row in eligible if row["gain"] > 0), key=lambda row: (-row["gain"], row["seed"]))[:2]
        controls = [row for row in eligible if row["gain"] == 0 and outcomes[row["seed"]][0]["strict-500"]["success"]][:1]
        chosen = [row["seed"] for row in (*losses, *gains, *controls)]
        if chosen:
            groups[name] = chosen
    return groups


def dependencies():
    return {**refine.dependencies(), refine.__name__: sha256(Path(refine.__file__))}


def run(root, source, release, recovery_archive, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected an aligned seed block and a positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint, timing = release / frozen["checkpoint"], tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*item) for item in frozen["corrections"])
    parent, recipe, panels, hashes = refine.parent_data(source, checkpoint)
    common = {key: parent[key] for key in ("environment", "incumbent", "timing", "rules", "strict_rules")}
    late_recovery = history.frozen_recovery(recovery_archive, checkpoint, timing, incumbent, common)
    fallback = [item.record() for item in late_recovery]
    if parent["reference_sha256"] != sha256(manifest) or parent["late_recovery"] != fallback:
        raise ValueError("interbatch reference differs from the completed parent run")
    parent_pairs = read_json(source / "development" / "results.json")
    pairs = {new["strict-500"]["seed"]: (old, new) for old, new in zip(parent_pairs["reference"], parent_pairs["selected"], strict=True)
             if "transfer" in new["strict-500"]["phase_geometry"][SECOND]}
    plan = training_scenes(panels, pairs)
    base = Interbatch(recipe.column, recipe.recipe)
    training = [seed for seeds in plan.values() for seed in seeds]
    if len(plan) < 3 or len(training) != len(set(training)):
        raise ValueError("interbatch selection requires distinct samples in at least three source groups")
    development = [seed for seed in range(block, block + 96) if choose_column("random", seed) == base.column]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    archives = [*root.parent.glob("*/protocol.json"), *release.glob("*-protocol.json"), *recovery_archive.glob("*protocol.json")]
    if reserved & set(pairs) or any(reserved & hover.scheduled_seeds(read_json(path).get("schedule", {}))
                                    for path in archives if path.parent != root):
        raise ValueError("coordinated placement requires unused evaluation scenes")
    protocol = root / "protocol.json"
    _record_once(protocol, {**common, "code_sha256": sha256(Path(__file__)), "reference_sha256": sha256(manifest),
                 "checkpoint_sha256": sha256(checkpoint), "dependency_sha256": dependencies(), "source_sha256": hashes,
                 "late_recovery": fallback, "primary": base.record(), "training_groups": plan,
                 "schedule": {"training": training, "development": development, "confirmation": confirmation},
                 "second_batch_window": WINDOW, "signals": SIGNALS, "recipes": RECIPES,
                 "confirmation_requires": routed.REQUIREMENTS})
    policy_type = partial(InterbatchPolicy, incumbent=incumbent, late_recovery=late_recovery)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent is collecting paired second-batch corrections in {len(training)} selected scenes", flush=True)
        jobs = [(checkpoint, timing, policy_type, base, seed) for seed in training]
        archived = {seed: pairs[seed][1] for seed in training}
        samples = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]),
                        partial(audit_trial, archived=archived))
        by_seed = {row["seed"]: row for row in samples}
        panels = {name: [{"seed": seed, "trial": by_seed[seed], "features": by_seed[seed]["trials"]["features"],
                          "effects": {option: effect(pairs[seed][0], terminal) for option, terminal in
                                      by_seed[seed]["trials"]["outcomes"].items()}} for seed in seeds] for name, seeds in plan.items()}
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"panels": panels})
        model, metrics = fit(panels, base)
        fit_path = root / "fit.json"
        records = [item.record() for item in model]
        _record_once(fit_path, {**metrics, "corrections": records, "dataset_sha256": sha256(dataset)})
        if not model:
            return {"status": "reference_retained_after_interbatch_training", "fit": metrics}
        print(f"Agent selected one combined policy for {len(development)} new development pairs", flush=True)
        paired = benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                    {"reference": (), "selected": model}, policy_type=policy_type)
        scores = hover.pair_stats(paired, model)
        selection = root / "development-selection.json"
        _record_once(selection, scores)
        if not all(placement_search.passes(row) for row in scores.values()):
            return {"status": "reference_retained_after_interbatch_development", "development": scores, "corrections": records}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "late_recovery": fallback,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection),
                     "seeds": confirmation, "corrections": records})
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                   confirmation[:4], policy_type=policy_type)
    if any(row["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for row in final.values()):
        raise ValueError("confirmation requires four-column balance")
    status = "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_interbatch_confirmation"
    return {"status": status, "confirmation": final, "corrections": records, "development": scores,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Optimize the second placement after the first-batch clearance correction.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--recovery-archive", type=Path, default=Path("evidence/release-recovery-20261121"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=166400)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.source.resolve(), args.release.resolve(), args.recovery_archive.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
