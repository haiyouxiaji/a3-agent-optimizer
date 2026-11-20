from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_descent_agent as descent, placement_early_clearance as early
from a3opt import placement_first_hover as first, placement_first_recovery as history, placement_hover_agent as hover
from a3opt import placement_routed_agent as routed, placement_search
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import PLACE
from a3opt.backends.release_residual import nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementTrace
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


SIGNALS = (*early.SIGNALS, "first_release_wall_axis_m", "planned_batch_gap_m", "target_yaw_rad", "finger_opening")
GROUP_SIZE = 8


@dataclass(frozen=True)
class Guard:
    column: int
    recipe: str
    conditions: tuple[tuple[int, float, bool], ...]
    batch = early.BATCH

    def __post_init__(self):
        early.Clearance(self.column, self.recipe)
        if (not 1 <= len(self.conditions) <= 2 or any(index not in range(len(SIGNALS)) or not np.isfinite(cutoff)
                or not isinstance(above, bool) for index, cutoff, above in self.conditions)):
            raise ValueError("invalid clearance guard")

    def record(self):
        return [self.column, self.batch, self.recipe, [list(condition) for condition in self.conditions]]

    def activates(self, values):
        return np.isfinite(values).all() and all((values[index] >= cutoff) == above
                                                  for index, cutoff, above in self.conditions)


class GuardedClearancePolicy(early.EarlyClearancePolicy):
    def early_signals(self, observation, task):
        values = super().early_signals(observation, task)
        if values is None:
            return None
        solver = nominal_kinematics()
        points = [descent.tool_xyz(solver, phases[PLACE][-1, :7])[:2] for phases in self.actions]
        projected = (points[0] - self.pose[2:4]) @ self.wall_axis()
        gap = (points[1] - points[0]) @ self.wall_axis()
        return np.r_[values, projected, gap, self.pose[4], observation["observation.state"][7]]


class PrefixComplete(Exception):
    pass


class GuardTrace(PlacementTrace):
    def act(self, observation, task):
        values = self.policy.early_signals(observation, task)
        if values is not None:
            self.features = values.tolist()
            raise PrefixComplete
        return super().act(observation, task)


def collect(checkpoint, timing, policy_type, column, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = GuardTrace(policy_type(checkpoint, *timing, probe_column=column))
        try:
            engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        except PrefixComplete:
            return {"seed": seed, "column": trace.column, "step": trace.step, "features": trace.features,
                    "pose": trace.policy.pose.tolist(), "prefix": trace.stages[early.BATCH]["transfer"],
                    "prefix_action_sha256": trace.digest.hexdigest(), "prefix_applied_sha256": trace.applied_digest.hexdigest(),
                    "checkpoint_sha256": sha256(checkpoint)}
        raise ValueError("no early release decision was reached in the current scene")
    finally:
        env.close()


def audit_probe(job, probe, labels):
    checkpoint, _, _, column, seed = job
    row = labels[seed]
    measured = np.asarray(probe["features"])
    if (probe["seed"] != seed or probe["column"] != column or probe["checkpoint_sha256"] != sha256(checkpoint)
            or probe["prefix"] != row["prefix"] or measured.shape != (len(SIGNALS),) or not np.isfinite(measured).all()
            or probe["step"] <= probe["prefix"]["step"] or probe["step"] >= RULES["max_steps"]
            or ("pose" in row and probe["pose"] != row["pose"])
            or ("early_features" in row and (probe["step"] != row["decision_step"]
                or not np.array_equal(measured[:len(early.SIGNALS)], row["early_features"])))):
        raise ValueError(f"public prefix does not reproduce the archived decision for seed {seed}")


def effect(rows, guard=None):
    active = [row for row in rows if guard is not None and guard.activates(row["features"])]
    return {"gain": sum(row["gain"] for row in active), "complete_delta": sum(row["complete_delta"] for row in active),
            "improved": sum(row["gain"] > 0 for row in active), "harmed": sum(row["gain"] < 0 for row in active),
            "unsafe": sum(row["unsafe"] for row in active), "activations": len(active)}


def select(panels, column, recipe):
    rows = [row for group in panels.values() for row in group]
    features = np.asarray([row["features"] for row in rows])
    group_ids = np.repeat(np.arange(len(panels)), [len(group) for group in panels.values()])
    gains = np.array([row["gain"] for row in rows])
    success_changes = np.array([row["complete_delta"] for row in rows])
    improved = gains > 0
    harmful = (gains < 0) | np.array([row["unsafe"] for row in rows])
    if not improved.any():
        return None
    tests = []
    for index in range(len(SIGNALS)):
        values = np.unique(features[:, index])
        for cutoff in (values[:-1] + values[1:]) / 2.:
            upper = features[:, index] >= cutoff
            for above, mask in ((True, upper), (False, ~upper)):
                if len(np.unique(group_ids[mask & improved])) >= 2:
                    tests.append(((index, float(cutoff), above), mask))

    ranking, selected = None, None
    seen = set()
    for test_id, (test, mask) in enumerate(tests):
        variants = [((test,), mask)]
        variants.extend(((test, other), mask & other_mask) for other, other_mask in tests[test_id + 1:])
        for conditions, enabled in variants:
            identity = enabled.tobytes()
            if (identity, len(conditions)) in seen:
                continue
            seen.add((identity, len(conditions)))
            responders = enabled & improved
            if len(np.unique(group_ids[responders])) < 2 or (enabled & harmful).any():
                continue
            complete_gain = int(success_changes[enabled].sum())
            if complete_gain < 0:
                continue
            rank = (int(gains[enabled].sum()), complete_gain, -len(conditions), -int(enabled.sum()))
            if ranking is None or rank > ranking:
                ranking, selected = rank, conditions
    return Guard(column, recipe, selected) if selected is not None else None


def fit(panels, column, recipe):
    heldout = {}
    for name, rows in panels.items():
        selected = select({key: rows for key, rows in panels.items() if key != name}, column, recipe)
        heldout[name] = {"guard": selected.record() if selected else None, "effect": effect(rows, selected)}
    final = select(panels, column, recipe)
    folds = [fold["effect"] for fold in heldout.values()]
    admitted = (final is not None and sum(fold["gain"] > 0 for fold in folds) >= 2
                and sum(fold["complete_delta"] for fold in folds) >= 0
                and all(not (fold["harmed"] or fold["unsafe"]) for fold in folds))
    return (final,) if admitted else (), {"admitted": admitted, "heldout": heldout,
                                           "full_fit": final.record() if final else None}


def parent_data(source, checkpoint):
    protocol = read_json(source / "protocol.json")
    previous = read_json(source / "result.json")
    fit_path, training_path = source / "fit.json", source / "training" / "dataset.json"
    frozen = read_json(fit_path)
    column, batch = protocol["scope"]
    if (previous["status"] != "reference_retained_after_early_clearance_development" or batch != early.BATCH
            or protocol["environment"] != simulator_identity() or protocol["code_sha256"] != sha256(Path(early.__file__))
            or protocol["dependency_sha256"] != early.dependencies() or protocol["checkpoint_sha256"] != sha256(checkpoint)
            or protocol["rules"] != RULES or protocol["strict_rules"] != STRICT_RULES
            or frozen["dataset_sha256"] != sha256(training_path)):
        raise ValueError("guard refinement requires a finished early-clearance development run")
    recipe, = [early.Clearance(item[0], *item[2:]) for item in frozen["corrections"]]
    historical = read_json(training_path)["panels"]
    refit, expected = early.fit(historical, column)
    if (recipe.signal is not None or refit != (recipe,) or previous["corrections"] != [recipe.record()]
            or any(frozen[key] != value for key, value in expected.items())):
        raise ValueError("guard refinement requires the original unconditional correction")
    panels = {}
    for name, members in historical.items():
        panels[name] = []
        for row in members:
            trial = row["trials"]
            old, new = [trial["outcomes"][option] for option in ("reference", recipe.recipe)]
            panels[name].append({"seed": row["seed"], "prefix": row["prefix"], "decision_step": trial["step"],
                                 "early_features": trial["features"], "gain": len(new["counted_ids"]) - len(old["counted_ids"]),
                                 "complete_delta": int(new["success"]) - int(old["success"]),
                                 "unsafe": bool(new["safety_reason"] or new["wrong_source"])})
    development_path = source / "development" / "results.json"
    development = read_json(development_path)
    development_seeds = protocol["schedule"]["development"]
    for arm, model in (("reference", ()), ("selected", (recipe,))):
        for seed, row in zip(development_seeds, development[arm], strict=True):
            if row != read_json(source / "development" / "episodes" / arm / f"{seed}.json"):
                raise ValueError("development aggregate differs from per-scene evidence")
            placement_benchmark.check((checkpoint, *protocol["timing"], seed, model), row)
    if hover.pair_stats(development, (recipe,)) != previous["development"]:
        raise ValueError("failed development decision does not reproduce")
    for index, (old, new) in enumerate(zip(development["reference"], development["selected"], strict=True)):
        reference, selected = old["strict-500"], new["strict-500"]
        if reference["column"] != column or reference["phase_geometry"][batch].get("transfer") is None:
            raise ValueError("cannot align a development scene to the early correction")
        terminal = tuple((old[rule], new[rule]) for rule in ("default", "strict-500"))
        panels.setdefault(f"development-{index // GROUP_SIZE + 1}", []).append({
            "seed": reference["seed"], "pose": reference["estimated_pose"], "prefix": reference["phase_geometry"][batch]["transfer"],
            "gain": min(trial["requested_score"] - control["requested_score"] for control, trial in terminal),
            "complete_delta": min(int(trial["success"]) - int(control["success"]) for control, trial in terminal),
            "unsafe": any(trial["safety_reason"] or trial["score"] != trial["requested_score"] for _, trial in terminal)})
    hashes = {path.name: sha256(path) for path in (source / "protocol.json", source / "result.json", fit_path,
                                                  training_path, development_path)}
    return protocol, recipe, panels, hashes


def dependencies():
    return {**early.dependencies(), early.__name__: sha256(Path(early.__file__))}


def run(root, source, release, recovery_archive, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected an aligned seed block and a positive worker budget")
    manifest = release / "controller.json"
    reference = read_json(manifest)
    checkpoint = release / reference["checkpoint"]
    timing = tuple(reference["timing"])
    incumbent = tuple(descent.Descent(*item) for item in reference["corrections"])
    parent, recipe, panels, fingerprints = parent_data(source, checkpoint)
    common = {key: parent[key] for key in ("environment", "incumbent", "timing", "rules", "strict_rules")}
    late_recovery = history.frozen_recovery(recovery_archive, checkpoint, timing, incumbent, common)
    fallback = [item.record() for item in late_recovery]
    if (sha256(manifest) != parent["reference_sha256"] or fallback != parent["late_recovery"]
            or sha256(Path(placement_benchmark.__file__)) != reference["evaluator_sha256"]):
        raise ValueError("refinement reference differs from the completed parent run")
    labels = {row["seed"]: row for rows in panels.values() for row in rows}
    if len(labels) != sum(map(len, panels.values())):
        raise ValueError("refinement training has duplicate scenes")
    development = [seed for seed in range(block, block + 96) if choose_column("random", seed) == recipe.column]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    archives = [*root.parent.glob("*/protocol.json"), *release.glob("*-protocol.json"), *recovery_archive.glob("*protocol.json")]
    if reserved & set(labels) or any(reserved & hover.scheduled_seeds(read_json(path).get("schedule", {}))
                                     for path in archives if path.parent != root):
        raise ValueError("guard refinement requires new development and confirmation blocks")
    protocol_path = root / "protocol.json"
    _record_once(protocol_path, {**common, "code_sha256": sha256(Path(__file__)), "reference_sha256": sha256(manifest),
                 "checkpoint_sha256": sha256(checkpoint), "dependency_sha256": dependencies(), "parent_sha256": fingerprints,
                 "late_recovery": fallback, "scope": [recipe.column, early.BATCH], "parent_correction": recipe.record(),
                 "schedule": {"training": list(labels), "development": development, "confirmation": confirmation},
                 "signals": SIGNALS, "confirmation_requires": routed.REQUIREMENTS,
                 "training_groups": {name: [row["seed"] for row in rows] for name, rows in panels.items()}})

    policy_type = partial(GuardedClearancePolicy, incumbent=incumbent, late_recovery=late_recovery)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent is resampling {len(labels)} prefixes with archived full paired outcomes", flush=True)
        jobs = [(checkpoint, timing, policy_type, recipe.column, seed) for seed in labels]
        probes = _jobs(pool, collect, jobs, root / "training" / "prefixes", lambda job: str(job[-1]),
                       partial(audit_probe, labels=labels))
        indexed = {probe["seed"]: probe for probe in probes}
        paired = {name: [{**row, **indexed[row["seed"]]} for row in rows] for name, rows in panels.items()}
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"panels": paired})
        model, metrics = fit(paired, recipe.column, recipe.recipe)
        records = [item.record() for item in model]
        fit_path = root / "fit.json"
        _record_once(fit_path, {**metrics, "corrections": records, "dataset_sha256": sha256(dataset)})
        if not model:
            return {"status": "reference_retained_after_clearance_guard_training", "fit": metrics}

        print(f"Agent selected a guard for {len(development)} new development pairs", flush=True)
        dev_pairs = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                 {"reference": (), "selected": model}, policy_type=policy_type)
        scores = hover.pair_stats(dev_pairs, model)
        selection = root / "development-selection.json"
        _record_once(selection, scores)
        if not all(placement_search.passes(row) for row in scores.values()):
            return {"status": "reference_retained_after_clearance_guard_development", "development": scores, "corrections": records}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": reference["corrections"], "late_recovery": fallback,
                     "protocol_sha256": sha256(protocol_path), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection),
                     "seeds": confirmation, "corrections": records})
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                   confirmation[:4], policy_type=policy_type)
    if any(stats["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for stats in final.values()):
        raise ValueError("confirmation requires four-column balance")
    status = "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_clearance_guard_confirmation"
    return {"status": status, "confirmation": final, "corrections": records, "development": scores,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Fit a selective early clearance guard from audited positive and negative pairs.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--recovery-archive", type=Path, default=Path("evidence/release-recovery-20261121"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=165600)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.source.resolve(), args.release.resolve(), args.recovery_archive.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
