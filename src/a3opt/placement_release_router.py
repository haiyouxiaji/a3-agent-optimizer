from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_descent_agent as descent, placement_first_hover as first
from a3opt import placement_contact_feedback as contact
from a3opt import placement_hover_agent as hover, placement_late_hover_gate as late, placement_release_hold as hold
from a3opt import placement_release_recovery as recovery, placement_routed_agent as routed, placement_search
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import PLACE, task_column
from a3opt.backends.release_residual import nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


@dataclass(frozen=True)
class HoldGate:
    recipe: str
    feature: str
    direction: int
    threshold: float

    def __post_init__(self):
        if (self.recipe not in hold.RECIPES or self.feature not in late.FEATURES
                or self.direction not in (-1, 1) or not np.isfinite(self.threshold)):
            raise ValueError("invalid stationary release gate")

    def activates(self, readings):
        value = readings[self.feature]
        return np.isfinite(value) and self.direction * (value - self.threshold) >= 0.

    def record(self):
        return [self.recipe, self.feature, self.direction, self.threshold]


@dataclass(frozen=True)
class ReleaseRoute:
    fallback: recovery.Recovery
    gate: HoldGate | None = None
    column = hover.COLUMN
    batch = hover.BATCH

    def record(self):
        return [self.fallback.record(), self.gate.record() if self.gate else None]


def decode(record):
    fallback, condition = record
    if fallback[:2] != [hover.COLUMN, hover.BATCH]:
        raise ValueError("release route has a different recovery scope")
    gate = HoldGate(*condition) if condition is not None else None
    return ReleaseRoute(recovery.Recovery(*fallback[2:]), gate)


class ReleaseRouterPolicy(recovery.RecoveryPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        if len(corrections) > 1:
            raise ValueError("expected a single release route")
        self.route = corrections[0] if corrections else None
        fallback = (self.route.fallback,) if self.route else ()
        super().__init__(checkpoint, focus, scale, fallback, incumbent=incumbent)

    def act(self, observation, task=""):
        route = self.route
        if (route and route.gate and self.actions is not None and self.cursor == (route.batch, PLACE, 0)
                and not self.retry_remaining and task_column(task) == route.column):
            if route.gate.activates(late.feature_row(self)):
                self.recovery_checked = True
                xyz = np.asarray(observation["observation.eef_pose"][:3], dtype=float)
                hold.replan(self.actions[route.batch], xyz, self.pose[4], route.gate.recipe, nominal_kinematics())
        return super().act(observation, task)


def fallback_outcome(row, fallback):
    trials = row["recovery"]["trials"]
    recipe = fallback.recipe if fallback and fallback.activates(trials["features"]) else "reference"
    return trials["outcomes"][recipe]


def measure(rows, fallback, entry=None, condition=None):
    differences, complete, unsafe, switched = [], 0, 0, 0
    for row in rows:
        baseline = fallback_outcome(row, fallback)
        active = entry is not None and (condition is None or condition.activates(row["readings"]))
        result = row["hold"]["trials"][entry] if active else baseline
        differences.append(len(result["counted_ids"]) - len(baseline["counted_ids"]))
        complete += int(result["success"]) - int(baseline["success"])
        unsafe += bool(result["safety_reason"] or result["wrong_source"])
        switched += active
    return {"gain": sum(differences), "helped": sum(delta > 0 for delta in differences),
            "harmed": sum(delta < 0 for delta in differences), "worst": min(differences, default=0),
            "complete_gain": complete, "unsafe": unsafe, "switched": switched}


def eligible(stats):
    return (stats["gain"] > 0 and stats["helped"] >= 2 and stats["complete_gain"] >= 0
            and stats["worst"] >= -1 and not stats["unsafe"])


def select(panels):
    fallback = recovery.select_gate({name: [row["recovery"] for row in rows] for name, rows in panels.items()})
    rows = [row for panel in panels.values() for row in panel]
    if not fallback:
        return None, fallback
    choices = []
    for name in late.FEATURES:
        values = sorted({row["readings"][name] for row in rows})
        for left, right in zip(values, values[1:]):
            for direction in (-1, 1):
                for recipe in hold.RECIPES:
                    gate = HoldGate(recipe, name, direction, (left + right) / 2.)
                    stats = measure(rows, fallback, recipe, gate)
                    if eligible(stats) and sum(measure(panel, fallback, recipe, gate)["gain"] > 0
                                               for panel in panels.values()) >= 2:
                        rank = (stats["gain"], stats["complete_gain"], -stats["harmed"], -stats["switched"])
                        choices.append((rank, ReleaseRoute(fallback, gate)))
    return max(choices, key=lambda item: item[0], default=(None, None))[1], fallback


def fit(panels):
    folds = {}
    for name, rows in panels.items():
        route, fallback = select({key: panel for key, panel in panels.items() if key != name})
        effects = measure(rows, fallback, route.gate.recipe, route.gate) if route else measure(rows, fallback)
        folds[name] = {"route": route.record() if route else None, "effects": effects,
                       "always_hold": {recipe: measure(rows, fallback, recipe) for recipe in hold.RECIPES}}
    selected, _ = select(panels)
    validation = [fold["effects"] for fold in folds.values()]
    gain = sum(stats["gain"] for stats in validation)
    always = {recipe: sum(fold["always_hold"][recipe]["gain"] for fold in folds.values()) for recipe in hold.RECIPES}
    admitted = (selected is not None and gain > max(0, *always.values())
                and sum(stats["gain"] > 0 for stats in validation) >= 2
                and sum(stats["complete_gain"] for stats in validation) >= 0
                and all(stats["worst"] >= -1 and not stats["unsafe"] for stats in validation))
    return ((selected,) if admitted else ()), {"folds": folds, "fold_gain": gain, "always_hold_gain": always,
                                              "full_fit": selected.record() if selected else None, "admitted": admitted}


def training_input(folder, module, common):
    protocol_path, fit_path, data_path = (folder / name for name in ("protocol.json", "fit.json", "training.json.gz"))
    protocol, parameters = read_json(protocol_path), read_json(fit_path)
    raw = gzip.decompress(data_path.read_bytes())
    if (any(protocol[key] != value for key, value in common.items())
            or sha256(Path(module.__file__)) != protocol["code_sha256"]
            or (module is recovery and sha256(Path(contact.__file__)) != protocol["contact_code_sha256"])
            or hashlib.sha256(raw).hexdigest() != parameters["dataset_sha256"]):
        raise ValueError(f"training archive differs from the frozen inputs: {folder}")
    return json.loads(raw), {"protocol": sha256(protocol_path), "fit": sha256(fit_path), "data": parameters["dataset_sha256"]}


def build_panels(panels, hold_rows, history, controller):
    joined = {}
    for name, rows in panels.items():
        joined[name] = []
        for row in rows:
            seed = row["seed"]
            measured = hold_rows[seed]
            endpoints = (row["continuation"], measured["continuation"])
            if any(endpoints[0][key] != endpoints[1][key] for key in endpoints[0] if key != "command_sha256"):
                raise ValueError(f"counterfactual reference endpoints differ for {seed}")
            controller.pose = np.asarray(history[seed]["estimated_pose"], dtype=float)
            controller.actions = controller._phase_actions(hover.COLUMN, controller.recipes[hover.COLUMN - 1])
            joined[name].append({"seed": seed, "readings": late.feature_row(controller), "hold": measured, "recovery": row})
    return joined


def dependencies():
    modules = (hold, recovery, contact, late, placement_search, placement_benchmark)
    return {module.__name__: sha256(Path(module.__file__)) for module in modules}


def run(root, release, archive, hold_archive, recovery_archive, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected an aligned seed block and positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint, timing = release / frozen["checkpoint"], tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*record) for record in frozen["corrections"])
    common = {"environment": simulator_identity(), "incumbent": frozen["corrections"], "timing": list(timing),
              "rules": RULES, "strict_rules": STRICT_RULES}
    if (common["environment"]["sim_revision"] != frozen["sim_revision"] or sha256(checkpoint) != frozen["checkpoint_sha256"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("release routing requires the frozen published controller")
    published, release_hashes = first.historical_rows(release, (), frozen, common, checkpoint, timing, incumbent)
    _, prior_panels, archive_hashes = late.routed_panels(archive, checkpoint, timing, common)
    archived = [pair["strict-500"] for panel in prior_panels.values() for pair in panel["reference"]]
    history = {row["seed"]: row for row in [*published, *archived]}
    if len(history) != len(published) + len(archived):
        raise ValueError("duplicate reference scene across published panels")
    source_common = {**common, "reference_sha256": sha256(manifest)}
    past_hold, hold_hashes = training_input(hold_archive, hold, source_common)
    past_recovery, recovery_hashes = training_input(recovery_archive, recovery, source_common)
    rows = [row for panel in past_recovery["panels"].values() for row in panel]
    training = [row["seed"] for row in rows]
    if len(training) != len(set(training)):
        raise ValueError("duplicate recovery training case")
    development = [seed for seed in range(block, block + 64) if choose_column("random", seed) == hover.COLUMN]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    prior = [hold_archive / "protocol.json", recovery_archive / "protocol.json", *root.parent.glob("*/protocol.json")]
    if reserved & set(history) or any(path.parent != root and reserved & hover.scheduled_seeds(
            read_json(path).get("schedule", {})) for path in prior):
        raise ValueError("release routing requires untouched evaluation scenes")

    training_seeds = set(training)
    hold_rows = {row["seed"]: row for row in past_hold["rows"] if row["seed"] in training_seeds}
    hold_type = partial(hold.ReleaseHoldPolicy, incumbent=incumbent)
    for row in rows:
        recovery.audit_training((checkpoint, timing, incumbent, row["seed"]), row, history=history)
    for row in hold_rows.values():
        hold.audit_training((checkpoint, timing, hold_type, row["seed"]), row, history=history)
    missing = [seed for seed in training if seed not in hold_rows]
    protocol = root / "protocol.json"
    _record_once(protocol, {**source_common, "code_sha256": sha256(Path(__file__)), "dependencies": dependencies(),
                 "sources": {"release": release_hashes, "routed": archive_hashes, "hold": hold_hashes, "recovery": recovery_hashes},
                 "schedule": {"training": training, "development": development, "confirmation": confirmation},
                 "reused_hold_seeds": [seed for seed in training if seed in hold_rows], "new_hold_seeds": missing,
                 "features": list(late.FEATURES), "confirmation_requires": routed.REQUIREMENTS})
    policy_type = partial(ReleaseRouterPolicy, incumbent=incumbent)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent reuses {len(hold_rows)} audited pairs and collects {len(missing)} missing stationary-release branches", flush=True)
        jobs = [(checkpoint, timing, hold_type, seed) for seed in missing]
        collected = _jobs(pool, hold.collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]),
                          partial(hold.audit_training, history=history))
        hold_rows.update((row["seed"], row) for row in collected)
        controller = descent.DescentPolicy(checkpoint, *timing, incumbent)
        panels = build_panels(past_recovery["panels"], hold_rows, history, controller)
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"panels": panels})
        model, stats = fit(panels)
        records = [item.record() for item in model]
        fit_path = root / "fit.json"
        _record_once(fit_path, {**stats, "corrections": records, "dataset_sha256": sha256(dataset)})
        if not model:
            return {"status": "reference_retained_after_release_router_training", "fit": stats}
        fallback = (ReleaseRoute(model[0].fallback),)
        names = {"reference": (), "selected": model, "recovery": fallback}
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                              names, policy_type=policy_type)
        selection = {name: hover.pair_stats({"reference": paired[name], "selected": paired["selected"]}, model)
                     for name in ("reference", "recovery")}
        selection_path = root / "development-selection.json"
        _record_once(selection_path, selection)
        if not all(placement_search.passes(row) for panel in selection.values() for row in panel.values()):
            return {"status": "reference_retained_after_release_router_development", "development": selection}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "seeds": confirmation, "corrections": records,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection_path)})
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                   model, confirmation[:4], policy_type=policy_type)
    if any(stats["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for stats in final.values()):
        raise ValueError("confirmation requires four-column balance")
    return {"status": "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_release_router_confirmation",
            "confirmation": final, "development": selection, "corrections": records,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Learn when to switch from contact recovery to stationary release.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--archive", type=Path, default=Path("evidence/routed-20261120"))
    parser.add_argument("--hold-archive", type=Path, default=Path("evidence/release-hold-20261121"))
    parser.add_argument("--recovery-archive", type=Path, default=Path("evidence/release-recovery-20261121"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=161000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.archive.resolve(), args.hold_archive.resolve(),
                     args.recovery_archive.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
