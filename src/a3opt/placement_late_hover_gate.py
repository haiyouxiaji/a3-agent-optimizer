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

from a3opt import placement_benchmark, placement_compose, placement_descent_agent as descent, placement_search
from a3opt import placement_first_hover as first, placement_hover_agent as hover, placement_hover_guard as guard
from a3opt import placement_routed_agent as routed
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import PLACE, TRANSFER, task_column
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


FEATURES = {"transfer_x": TRANSFER, "descent_x": PLACE}
SCORE_RULES = ("default", "strict-500")


def feature_row(policy):
    phases = policy.actions[hover.BATCH]
    return {name: guard.planned_x(policy.pose, phases, phase) for name, phase in FEATURES.items()}


@dataclass(frozen=True)
class LateHoverGate:
    recipe: str
    feature: str
    direction: int
    threshold_m: float
    column = hover.COLUMN
    batch = hover.BATCH

    def __post_init__(self):
        hover.Hover(self.recipe)
        if self.feature not in FEATURES or self.direction not in (-1, 1) or not np.isfinite(self.threshold_m):
            raise ValueError("invalid second-batch hover gate")

    def record(self):
        return [self.recipe, self.feature, self.direction, self.threshold_m]

    def activates(self, readings):
        return self.direction * (readings[self.feature] - self.threshold_m) >= 0.


class LateHoverPolicy(descent.DescentPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        if len(corrections) > 1:
            raise ValueError("expected one second-batch hover gate")
        self.late_gate = corrections[0] if corrections else None
        super().__init__(checkpoint, focus, scale, incumbent)

    def act(self, observation, task=""):
        gate = self.late_gate
        if (gate is not None and self.actions is not None and self.cursor == (gate.batch, PLACE, 0)
                and not self.retry_remaining and task_column(task) == gate.column and gate.activates(feature_row(self))):
            self.shift_placement(gate.batch, np.asarray(hover.OFFSETS[gate.recipe]))
        return super().act(observation, task)


def routed_panels(folder, checkpoint, timing, common):
    manifest = read_json(folder / "manifest.json")
    model = routed.decode(manifest["corrections"])
    panels, fingerprints = {}, {}
    for name in ("confirmation", "replication"):
        protocol_path = folder / f"{name}-protocol.json"
        protocol = read_json(protocol_path)
        if (sha256(protocol_path) != manifest["files_sha256"][protocol_path.name]
                or any(protocol[key] != value for key, value in common.items())
                or protocol["evaluator_sha256"] != sha256(Path(placement_benchmark.__file__))):
            raise ValueError("routed archive does not match the release and evaluator")
        reserved = protocol["schedule"][name]
        seeds = reserved["random"] if isinstance(reserved, dict) else reserved
        path = folder / f"{name}-rollouts.json.gz"
        raw = gzip.decompress(path.read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        if digest != manifest["rollout_sha256"][path.name]:
            raise ValueError("routed archive rollouts differ from the published hashes")
        pairs = json.loads(raw)
        for arm, corrections in (("reference", ()), ("selected", model)):
            for seed, row in zip(seeds, pairs[arm], strict=True):
                placement_benchmark.check((checkpoint, *timing, seed, corrections), row)
        hover.pair_stats(pairs, model)
        panels[f"routed-{name}"] = pairs
        fingerprints[path.name] = digest
    return model[0], panels, fingerprints


def data_panel(pairs, policy):
    output = []
    for control, selected in zip(pairs["reference"], pairs["selected"], strict=True):
        reference = control["strict-500"]
        if reference["column"] != hover.COLUMN:
            continue
        policy.pose = np.asarray(reference["estimated_pose"])
        policy.actions = policy._phase_actions(hover.COLUMN, policy.recipes[hover.COLUMN - 1])
        output.append({"seed": reference["seed"], "readings": feature_row(policy),
                       "delta": [selected[rule]["requested_score"] - control[rule]["requested_score"] for rule in SCORE_RULES],
                       "complete_delta": [int(selected[rule]["success"]) - int(control[rule]["success"]) for rule in SCORE_RULES],
                       "unsafe": any(selected[rule]["safety_reason"] or selected[rule]["score"] != selected[rule]["requested_score"]
                                     for rule in SCORE_RULES)})
    return output


def effects(rows):
    return {"activations": len(rows), "gain": [sum(row["delta"][index] for row in rows) for index in range(2)],
            "complete_gain": [sum(row["complete_delta"][index] for row in rows) for index in range(2)],
            "helped": sum(min(row["delta"]) > 0 for row in rows), "harmed": sum(min(row["delta"]) < 0 for row in rows),
            "major_losses": sum(min(row["delta"]) <= -placement_search.MAJOR_LOSS for row in rows),
            "unsafe": sum(row["unsafe"] for row in rows)}


def usable(summary):
    return (summary["helped"] >= 2 and min(summary["gain"]) > 0 and min(summary["complete_gain"]) >= 0
            and not (summary["major_losses"] or summary["unsafe"]))


def choose(panels, recipe):
    rows = [row for panel in panels for row in panel]
    candidates = []
    for feature in FEATURES:
        values = sorted({row["readings"][feature] for row in rows})
        for low, high in zip(values, values[1:]):
            for direction in (-1, 1):
                gate = LateHoverGate(recipe, feature, direction, (low + high) / 2.)
                active = [row for row in rows if gate.activates(row["readings"])]
                summary = effects(active)
                groups = [effects([row for row in panel if gate.activates(row["readings"])]) for panel in panels]
                if usable(summary) and not summary["harmed"] and sum(min(group["gain"]) > 0 for group in groups) >= 2:
                    rank = (min(summary["gain"]), min(summary["complete_gain"]), -len(active))
                    candidates.append((rank, gate))
    return max(candidates, key=lambda candidate: candidate[0])[1] if candidates else None


def fit(panels, recipe):
    observed, validation = [], {}
    for name, panel in panels.items():
        gate = choose([rows for label, rows in panels.items() if label != name], recipe)
        active = [] if gate is None else [row for row in panel if gate.activates(row["readings"])]
        observed.extend(active)
        validation[name] = {"rule": gate.record() if gate else None, "always_hover": effects(panel), "gated": effects(active)}
    gated = effects(observed)
    full = effects([row for panel in panels.values() for row in panel])
    gate = choose(list(panels.values()), recipe)
    accept = (gate is not None and usable(gated) and gated["harmed"] < full["harmed"]
              and min(gated["gain"]) >= min(full["gain"]) and sum(min(fold["gated"]["gain"]) > 0 for fold in validation.values()) >= 2)
    return gate if accept else None, {"validation": validation, "always_hover": full, "cross_validated_gate": gated,
                                      "full_fit": gate.record() if gate else None, "admitted": accept}


def run(root, release, late, routed_archive, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected aligned seeds and a positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint, timing = release / frozen["checkpoint"], tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*record) for record in frozen["corrections"])
    common = {"environment": simulator_identity(), "incumbent": frozen["corrections"], "timing": list(timing),
              "rules": RULES, "strict_rules": STRICT_RULES}
    if (common["environment"]["sim_revision"] != frozen["sim_revision"] or sha256(checkpoint) != frozen["checkpoint_sha256"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("late hover gating requires the frozen published controller")
    prior, panels, fingerprints = routed_panels(routed_archive, checkpoint, timing, common)
    late_common = {**common, "reference_sha256": sha256(manifest)}
    protocol, original_fit, independent_hashes = placement_compose.source_refinement(late, hover, late_common)
    if original_fit["selected"] != prior.recipe:
        raise ValueError("the two source experiments must use the same second-batch residual")
    for stage in ("development", "confirmation"):
        seeds = protocol["schedule"][stage]
        receipt = routed.source_panel(late, stage, seeds, checkpoint, timing, (prior,))
        path = late / ("confirmation/rollouts" if stage == "confirmation" else "development") / "results.json"
        panels[f"hover-{stage}"] = read_json(path)
        independent_hashes[stage] = receipt["rollout_sha256"]
    controller = descent.DescentPolicy(checkpoint, *timing, incumbent)
    dataset = {name: data_panel(pairs, controller) for name, pairs in panels.items()}
    training = [row["seed"] for rows in dataset.values() for row in rows]
    development = [seed for seed in range(block, block + 64) if choose_column("random", seed) == hover.COLUMN]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    history = [*root.parent.glob("*/protocol.json"), late / "protocol.json",
               *(routed_archive / f"{name}-protocol.json" for name in ("confirmation", "replication"))]
    if (len(training) != len(set(training)) or reserved & set(training) or any(path.parent != root and
            reserved & hover.scheduled_seeds(read_json(path).get("schedule", {})) for path in history)):
        raise ValueError("late hover requires non-overlapping training and evaluation panels")
    protocol_path = root / "protocol.json"
    _record_once(protocol_path, {**late_common, "schedule": {"training": training, "development": development, "confirmation": confirmation},
                 "sources": {"routed": fingerprints, "hover": independent_hashes}, "recipe": prior.recipe,
                 "features": list(FEATURES), "code_sha256": sha256(Path(__file__)), "confirmation_requires": routed.REQUIREMENTS})
    dataset_path = root / "training" / "dataset.json"
    _record_once(dataset_path, dataset)
    model, summary = fit(dataset, prior.recipe)
    records = [model.record()] if model else []
    fit_path = root / "fit.json"
    _record_once(fit_path, {**summary, "corrections": records, "dataset_sha256": sha256(dataset_path)})
    if model is None:
        return {"status": "reference_retained_after_late_hover_training", **summary}
    corrections = (model,)
    policy_type = partial(LateHoverPolicy, incumbent=incumbent)
    print(f"Agent selected a second-batch hover gate from {len(training)} archived pairs", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                              {"reference": (), "selected": corrections}, policy_type=policy_type)
        results = hover.pair_stats(paired, corrections)
        selection = root / "development-selection.json"
        _record_once(selection, results)
        if not all(placement_search.passes(stats) for stats in results.values()):
            return {"status": "reference_retained_after_late_hover_development", "development": results, "corrections": records}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "seeds": confirmation, "corrections": records,
                     "protocol_sha256": sha256(protocol_path), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)})
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                   corrections, confirmation[:4], policy_type=policy_type)
    if any(row["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for row in final.values()):
        raise ValueError("confirmation requires four-column balance")
    return {"status": "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_late_hover_confirmation",
            "development": results, "confirmation": final, "corrections": records,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Fit a second-batch hover gate from independent historical placement pairs.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--archive", type=Path, default=Path("evidence/routed-20261120"))
    parser.add_argument("--late", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=153000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.late.resolve(), args.archive.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
