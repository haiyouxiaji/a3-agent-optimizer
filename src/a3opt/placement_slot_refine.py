from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_descent_agent, placement_hover_agent, placement_search, slot_alignment_fit
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import PLACE, task_column
from a3opt.backends.slot_alignment import FEATURE_COUNT, predict_shift, release_features
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


COLUMN = 2
BATCH = 0
DEVELOPMENT_SCENES = 12
CONFIRMATION_SCENES = 64


@dataclass(frozen=True)
class RowAlignment:
    center: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    column = COLUMN
    batch = BATCH

    def __post_init__(self):
        vectors = (self.center, self.scale, self.weights)
        if (any(len(values) != FEATURE_COUNT for values in vectors) or not np.isfinite(vectors).all()
                or min(self.scale) <= 0 or not np.isfinite(self.bias)):
            raise ValueError("invalid row-alignment parameters")

    def parameters(self):
        return {"center": np.array(self.center), "scale": np.array(self.scale),
                "weights": np.array(self.weights), "bias": self.bias}

    def record(self):
        return [self.column, self.batch, list(self.center), list(self.scale), list(self.weights), self.bias]


class RowPolicy(placement_descent_agent.DescentPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        if len(corrections) > 1:
            raise ValueError("expected at most one first-row correction")
        super().__init__(checkpoint, focus, scale, incumbent)
        self.row_parameters = corrections[0].parameters() if corrections else None

    def act(self, observation, task=""):
        if (self.row_parameters is not None and self.actions is not None and self.cursor == (BATCH, PLACE, 0)
                and task_column(task) == COLUMN):
            features = release_features(self.pose, self.actions[BATCH])
            shift = float(predict_shift(self.row_parameters, features))
            if shift:
                self.shift_placement(BATCH, np.array([0., shift, 0.]))
        return super().act(observation, task)


def training_rows(rows, policy, slots):
    positions = np.unique(slots[:, 0])
    samples = []
    for paired in rows:
        row = paired["strict-500"]
        retract = row["phase_geometry"][BATCH].get("retract")
        if retract is None:
            continue
        first = retract["batches"][BATCH]
        if len(first) != 5 or any(not cookie["scored"] for cookie in first):
            continue
        x_center = np.median([cookie["xyz_m"][0] for cookie in first])
        target_x = positions[np.argmin(abs(positions - x_center))]
        policy.pose = np.array(row["estimated_pose"])
        phases = policy._phase_actions(COLUMN, policy.recipes[COLUMN - 1])
        features = release_features(policy.pose, phases[BATCH])
        snapshot = {"step": retract["step"], "batches": [[{**cookie, "target_xyz_m": cookie["xyz_m"]} for cookie in first]]}
        samples.append({**{key: row[key] for key in ("seed", "column", "score", "requested_score", "safety_reason", "action_sha256")},
                        "first_after_retract": snapshot, "front_slot_xy_m": slots[slots[:, 0] == target_x].tolist(),
                        "first_release_features": features.tolist()})
    return samples


def supports_alignment(stats, confirmation=False):
    strict = stats["strict-500"]
    if strict["complete"] <= strict["reference_complete"]:
        return False
    if any(row["paired_count_delta"] < 0 or row["complete"] < row["reference_complete"]
           or row["major_losses"] or row["wrong_source"] or row["safety_stops"] for row in stats.values()):
        return False
    return not confirmation or (stats["default"]["mean"] > placement_search.TARGET
                                and strict["paired_success_rate_ci95"][0] > 0
                                and all(row["paired_count_ci95"][0] >= 0 for row in stats.values()))


def run(root, release, acquisition, block, workers, fit_only):
    from a3_dual_arm_sim.sim.randomization import choose_column
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected an aligned seed block and a positive worker budget")
    released_path = release / "controller.json"
    released = read_json(released_path)
    checkpoint = release / released["checkpoint"]
    timing = tuple(released["timing"])
    incumbent = tuple(placement_descent_agent.Descent(*entry) for entry in released["corrections"])
    if any(item.column == COLUMN for item in incumbent):
        raise ValueError("first-row calibration requires an independent active column")
    environment = simulator_identity()
    source_protocol_path = acquisition / "protocol.json"
    source = read_json(source_protocol_path)
    scout_path = acquisition / "scout" / "results.json"
    scout_seeds = source["schedule"]["scout"]
    scout = read_json(scout_path)["reference"]
    checkpoint_hash = sha256(checkpoint)
    if (source["reference_sha256"] != sha256(released_path) or source["environment"] != environment
            or source["rules"] != RULES or source["strict_rules"] != STRICT_RULES or tuple(source["timing"]) != timing
            or checkpoint_hash != released["checkpoint_sha256"]
            or source["reference_checkpoint_sha256"] != checkpoint_hash or source["incumbent"] != released["corrections"]
            or environment["sim_revision"] != released["sim_revision"]
            or sha256(Path(placement_descent_agent.__file__)) != released["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != released["evaluator_sha256"]
            or sha256(scout_path) != read_json(acquisition / "acquisition-plan.json")["scout_sha256"]):
        raise ValueError("scout acquisition does not match the frozen published policy and protocol")
    for seed, row in zip(scout_seeds, scout, strict=True):
        placement_benchmark.check((checkpoint, *timing, seed, ()), row)
        if row["strict-500"]["column"] != COLUMN or row != read_json(acquisition / "scout" / "episodes" / "reference" / f"{seed}.json"):
            raise ValueError("expected one audited scout receipt per active-column seed")
    development = [seed for seed in range(block, block + DEVELOPMENT_SCENES * 4) if choose_column("random", seed) == COLUMN]
    confirmation = list(range(block + 400, block + 400 + CONFIRMATION_SCENES))
    if len(development) != DEVELOPMENT_SCENES:
        raise ValueError("development must include every reserved active-column pair")
    history = [source_protocol_path, release / "confirmation-protocol.json", release / "replication-protocol.json",
               *(path for path in root.parent.glob("*/protocol.json") if path.parent != root)]
    for path in history:
        if set(development + confirmation) & placement_hover_agent.scheduled_seeds(read_json(path).get("schedule", {})):
            raise ValueError(f"new evaluation seeds overlap {path}")
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **RULES)
    if sha256(Path(engine.config_path)) != environment["sim_config_sha256"]["cookie_batch.yaml"]:
        raise ValueError("geometry labels require the frozen simulator configuration")
    slots = np.asarray(engine.config.cookie_transfer.target_slots_local_m)
    protocol = root / "protocol.json"
    _record_once(protocol, {"schedule": {"training": scout_seeds, "development": development, "confirmation": confirmation},
                           "profile": "advanced", "environment": environment, "rules": RULES, "strict_rules": STRICT_RULES,
                           "timing": list(timing), "incumbent": released["corrections"], "checkpoint_sha256": checkpoint_hash,
                           "reference_sha256": sha256(released_path), "scout_sha256": sha256(scout_path),
                           "scout_protocol_sha256": sha256(source_protocol_path), "code_sha256": sha256(Path(__file__)),
                           "fit_code_sha256": sha256(Path(slot_alignment_fit.__file__)),
                           "evaluator_sha256": released["evaluator_sha256"], "target_slots_local_m": slots.tolist()})
    policy_type = partial(RowPolicy, incumbent=incumbent)
    rows = training_rows(scout, policy_type(checkpoint, *timing), slots)
    parameters, dataset = slot_alignment_fit.fit_alignment(rows, COLUMN)
    dataset_path = root / "training" / "dataset.json"
    _record_once(dataset_path, {**dataset, "unusable_seeds": sorted(set(scout_seeds) - {row["seed"] for row in rows})})
    model = () if parameters is None else (RowAlignment(tuple(parameters["center"]), tuple(parameters["scale"]),
                                                         tuple(parameters["weights"]), parameters["bias"]),)
    fit_path = root / "fit.json"
    _record_once(fit_path, {"dataset_sha256": sha256(dataset_path), "corrections": [item.record() for item in model]})
    print(f"Audited {len(scout)} scout episodes, kept {len(dataset['samples'])} geometry labels", flush=True)
    if not model:
        return {"status": "reference_retained_after_training"}
    if fit_only:
        return {"status": "candidate_fitted"}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                              {"reference": (), "selected": model}, policy_type=policy_type)
        progress = placement_hover_agent.pair_stats(paired, model)
        selection = root / "development-selection.json"
        _record_once(selection, progress)
        if not supports_alignment(progress):
            return {"status": "reference_retained_after_development", "development": progress}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                     "timing": list(timing), "incumbent": released["corrections"], "corrections": [item.record() for item in model],
                     "seeds": confirmation, "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path),
                     "selection_sha256": sha256(selection)})
        measured = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                      model, confirmation[:4], policy_type=policy_type)
    balance = Counter({str(column): CONFIRMATION_SCENES // 4 for column in range(1, 5)})
    if any(Counter(row["scored_pairs_by_column"]) != balance for row in measured.values()):
        raise ValueError("row-alignment confirmation is not balanced across all four columns")
    return {"status": "candidate_promoted" if supports_alignment(measured, confirmation=True) else "reference_retained_after_confirmation",
            "development": progress, "confirmation": measured}


def main():
    parser = argparse.ArgumentParser(description="Learn first-row target-box alignment from frozen advanced scout receipts.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=142000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.acquisition.resolve(), args.seed_block, args.workers, args.fit_only)
        if result["status"] != "candidate_fitted":
            _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
