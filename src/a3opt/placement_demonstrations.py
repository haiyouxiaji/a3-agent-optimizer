from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from a3opt import advanced_sim, placement_benchmark, placement_pooled
from a3opt.advanced_agent import _audit_expert, _jobs, _npz, _record_once
from a3opt.backends.adaptive_phase import APPROACH, PLACE, RETRACT, TRANSFER
from a3opt.backends.advanced_phase import align_phases
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.placement_hybrid import decode, development_pairs
from a3opt.placement_search import TARGET, compare, confirm_combined, encode, passes
from a3opt.placement_sim import Correction, PlacementPolicy
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import ReleaseTimingPolicy


DATA = ("poses", "traces", "lengths", "columns", "batches", "seeds")
ATTEMPTS_PER_COLUMN = 2


@dataclass(frozen=True)
class Demonstrations:
    column: int
    checkpoint: str
    digest: str

    @property
    def batch(self):
        return 0

    def record(self):
        return [self.column, self.batch, self.checkpoint, self.digest]


def blend_prefix(trajectory, previous):
    length = min(3, len(trajectory))
    weights = np.linspace(0., 1., length + 1)[1:, None]
    trajectory[:length] = (1. - weights) * previous + weights * trajectory[:length]


class DemonstrationPolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        demos = [item for item in corrections if isinstance(item, Demonstrations)]
        super().__init__(checkpoint, focus, scale, tuple(item for item in corrections if isinstance(item, Correction)))
        self.demo_columns = {item.column for item in demos}
        self.augmented = None
        archives = {(item.checkpoint, item.digest) for item in demos}
        if archives:
            if len(archives) != 1:
                raise ValueError("placement demonstrations must share their frozen training dataset")
            path, digest = archives.pop()
            if sha256(Path(path)) != digest:
                raise ValueError("placement demonstration dataset changed")
            self.augmented = ReleaseTimingPolicy(checkpoint, focus, scale)
            with np.load(path, allow_pickle=False) as archive:
                self.augmented.model = {**self.augmented.model,
                                        **{key: np.concatenate((self.model[key], archive[key])) for key in DATA}}

    def _phase_actions(self, column, recipe):
        original = super()._phase_actions(column, recipe)
        if column not in self.demo_columns:
            return original
        self.augmented.pose = self.pose
        adapted = self.augmented._phase_actions(column, recipe)
        for batch in range(2):
            original[batch][PLACE:] = adapted[batch][PLACE:]
            blend_prefix(original[batch][PLACE], original[batch][TRANSFER][-1])
        blend_prefix(original[1][APPROACH], original[0][RETRACT][-1])
        return original


def collection_plan(pairs, option):
    candidates = {column: [] for column in range(1, 5)}
    for pair in pairs:
        row = pair["reference"]
        if row["fork"] is None or row["safety_reason"] or row["wrong_source"]:
            continue
        endpoint = row["fork"]["outcomes"][option]["physics"]
        count = len(endpoint["counted_ids"])
        if count < 10:
            candidates[row["column"]].append((count, row["seed"]))
    return [(seed, column) for column, rows in candidates.items()
            for _, seed in sorted(rows)[:ATTEMPTS_PER_COLUMN]]


def fit(rows, root):
    training = {key: [] for key in DATA}
    rejected = {}
    for row in rows:
        if row["batches"] != [0, 1] or not row["success"] or row["score"] != 10:
            rejected[str(row["seed"])] = "expert did not complete both training batches"
            continue
        with np.load(root / row["shard"], allow_pickle=False) as archive:
            if (int(archive["seed"]), int(archive["column"])) != (row["seed"], row["column"]):
                raise ValueError("expert training shard provenance changed")
            try:
                phases = [align_phases(archive[f"actions_{batch}"], archive[f"phases_{batch}"]) for batch in (0, 1)]
            except ValueError as error:
                rejected[str(row["seed"])] = str(error)
                continue
            for batch, (trace, lengths) in enumerate(phases):
                values = (archive["pose"], trace, lengths, row["column"], batch, row["seed"])
                for key, value in zip(DATA, values, strict=True):
                    training[key].append(value)
    return {key: np.asarray(values) for key, values in training.items()}, rejected


def run(root, reference, timing_run, source, parent, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and verified joint evaluator are required")
    parent_protocol_path, parent_fit_path = parent / "protocol.json", parent / "fit.json"
    parent_protocol, pooled_fit = read_json(parent_protocol_path), read_json(parent_fit_path)
    if not pooled_fit["selected"]:
        return {"status": "reference_retained_after_training"}
    development = parent_protocol["schedule"]["development"]
    confirmation = list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    pairs, evidence = placement_pooled.data(source, identity, checkpoint, timing, development + confirmation)
    if (pooled_fit["training_source"] != evidence or parent_protocol["training_source"] != evidence
            or set(confirmation).intersection(development + parent_protocol["schedule"]["confirmation"]["random"])
            or parent_protocol["evaluator_sha256"] != sha256(Path(placement_benchmark.__file__))):
        raise ValueError("expert collection requires disjoint splits and a compatible parent training set")
    plan = collection_plan(pairs, pooled_fit["selected"])
    splits["training"] = [seed for seed, _ in plan]
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "collection_plan": [list(item) for item in plan],
                 "training_source": evidence, "shared_development_protocol_sha256": sha256(parent_protocol_path),
                 "parent_fit_sha256": sha256(parent_fit_path), "parity_sha256": sha256(parity),
                 "training_code_sha256": sha256(Path(__file__)), "expert_code_sha256": sha256(Path(advanced_sim.__file__)),
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__)), "target_mean_strictly_above": TARGET})
    print(f"Reserved {len(plan)} targeted expert attempts", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        shards = root / "training" / "shards"
        shards.mkdir(parents=True, exist_ok=True)
        jobs = [(seed, column, shards) for seed, column in plan]
        rows = _jobs(pool, advanced_sim.collect_expert, jobs, root / "training" / "episodes", lambda job: str(job[0]), _audit_expert)
        dataset, rejected = fit(rows, shards)
        artifact = root / "training" / "phase_demonstrations.npz"
        if not artifact.is_file():
            _npz(artifact, **dataset)
        else:
            with np.load(artifact, allow_pickle=False) as existing:
                if any(not np.array_equal(existing[key], dataset[key]) for key in DATA):
                    raise ValueError("resumed expert dataset differs from the training receipts")
        demos = tuple(Demonstrations(int(column), str(artifact), sha256(artifact)) for column in np.unique(dataset["columns"]))
        pooled = decode(pooled_fit["corrections"])
        model = (*demos, *pooled)
        _record_once(root / "fit.json", {"attempts": rows, "accepted_seeds": np.unique(dataset["seeds"]).tolist(),
                     "rejected": rejected, "dataset_sha256": sha256(artifact), "corrections": encode(model)})
        print(f"Retained {len(dataset['seeds'])} completed expert trajectories", flush=True)
        if not demos:
            return {"status": "reference_retained_after_training"}
        observed = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                {"selected": model}, policy_type=DemonstrationPolicy)
        parents, parent_hash = development_pairs(parent, parent_protocol, checkpoint, timing, development, pooled, timeout=1800)
        scores = {name: {rule: compare([row[rule] for row in parents[arm]], [row[rule] for row in observed["selected"]], model)
                         for rule in ("default", "strict-500")} for name, arm in (("reference", "reference"), ("pooled", "selected"))}
        _record_once(root / "development-selection.json", {"scores": scores, "parent_results_sha256": parent_hash})
        if not all(passes(stats) for group in scores.values() for stats in group.values()):
            return {"status": "reference_retained_after_development", "development": scores}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model, confirmation[:4],
                                   policy_type=DemonstrationPolicy)
    primary = checked["default"]
    above = primary["mean"] > TARGET
    accepted = above and all(passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation",
            "target_point_estimate": above, "target_lower_bound": primary["mean_ci95"][0] > TARGET,
            "active_policy_above_target": accepted, "development": scores, "confirmation": checked}


def main():
    parser = argparse.ArgumentParser(description="Collect expert data on training failures and retrain placement trajectories.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-block", type=int, default=121000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.source.resolve(), args.parent.resolve(),
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
