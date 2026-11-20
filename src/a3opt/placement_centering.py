from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_sim
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import PLACE, task_column
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.placement_search import TARGET, compare, confirm_combined, encode, passes, reuse_training
from a3opt.refinement_eval import read_json, sha256


SLOT_X_M = (.028, -.028)
MAX_OFFSET_M = .006


@dataclass(frozen=True)
class Centering:
    column: int
    batch: int
    cookie_minus_tool_x_m: float
    deadband_m: float

    def __post_init__(self):
        if (self.column not in range(1, 5) or self.batch not in (0, 1)
                or not np.isfinite(self.cookie_minus_tool_x_m) or not 0. <= self.deadband_m <= MAX_OFFSET_M):
            raise ValueError("invalid learned lateral placement calibration")

    def offset(self, tool_x_m: float) -> float:
        predicted_cookie_x = tool_x_m + self.cookie_minus_tool_x_m
        excess = predicted_cookie_x - SLOT_X_M[self.batch] - self.deadband_m
        return -float(np.clip(excess, 0., MAX_OFFSET_M))

    def record(self) -> list:
        return list(astuple(self))


class CenteringPolicy(placement_sim.PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.centering = {(item.column, item.batch): item for item in corrections}
        if len(self.centering) != len(corrections):
            raise ValueError("duplicate lateral placement calibration")
        super().__init__(checkpoint, focus, scale)

    def act(self, observation, task=""):
        if self.actions is not None and self.cursor is not None and self.cursor[1:] == (PLACE, 0):
            batch = self.cursor[0]
            item = self.centering.get((task_column(task), batch))
            if item is not None:
                x_position = placement_sim.projected_tool_x(observation["observation.eef_pose"], self.pose)
                shift = item.offset(x_position)
                if shift:
                    self.shift_placement(batch, np.array([shift, 0., 0.]))
        return super().act(observation, task)


def samples(rows: list[dict]) -> list[dict]:
    data = []
    for row in rows:
        if row["safety_reason"]:
            continue
        for batch in range(2):
            transfer = row["phase_geometry"][batch].get("transfer")
            observed = row["descent_observations"][batch]
            if not transfer or observed is None:
                continue
            cookies = transfer["batches"][batch]
            if len(cookies) != 5 or any(cookie["bottom_above_floor_m"] < .035 for cookie in cookies):
                continue
            x_coords = [cookie["xyz_m"][0] for cookie in cookies]
            tool_x = placement_sim.projected_tool_x(observed["eef_xyz_m"], observed["predicted_pose"])
            data.append({"seed": row["seed"], "column": row["column"], "batch": batch, "tool_x_m": tool_x,
                         "cookie_center_x_m": float((min(x_coords) + max(x_coords)) / 2)})
    return data


def fit(rows: list[dict]) -> tuple[tuple[Centering, ...], dict]:
    fitted, diagnostics = [], {}
    for column in range(1, 5):
        for batch in range(2):
            subset = [row for row in rows if (row["column"], row["batch"]) == (column, batch)]
            if len(subset) < 4:
                continue
            deviations = np.asarray([row["cookie_center_x_m"] - row["tool_x_m"] for row in subset])
            errors = [abs(value - np.median(np.delete(deviations, index))) for index, value in enumerate(deviations)]
            deadband = float(np.clip(np.quantile(errors, .75), .001, MAX_OFFSET_M))
            fitted.append(Centering(column, batch, float(np.median(deviations)), deadband))
            diagnostics[f"{column}/{batch}"] = {"seeds": [row["seed"] for row in subset], "held_out_errors_m": list(map(float, errors)),
                                                 "median_absolute_error_m": float(np.median(errors)), "deadband_m": deadband}
    return tuple(fitted), diagnostics


def run(root, reference, timing_run, sources, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and native evaluation audit are required")
    seeds = list(range(block, block + 12))
    confirmation_seeds = list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": [], "development": seeds, "confirmation": {"random": confirmation_seeds}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    acquired = [reuse_training(source, checkpoint, timing, identity) for source in sources]
    train = [row for rows, _ in acquired for row in rows]
    splits["training"] = [row["seed"] for row in train]
    if (len(set(splits["training"])) != len(train) or set(splits["training"]).intersection(seeds + confirmation_seeds)):
        raise ValueError("lateral placement requires unique training seeds outside evaluation")
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "sources": [evidence for _, evidence in acquired],
                 "training_code_sha256": sha256(Path(__file__)), "native_parity_sha256": sha256(parity),
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__)), "slot_x_m": list(SLOT_X_M), "target": TARGET})
    dataset = root / "training" / "dataset.json"
    data = samples(train)
    _record_once(dataset, {"rows": data})
    model, diagnostics = fit(data)
    models = {"second-only": tuple(item for item in model if item.batch == 1), "both-batches": model}
    _record_once(root / "fit.json", {"dataset_sha256": sha256(dataset), "calibration": diagnostics,
                 "models": {name: encode(candidate) for name, candidate in models.items()}})
    if len(model) != 8:
        return {"status": "reference_retained_after_training", "trained_column_batches": len(model)}
    print(f"Fitted lateral calibration from {len(data)} transfer measurements", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, seeds,
                                                {"reference": (), **models}, policy_type=CenteringPolicy)
        development = {name: {rule: compare([row[rule] for row in outcomes["reference"]], [row[rule] for row in outcomes[name]], candidate)
                              for rule in ("default", "strict-500")} for name, candidate in models.items()}
        _record_once(root / "development-selection.json", development)
        passed = [name for name, stats in development.items() if all(passes(row) for row in stats.values())]
        if not passed:
            return {"status": "reference_retained_after_development", "development": development}
        selected = max(passed, key=lambda name: (development[name]["default"]["mean"], development[name]["strict-500"]["complete"],
                                                 -len(models[name])))
        model = models[selected]
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation_seeds, "corrections": encode(model), "variant": selected,
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        confirmation = confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation_seeds, model,
                                        confirmation_seeds[:4], policy_type=CenteringPolicy)
    confirmed = all(passes(stats) for stats in confirmation.values())
    return {"status": "candidate_promoted" if confirmed else "reference_retained_after_confirmation", "variant": selected,
            "active_policy_above_target": confirmed and confirmation["default"]["mean"] > TARGET,
            "development": development, "confirmation": confirmation}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_CENTERING_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("lateral placement policy requires the frozen checkpoint")
    return CenteringPolicy(checkpoint, *lock["timing"], tuple(Centering(*item) for item in lock["corrections"]))


def main():
    parser = argparse.ArgumentParser(description="Fit a lateral placement policy from the Agent's transfer measurements.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-block", type=int, default=114000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), [source.resolve() for source in args.source],
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
