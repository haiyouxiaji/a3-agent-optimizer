from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_pooled, placement_search, placement_tail, placement_train
from a3opt.advanced_agent import _jobs, _npz, _record_once
from a3opt.backends.adaptive_phase import LIFT, PLACE, TRANSFER, task_column
from a3opt.backends.advanced_trajectory import visual_features
from a3opt.backends.trajectory import FRONT
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_agent import reference_setup
from a3opt.placement_sim import PlacementPolicy, PlacementTrace
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES


FRAME_SIZE = 256
CAPTURE = (1, LIFT, 0)


def target_mask(image):
    import cv2

    outline = visual_features(image)[0][:8].reshape(4, 2)
    if not np.any(outline):
        return None
    center = outline.mean(axis=0)
    points = np.rint(center + .84 * (outline - center)).astype(np.int32)
    if cv2.contourArea(points) < 80:
        return None
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 1)
    return mask.astype(bool)


def chroma_pixels(image, mask):
    import cv2

    pixels = cv2.resize(image, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)[mask].astype(float)
    return np.ptp(pixels, axis=1) / np.maximum(pixels.max(axis=1), 40.)


@dataclass(frozen=True)
class VisualPlacement:
    column: int
    recipe: str
    maximum_chroma_change: float

    def __post_init__(self):
        if (self.column not in range(1, 5) or self.recipe not in placement_tail.OFFSETS or self.recipe == "reference"
                or not 0. <= self.maximum_chroma_change <= 1.):
            raise ValueError("invalid visual placement checkpoint")

    @property
    def batch(self):
        return 1

    def record(self):
        return [self.column, self.recipe, self.maximum_chroma_change]


class VisualPlacementPolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, sample=False):
        self.routing = {item.column: item for item in corrections}
        if len(self.routing) != len(corrections):
            raise ValueError("duplicate visual placement correction")
        self.sample = sample
        super().__init__(checkpoint, focus, scale)

    def reset(self, context=None):
        super().reset(context)
        self.mask = self.background = self.chroma_change = None
        self.initial_image = self.sample_image = None

    @property
    def camera_names(self):
        if self.actions is None or (self.mask is not None and self.cursor == CAPTURE):
            return ("front",)
        return ()

    def act(self, observation, task=""):
        column = task_column(task)
        correction = self.routing.get(column)
        if self.actions is None and (self.sample or correction):
            if self.sample:
                self.initial_image = observation[FRONT].copy()
            self.mask = target_mask(observation[FRONT])
            if self.mask is not None:
                self.background = chroma_pixels(observation[FRONT], self.mask)
        elif self.mask is not None and self.cursor == CAPTURE:
            if self.sample:
                self.sample_image = observation[FRONT].copy()
            observed = chroma_pixels(observation[FRONT], self.mask)
            self.chroma_change = float(np.mean(np.maximum(0., observed - self.background)))
        elif (correction and self.chroma_change is not None and self.cursor == (1, PLACE, 0)
              and self.chroma_change <= correction.maximum_chroma_change):
            self.shift_placement(1, np.asarray(placement_tail.OFFSETS[correction.recipe]))
        return super().act(observation, task)


class VisualSample(PlacementTrace):
    def reset(self, context=None):
        super().reset(context)
        self.post_first_count = self.suffix = self.outcome = None

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.policy.cursor == CAPTURE:
            self.post_first_count = int(np.count_nonzero(self.target & (self.columns == self.column)))
        if self.suffix is not None:
            self.suffix.update(placement_train.command_bytes(action, info))
            if self.safety or self.step == RULES["max_steps"]:
                self.outcome = placement_train.outcome(env, info, np.empty(0, dtype=int), self.suffix.hexdigest())
        if (self.policy.batch == 1 and self.policy.stage == TRANSFER and self.policy.ended
                and not self.safety and self.step < RULES["max_steps"]):
            self.suffix = hashlib.sha256()


def collect(checkpoint, focus, scale, shards, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = VisualSample(VisualPlacementPolicy(checkpoint, focus, scale, sample=True))
        score = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        image_sha256 = None
        if trace.policy.sample_image is not None:
            path = shards / f"{seed}.npz"
            _npz(path, initial=trace.policy.initial_image, observed=trace.policy.sample_image)
            image_sha256 = sha256(path)
        geometry = trace.stages[1].get("transfer", {}) if trace.suffix is not None else {}
        return {"seed": seed, "column": score.source_column, "profile": score.randomization["profile"],
                "count": score.score, "chroma_change": trace.policy.chroma_change, "post_first_count": trace.post_first_count,
                "outcome": trace.outcome, "prefix": geometry.get("prefix_sha256"), "image_sha256": image_sha256,
                "checkpoint_sha256": sha256(checkpoint), "timing": [focus, scale]}
    finally:
        env.close()


def audit(job, row, expected):
    checkpoint, focus, scale, shards, seed = job
    prior = expected[seed]
    if (row["seed"] != seed or row["column"] != prior["column"] or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != [focus, scale]
            or row["count"] != prior["score"] or row["outcome"] != prior["continuation"]
            or row["prefix"] != (prior["fork"]["prefix_sha256"] if prior["fork"] else None)):
        raise ValueError("acquiring visual feedback changed the paired physical rollout")
    observed, counted = row["chroma_change"], row["post_first_count"]
    if (observed is not None and (not 0. <= observed <= 1. or counted is None
                                 or row["image_sha256"] != sha256(shards / f"{seed}.npz"))
            or (counted is not None and counted not in range(11))):
        raise ValueError("invalid first-placement training label")


def threshold(rows):
    empty = [row["chroma_change"] for row in rows if row["post_first_count"] <= 1]
    normal = [row["chroma_change"] for row in rows if row["post_first_count"] >= 4]
    if not empty or len(normal) < 4 or max(empty) >= min(normal):
        return None
    return float((max(empty) + min(normal)) / 2.)


def fit(rows, source):
    usable = [row for row in rows if row["chroma_change"] is not None and source[row["seed"]]["fork"] is not None]
    cutoff = threshold(usable)
    labels = {}
    for row in usable:
        heldout = threshold([other for other in usable if other["seed"] != row["seed"]])
        labels[row["seed"]] = heldout is not None and row["chroma_change"] <= heldout
    options = {}
    if cutoff is not None:
        for recipe in placement_tail.OFFSETS:
            if recipe == "reference":
                continue
            effects = []
            for row in usable:
                seed = row["seed"]
                if row["chroma_change"] > cutoff:
                    continue
                observed = source[seed]["fork"]["outcomes"]
                original, trial = (observed[name]["physics"] for name in ("reference", recipe))
                gain = len(trial["counted_ids"]) - len(original["counted_ids"])
                unsafe = bool(observed[recipe]["wrong_source"] or trial["safety_reason"]
                              or trial["source_remaining"] != original["source_remaining"]
                              or trial["filled_slots"] < original["filled_slots"])
                effects.append({"seed": seed, "column": row["column"], "delta": gain, "unsafe": unsafe, "crossfit_trigger": labels[seed]})
            options[recipe] = effects
    eligible = [recipe for recipe, effects in options.items() if all(item["delta"] >= 0 and not item["unsafe"] for item in effects)
                and len({item["column"] for item in effects if item["delta"] > 0 and item["crossfit_trigger"]}) >= 2]
    winner = max(eligible, key=lambda recipe: (sum(item["delta"] for item in options[recipe]),
                                                -sum(map(abs, placement_tail.OFFSETS[recipe]))), default=None)
    model = () if winner is None else tuple(VisualPlacement(column, winner, cutoff) for column in range(1, 5))
    return model, {"cutoff": cutoff, "usable": len(usable), "crossfit_triggers": labels, "options": options, "recipe": winner}


def paired_stats(rows, model):
    return {rule: placement_search.compare([episode[rule] for episode in rows["reference"]],
                                          [episode[rule] for episode in rows["selected"]], model)
            for rule in ("default", "strict-500")}


def run(root, reference, timing_run, source, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and verified joint evaluator are required")
    development, confirmation = list(range(block, block + 12)), list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    pairs, evidence = placement_pooled.data(source, identity, checkpoint, timing, development + confirmation)
    expected = {row["reference"]["seed"]: row["reference"] for row in pairs}
    splits["training"] = list(expected)
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "training": evidence, "observations": [FRONT],
                 "code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "parity_sha256": sha256(parity), "target_mean_strictly_above": placement_search.TARGET})
    print(f"Collecting a second front-camera observation on {len(expected)} locked training scenes", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        shards = root / "training" / "frames"
        shards.mkdir(parents=True, exist_ok=True)
        jobs = [(checkpoint, *timing, shards, seed) for seed in expected]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]),
                     lambda job, row: audit(job, row, expected))
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"rows": rows})
        model, fit_report = fit(rows, expected)
        _record_once(root / "fit.json", {**fit_report, "corrections": placement_search.encode(model), "dataset_sha256": sha256(dataset)})
        print(f"Audited {len(rows)} visual training receipts; selected {fit_report['recipe']}", flush=True)
        if not model:
            return {"status": "reference_retained_after_training", "training": fit_report}
        models = {"reference": (), "selected": model}
        outputs = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development, models,
                                               policy_type=VisualPlacementPolicy)
        dev = paired_stats(outputs, model)
        _record_once(root / "development-selection.json", dev)
        if not all(placement_search.passes(stats) for stats in dev.values()):
            return {"status": "reference_retained_after_development", "development": dev}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": placement_search.encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                   confirmation[:4], policy_type=VisualPlacementPolicy)
    main_score = checked["default"]
    accepted = main_score["mean"] > placement_search.TARGET and all(placement_search.passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation", "development": dev,
            "confirmation": checked, "active_policy_above_target": accepted,
            "target_point_estimate": main_score["mean"] > placement_search.TARGET,
            "target_lower_bound": main_score["mean_ci95"][0] > placement_search.TARGET}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_VISUAL_PLACEMENT_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("visual placement requires the frozen checkpoint")
    return VisualPlacementPolicy(checkpoint, *lock["timing"], tuple(VisualPlacement(*item) for item in lock["corrections"]))


def main():
    parser = argparse.ArgumentParser(description="Train a second-batch placement router from public camera observations.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-block", type=int, default=127000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.source.resolve(), args.parity.resolve(),
                     args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
