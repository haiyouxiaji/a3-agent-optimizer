from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from a3opt.backends.advanced_phase import AdvancedPhasePolicy
from a3opt.backends.advanced_trajectory import AdvancedTrajectoryPolicy
from a3opt.backends.local_sim import load_policy
from a3opt.backends.trajectory import TrajectoryPolicy

CHECKPOINT_ENV = "A3OPT_CHECKPOINT"
DEVICE_ENV = "A3OPT_DEVICE"
HORIZON_ENV = "A3OPT_EXECUTION_HORIZON"
BC2_FORMAT = "a3opt-bc2-v1"


class _TrajectoryPlugin(TrajectoryPolicy):
    def reset(self, context: Any = None) -> None:
        super().reset()

    def close(self) -> None:
        self.reset()


class _AdvancedTrajectoryPlugin(AdvancedTrajectoryPolicy):
    def close(self) -> None:
        self.reset()


def resolve_checkpoint() -> str:
    checkpoint = os.environ.get(CHECKPOINT_ENV, "").strip()
    if not checkpoint:
        raise RuntimeError(
            f"{CHECKPOINT_ENV} must point to an a3opt checkpoint directory or trajectory .npz"
        )
    return checkpoint


def create_policy():
    device = os.environ.get(DEVICE_ENV, "cpu").strip() or "cpu"
    checkpoint = resolve_checkpoint()
    if Path(checkpoint).is_file() and Path(checkpoint).suffix == ".npz":
        with np.load(checkpoint, allow_pickle=False) as archive:
            format_name = str(archive["format"]) if "format" in archive else None
        if format_name == AdvancedTrajectoryPolicy.name:
            return _AdvancedTrajectoryPlugin(checkpoint)
        if format_name == AdvancedPhasePolicy.name:
            return AdvancedPhasePolicy(checkpoint)
        return _TrajectoryPlugin(checkpoint)
    spec_path = os.path.join(checkpoint, "spec.json")
    if os.path.isfile(spec_path):
        try:
            with open(spec_path, encoding="utf-8") as stream:
                spec = json.load(stream)
        except (OSError, ValueError):
            spec = {}
        if spec.get("format") == BC2_FORMAT:
            from a3opt.backends.bc2 import BC2Policy

            horizon = int(os.environ.get(HORIZON_ENV, spec.get("execution_horizon", 8)) or 8)
            return BC2Policy(checkpoint, device=device, execution_horizon=horizon)
    return load_policy(checkpoint, device=device)


__all__ = [
    "BC2_FORMAT",
    "CHECKPOINT_ENV",
    "DEVICE_ENV",
    "HORIZON_ENV",
    "create_policy",
    "resolve_checkpoint",
]
