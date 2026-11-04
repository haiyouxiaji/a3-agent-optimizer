from __future__ import annotations

import os

from a3opt.backends.local_sim import BCPolicy, load_policy

CHECKPOINT_ENV = "A3OPT_CHECKPOINT"
DEVICE_ENV = "A3OPT_DEVICE"


def resolve_checkpoint() -> str:
    checkpoint = os.environ.get(CHECKPOINT_ENV, "").strip()
    if not checkpoint:
        raise RuntimeError(
            f"{CHECKPOINT_ENV} must point to an a3opt behavior-cloning checkpoint directory"
        )
    return checkpoint


def create_policy() -> BCPolicy:
    device = os.environ.get(DEVICE_ENV, "cpu").strip() or "cpu"
    return load_policy(resolve_checkpoint(), device=device)


__all__ = ["CHECKPOINT_ENV", "DEVICE_ENV", "create_policy", "resolve_checkpoint"]
