from __future__ import annotations

import os
from pathlib import Path

from a3opt import placement_source_guard as guard
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES

from replay_placement import decode_model, verify_policy_code


def load_policy(lock_path, parent_lock_path, checkpoint_override=None):
    protocol_path = lock_path.parent / "protocol.json"
    fit_path = lock_path.parent / "fit.json"
    parent_protocol_path = parent_lock_path.parent / "protocol.json"
    protocol, lock, parent_protocol, parent_lock = map(read_json, (protocol_path, lock_path, parent_protocol_path, parent_lock_path))
    checkpoint = checkpoint_override or Path(lock["checkpoint"])
    if (protocol["environment"] != simulator_identity() or (protocol["rules"], protocol["strict_rules"]) != (RULES, STRICT_RULES)
            or protocol["code_sha256"] != sha256(Path(guard.__file__)) or lock["protocol_sha256"] != sha256(protocol_path)
            or lock["fit_sha256"] != sha256(fit_path) or lock["seeds"] != protocol["schedule"]["confirmation"]
            or parent_lock["protocol_sha256"] != sha256(parent_protocol_path) or protocol["parent_lock_sha256"] != sha256(parent_lock_path)
            or sha256(checkpoint) != lock["checkpoint_sha256"]
            or any(lock[key] != protocol[key] or lock[key] != parent_lock[key] for key in ("incumbent", "late_recovery", "timing", "checkpoint_sha256"))):
        raise ValueError("source policy differs from the frozen simulator, controller, or source lock")
    parent_policy, anchors = decode_model(parent_lock["corrections"], "expert-landing", incumbent=parent_lock["incumbent"],
                                          late_recovery=parent_lock["late_recovery"])
    verify_policy_code(parent_protocol, parent_lock, "expert-landing", parent_policy)
    if parent_protocol["dependency_sha256"] != guard.landing.dependencies():
        raise ValueError("release policy dependencies differ from the frozen evaluation")
    *release_records, source_record = lock["corrections"]
    source = guard.SourceGuard(*source_record)
    if (release_records != [anchor.record() for anchor in anchors] or source_record[:2] != protocol["scope"]
            or source_record != read_json(fit_path)["guard"] or source.recipe not in protocol["options"]):
        raise ValueError("source policy corrections differ from the frozen placement and grasp scopes")
    return guard.SourceGuardPolicy(checkpoint, *lock["timing"], (*anchors, source), **parent_policy.keywords)


def upstream_policy():
    lock_path = Path(os.environ["A3OPT_PLACEMENT_LOCK"]).resolve(strict=True)
    parent_lock_path = Path(os.environ["A3OPT_PLACEMENT_PARENT_LOCK"]).resolve(strict=True)
    checkpoint = os.environ.get("A3OPT_PLACEMENT_CHECKPOINT")
    return load_policy(lock_path, parent_lock_path, Path(checkpoint).resolve(strict=True) if checkpoint else None)
