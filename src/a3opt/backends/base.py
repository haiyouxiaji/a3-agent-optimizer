from __future__ import annotations

from typing import Any

from a3opt.backends.stub import StubBackend


def make_backend(name: str, **kwargs: Any) -> Any:
    key = name.strip().lower().replace("-", "_")
    if key in ("stub", "simulated", "offline"):
        return StubBackend(**kwargs)
    if key in ("local_sim", "local", "sim"):
        from a3opt.backends.local_sim import LocalSimBackend

        return LocalSimBackend(**kwargs)
    if key in ("a3", "a3_sim_cli"):
        from a3opt.backends.a3_sim import A3SimCliBackend

        return A3SimCliBackend(**kwargs)
    raise ValueError(f"unknown backend: {name}")
