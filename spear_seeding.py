#!/usr/bin/env python3
"""
spear_seeding.py — one knob to make a full SPEAR / autonomous run reproducible.

Reviewer #109A objected that the reported "1800 samples" share a replay buffer
and are therefore NOT independent (not i.i.d.).  The rebuttal commits to
"independently seeded runs".  This module provides that: a single integer seed
(env var SPEAR_SEED, or passed explicitly) that seeds every stochastic source
used in a run — Python `random`, NumPy, PyTorch (CPU + CUDA), and, via SB3's
own `set_random_seed`, the Gym/SB3 action-sampling and network-init RNGs.

Running the SAME pipeline under N DIFFERENT seeds yields N statistically
independent trajectories.  Aggregating across those seeds (see
spear_multiseed_runner.py + spear_sample_efficiency.py) is what lets you report
a pooled-SD Cohen's d that survives review, instead of treating correlated
circles from one run as independent samples.

Usage
-----
    from spear_seeding import seed_everything, get_run_seed
    seed = seed_everything()          # reads SPEAR_SEED, default 0
    # ... or ...
    seed_everything(1234)

Per-agent seeding
-----------------
`seed_everything` sets the GLOBAL RNGs.  To also seed each SB3 agent
deterministically (recommended so agent init is reproducible), pass the value of
`get_run_seed()` as the `seed=` kwarg when the agents are constructed, or call
`seed_sb3_agent(agent, seed)` after construction.
"""
from __future__ import annotations
import os
import random
from typing import Optional

_ACTIVE_SEED: Optional[int] = None


def get_run_seed() -> int:
    """The seed in effect for this process (after seed_everything), or the
    SPEAR_SEED env value, or 0."""
    if _ACTIVE_SEED is not None:
        return _ACTIVE_SEED
    try:
        return int(os.environ.get("SPEAR_SEED", "0"))
    except ValueError:
        return 0


def seed_everything(seed: Optional[int] = None, *, deterministic_torch: bool = False) -> int:
    """Seed all RNGs used across the pipeline.

    Args:
        seed: the seed. If None, read env SPEAR_SEED (default 0).
        deterministic_torch: if True, also force cuDNN deterministic algorithms
            (slower; only needed for bit-exact GPU reproducibility).

    Returns:
        the integer seed actually applied.
    """
    global _ACTIVE_SEED
    if seed is None:
        try:
            seed = int(os.environ.get("SPEAR_SEED", "0"))
        except ValueError:
            seed = 0
    seed = int(seed)
    _ACTIVE_SEED = seed

    # PYTHONHASHSEED affects set/dict ordering in subprocesses; set for children.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        pass

    # SB3 seeds Gym spaces' action sampling + its own helpers.
    try:
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(seed)
    except Exception:
        pass

    print(f"   🎲 seed_everything: all RNGs seeded with SPEAR_SEED={seed}")
    return seed


def seed_sb3_agent(agent, seed: Optional[int] = None) -> None:
    """Seed a constructed SB3 agent (and its env's action space) in place.
    Safe no-op if the agent doesn't expose `.set_random_seed`."""
    if seed is None:
        seed = get_run_seed()
    try:
        agent.set_random_seed(int(seed))
    except Exception:
        pass


if __name__ == "__main__":
    s = seed_everything()
    print(f"Applied seed {s}; get_run_seed()={get_run_seed()}")
