"""
acn_benign_pool.py
==================
Process-wide singleton wrapper around ACNDataLoader so that the IDS warm-up
sequences in `enhanced_integrated_evcs_system.py` (report evaluation) and
`attack_specific_rl_agents.py` (RL training) sample REAL ACN-Data windows
instead of uniform-random noise.

Usage:
    from acn_benign_pool import sample_benign_window

    window = sample_benign_window(seq_len=10, system_id=3)
    # window is a list of dicts ready for AnomalyDetector.extract_features()
    # or compare_ids_models_update._evcs_features_14d().

The first call lazily loads ACN sessions from
`evcs_data/ACN-Data-Static-main/time series data/`. Subsequent calls reuse
the cached loader.

If ACN data is unavailable, sample_benign_window() returns None — callers
must check for None and fall back to a synthetic warm-up.
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict

import numpy as np

_LOADER = None
_LOAD_FAILED = False

# Site definitions identical to compare_ids_models_update.py:65-68 so the
# warm-up samples come from the same ACN sites the IDS was trained on.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_ACN_DATA_ROOT = os.path.join(_ROOT, "evcs_data",
                              "ACN-Data-Static-main", "time series data")
_SITES = [
    (os.path.join(_ACN_DATA_ROOT, "caltech", "California_Garage_01"), 80),
    (os.path.join(_ACN_DATA_ROOT, "jpl",     "Arroyo_Garage_01"),     50),
    (_ACN_DATA_ROOT,                                                     6),
]


def _ensure_loaded():
    """Load ACN sessions once on first call. Idempotent."""
    global _LOADER, _LOAD_FAILED
    if _LOADER is not None or _LOAD_FAILED:
        return
    try:
        from acn_sim_interface import ACNDataLoader
        loader = ACNDataLoader(_ACN_DATA_ROOT)
        n = loader.load_from_dirs(_SITES)
        if n == 0:
            _LOAD_FAILED = True
            print("[acn_benign_pool] No ACN sessions found - warm-up will be synthetic")
            return
        _LOADER = loader
        print(f"[acn_benign_pool] Loaded {n} real ACN sessions for warm-up sequences")
    except Exception as exc:
        _LOAD_FAILED = True
        print(f"[acn_benign_pool] ACN load failed ({exc}) - warm-up will be synthetic")


def sample_benign_window(seq_len: int = 10,
                         system_id: int = 1,
                         rng: Optional[np.random.Generator] = None,
                         ) -> Optional[List[Dict]]:
    """Return a list of `seq_len` real-ACN benign feature dicts, or None
    if ACN data isn't available."""
    _ensure_loaded()
    if _LOADER is None:
        return None
    try:
        return _LOADER.sample_benign_window(seq_len, system_id=system_id, rng=rng)
    except Exception as exc:
        print(f"[acn_benign_pool] sample_benign_window failed: {exc}")
        return None


def is_available() -> bool:
    """Return True if ACN data is loaded and ready to sample."""
    _ensure_loaded()
    return _LOADER is not None
