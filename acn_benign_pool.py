

from __future__ import annotations

import os
from typing import Optional, List, Dict

import numpy as np

_LOADER = None
_LOAD_FAILED = False


_ROOT = os.path.dirname(os.path.abspath(__file__))
_ACN_DATA_ROOT = os.path.join(_ROOT, "evcs_data",
                              "ACN-Data-Static-main", "time series data")
_SITES = [
    (os.path.join(_ACN_DATA_ROOT, "caltech", "California_Garage_01"), 80),
    (os.path.join(_ACN_DATA_ROOT, "jpl",     "Arroyo_Garage_01"),     50),
    (_ACN_DATA_ROOT,                                                     6),
]


def _ensure_loaded():

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

    _ensure_loaded()
    if _LOADER is None:
        return None
    try:
        return _LOADER.sample_benign_window(seq_len, system_id=system_id, rng=rng)
    except Exception as exc:
        print(f"[acn_benign_pool] sample_benign_window failed: {exc}")
        return None


def is_available() -> bool:

    _ensure_loaded()
    return _LOADER is not None
