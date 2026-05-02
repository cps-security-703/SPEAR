"""
best_ids_model.py
=================
Runtime IDS wrapper used by hierarchical_cosimulation.py and
enhanced_llm_rl_coordinator.py.

After running compare_ids_models.py the best-performing sklearn model (by AUC)
is serialised to  models/best_ids_model.pkl.  BestIDSDetector loads that bundle
at startup and exposes a single  detect(feature_14d)  method that the simulation
calls once per timestep per station.

If the pickle does not exist yet (i.e. the benchmark hasn't been run) the
detector silently falls back to returning (False, 0.0) so the simulation can
still run without ML-IDS active.

Interface expected by the simulation
-------------------------------------
  detector.enabled      → bool   (read by enhanced_llm_rl_coordinator.py)
  detector.anomaly_count → int   (read by enhanced_llm_rl_coordinator.py)
  detector.model_name   → str    (informational)
  detector.detect(feat) → (bool, float)  is_attack, confidence ∈ [0,1]
  detector.reset()      → None   clear window + counter
"""

import os
import sys
import pickle
import numpy as np

ROOT       = os.path.dirname(os.path.abspath(__file__))
_MODEL_PKL = os.path.join(ROOT, "models", "best_ids_model.pkl")

# Pre-register neural model classes into __main__ so that pickle can resolve
# references like '__main__.TransformerIDSWrapper' produced when a training
# script was run directly (i.e. __name__ == '__main__').  Future pkls will
# reference 'ids_neural_models.TransformerIDSWrapper' directly.
try:
    from ids_neural_models import (
        TransformerIDS, TransformerIDSWrapper,
        AutoencoderIDS, LSTMIDSBenchmark,
    )
    _main = sys.modules.get("__main__")
    if _main is not None:
        for _cls in [TransformerIDS, TransformerIDSWrapper,
                     AutoencoderIDS, LSTMIDSBenchmark]:
            if not hasattr(_main, _cls.__name__):
                setattr(_main, _cls.__name__, _cls)
except ImportError:
    pass

_SEQ_LEN     = 10   # must match compare_ids_models.SEQ_LEN
_FEATURE_DIM = 14   # must match compare_ids_models.FEATURE_DIM


class BestIDSDetector:
    """Sliding-window IDS detector backed by the best trained sklearn model.

    The detector maintains a per-station rolling buffer of the last _SEQ_LEN
    feature vectors.  Once the buffer is full it flattens the window, applies
    the stored StandardScaler, and queries the sklearn model's predict_proba.
    """

    def __init__(self, model_path: str = _MODEL_PKL):
        self.enabled       = True
        self.anomaly_count = 0
        self.model_name    = "heuristic"

        self._model     = None
        self._scaler    = None
        self._threshold = 0.5
        self._window    = []          # rolling list of np.ndarray[FEATURE_DIM]
        self._roc_auc   = None

        self._load(model_path)

    # ------------------------------------------------------------------
    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as fh:
                bundle = pickle.load(fh)
            self._model     = bundle["model"]
            self._scaler    = bundle["scaler"]
            self._threshold = float(bundle.get("threshold", 0.5))
            self.model_name = str(bundle.get("model_name", "unknown"))
            self._roc_auc   = bundle.get("roc_auc")
            print(f"[BestIDSDetector] Loaded '{self.model_name}' "
                  f"(AUC={self._roc_auc:.4f}, thr={self._threshold:.3f})")
        except Exception as exc:
            print(f"[BestIDSDetector] Could not load {path}: {exc}  "
                  f"— falling back to heuristic mode.")

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._scaler is not None

    # ------------------------------------------------------------------
    def detect(self, feature_14d) -> tuple:
        """Feed one 14-D timestep and return (is_attack: bool, confidence: float).

        Returns (False, 0.0) until the internal window has accumulated
        _SEQ_LEN steps, or if no model is loaded.
        """
        feat = np.asarray(feature_14d, dtype=np.float32).flatten()

        # Pad or truncate to exactly FEATURE_DIM
        if len(feat) < _FEATURE_DIM:
            feat = np.pad(feat, (0, _FEATURE_DIM - len(feat)))
        else:
            feat = feat[:_FEATURE_DIM]

        self._window.append(feat)
        if len(self._window) > _SEQ_LEN:
            self._window.pop(0)

        if not self.is_loaded or len(self._window) < _SEQ_LEN:
            return False, 0.0

        window_flat = np.array(self._window, dtype=np.float32).flatten().reshape(1, -1)
        try:
            window_scaled = self._scaler.transform(window_flat)
            proba         = float(self._model.predict_proba(window_scaled)[0, 1])
            is_attack     = proba >= self._threshold
            if is_attack:
                self.anomaly_count += 1
            return bool(is_attack), proba
        except Exception as exc:
            print(f"[BestIDSDetector] Inference error: {exc}")
            return False, 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear window and anomaly counter (call between simulation episodes)."""
        self._window.clear()
        self.anomaly_count = 0
