"""
ACN network layout for node-level (Phase 2) attack targeting.

Phase 2 model
-------------
Each distribution system (DS) contains N_NODES_PER_DS "nodes".  Each node is a
FULL ACN network — either Caltech (~54 EVSEs) or JPL (~52 EVSEs) — assigned
randomly but SEEDED so the 6×10 = 60-network layout is fixed across runs
(reproducible for publication).

Attacking a node = perturbing that network's CMS input, which impacts ALL EVSEs
connected to that network.  The DQN's job in Phase 2 is to pick WHICH node
(network) within the DS the outer loop assigned; the outer loop picks the DS;
SAC picks the attack parameters.

Training vs test
----------------
- TRAINING (fast, PINN surrogate): the per-network aggregate load/SOC used to
  drive the PINN CMS input is built here (scaled by EVSE count + network type;
  real session profiles from `evcs_data/ACN-Data-Static-main` can be layered in
  via `sample_network_profile` when available).
- TEST (ground truth): the ACN-Sim harness (see acn_node_test_harness.py)
  instantiates the canonical caltech_acn / jpl_acn topologies for the networks
  named in the final deployment plan and replays the same attack.

This module has NO heavy dependencies (numpy only) so it is safe to import from
the training env and unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

# ── Canonical ACN network sizes (EVSE points) ───────────────────────────────
# caltech_acn = the published Caltech adaptive network (California Garage).
# jpl_acn     = the published JPL network (Arroyo Garage).
# Confirmed against evcs_data/ACN-Data-Static-main: JPL=52 distinct EVSEs,
# Caltech main garage ≈ 54 (full 5-garage static site = 151, not used for the
# sim topology — that dataset supplies session/load PROFILES, not the topology).
CALTECH_EVSE_COUNT = 54
JPL_EVSE_COUNT = 52

NETWORK_TYPES = ("caltech", "jpl")
EVSE_COUNTS = {"caltech": CALTECH_EVSE_COUNT, "jpl": JPL_EVSE_COUNT}

N_NODES_PER_DS = 10
DEFAULT_NUM_SYSTEMS = 10  # 10 distribution systems (Phase 1 baseline; ≤6 attacked simultaneously)
DEFAULT_SEED = 42


@dataclass
class NetworkNode:
    """One node in a DS: a full ACN network (Caltech or JPL)."""
    ds_id: int
    node_id: int            # 0-indexed node index within the DS (DQN action)
    network_type: str       # 'caltech' | 'jpl'
    evse_count: int

    @property
    def name(self) -> str:
        return f"DS{self.ds_id}_N{self.node_id:02d}_{self.network_type}"


def build_layout(num_systems: int = DEFAULT_NUM_SYSTEMS,
                 n_nodes: int = N_NODES_PER_DS,
                 seed: int = DEFAULT_SEED) -> Dict[int, List[NetworkNode]]:
    """Build the seeded {ds_id: [NetworkNode, ...]} layout.

    Deterministic for a given (num_systems, n_nodes, seed) so the network
    assignment is reproducible across runs.
    """
    rng = np.random.default_rng(seed)
    layout: Dict[int, List[NetworkNode]] = {}
    for ds in range(1, num_systems + 1):
        nodes: List[NetworkNode] = []
        for k in range(n_nodes):
            ntype = NETWORK_TYPES[int(rng.integers(0, len(NETWORK_TYPES)))]
            nodes.append(NetworkNode(
                ds_id=ds,
                node_id=k,
                network_type=ntype,
                evse_count=EVSE_COUNTS[ntype],
            ))
        layout[ds] = nodes
    return layout


def sample_network_profile(node: NetworkNode,
                           step: int = 0,
                           rng: Optional[np.random.Generator] = None) -> Dict:
    """Per-network aggregate features used to drive the PINN CMS input.

    Phase-2 training surrogate: aggregate load/SOC/demand for the whole network,
    scaled by EVSE count and network type.  This is intentionally lightweight so
    training stays fast.  A real-session variant (drawing from the ACN-Data
    static time series) can replace the body here without changing callers — the
    return schema is what the env's `_build_baseline_station_data` consumes.
    """
    if rng is None:
        rng = np.random.default_rng()

    evse = node.evse_count
    # Utilization differs by network character: Caltech (campus) tends to run
    # busier / longer dwell than JPL (office).  Mild daily profile via `step`.
    base_util = 0.65 if node.network_type == "caltech" else 0.50
    daily = 0.15 * np.sin(step * 0.01)
    utilization = float(np.clip(base_util + daily + rng.normal(0, 0.05), 0.05, 0.98))

    active_evse = max(1, int(round(evse * utilization)))
    mean_soc = float(np.clip(rng.uniform(0.30, 0.80), 0.0, 1.0))
    demand_factor = float(np.clip(utilization + rng.normal(0, 0.05), 0.05, 1.0))

    # Aggregate power scales with the number of ACTIVE EVSEs in the network,
    # so a bigger network (Caltech) is a larger, more impactful target.
    per_evse_kw = 6.6  # typical L2 draw
    aggregate_power_kw = active_evse * per_evse_kw * demand_factor

    return {
        "network_type": node.network_type,
        "evse_count": evse,
        "active_evse": active_evse,
        "utilization": utilization,
        "mean_soc": mean_soc,
        "demand_factor": demand_factor,
        "load_factor": demand_factor,
        "aggregate_power_kw": aggregate_power_kw,
    }


def layout_summary(layout: Dict[int, List[NetworkNode]]) -> Dict:
    """Human-readable summary: counts of each network type per DS + totals."""
    summary = {"per_ds": {}, "totals": {"caltech": 0, "jpl": 0, "evse": 0}}
    for ds, nodes in layout.items():
        c = sum(1 for n in nodes if n.network_type == "caltech")
        j = sum(1 for n in nodes if n.network_type == "jpl")
        evse = sum(n.evse_count for n in nodes)
        summary["per_ds"][ds] = {"caltech": c, "jpl": j, "evse": evse}
        summary["totals"]["caltech"] += c
        summary["totals"]["jpl"] += j
        summary["totals"]["evse"] += evse
    return summary


def describe_layout_for_prompt(layout: Dict[int, List[NetworkNode]],
                               systems: Optional[List[int]] = None) -> str:
    """Compact per-DS node listing for the Gemini deployment prompt.

    For each DS it lists which node indices are Caltech (54 EVSEs, ~65% util —
    larger/busier target) vs JPL (52 EVSEs, ~50% util), so Gemini can choose a
    specific ACN network (node) to attack within the DS it assigns.  Restrict to
    `systems` (e.g. the ones being assigned) to keep the prompt short.
    """
    lines = []
    ds_ids = systems if systems else sorted(layout.keys())
    for ds in ds_ids:
        nodes = layout.get(ds, [])
        cal = [n.node_id for n in nodes if n.network_type == "caltech"]
        jpl = [n.node_id for n in nodes if n.network_type == "jpl"]
        lines.append(f"  System {ds}: caltech nodes {cal} (54 EVSEs, busier) | "
                     f"jpl nodes {jpl} (52 EVSEs)")
    return "\n".join(lines)


if __name__ == "__main__":
    lay = build_layout()
    summ = layout_summary(lay)
    print("=" * 66)
    print(f"ACN NODE LAYOUT  (seed={DEFAULT_SEED}, "
          f"{DEFAULT_NUM_SYSTEMS} DS × {N_NODES_PER_DS} nodes)")
    print("=" * 66)
    for ds, nodes in lay.items():
        row = " ".join(f"{n.network_type[0].upper()}{n.evse_count}" for n in nodes)
        s = summ["per_ds"][ds]
        print(f"  DS{ds}: {row}   "
              f"[caltech={s['caltech']} jpl={s['jpl']} evse={s['evse']}]")
    t = summ["totals"]
    print("-" * 66)
    print(f"  TOTAL: caltech={t['caltech']} jpl={t['jpl']} "
          f"networks={t['caltech'] + t['jpl']} evse={t['evse']}")
    print("=" * 66)
    # Determinism check
    assert layout_summary(build_layout())["totals"] == t, "layout not deterministic!"
    print("✅ layout is deterministic for the fixed seed")
