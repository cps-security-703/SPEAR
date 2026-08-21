

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np


CALTECH_EVSE_COUNT = 54
JPL_EVSE_COUNT = 52

NETWORK_TYPES = ("caltech", "jpl")
EVSE_COUNTS = {"caltech": CALTECH_EVSE_COUNT, "jpl": JPL_EVSE_COUNT}

N_NODES_PER_DS = 10
DEFAULT_NUM_SYSTEMS = 10
DEFAULT_SEED = 42


@dataclass
class NetworkNode:

    ds_id: int
    node_id: int
    network_type: str
    evse_count: int

    @property
    def name(self) -> str:
        return f"DS{self.ds_id}_N{self.node_id:02d}_{self.network_type}"


def build_layout(num_systems: int = DEFAULT_NUM_SYSTEMS,
                 n_nodes: int = N_NODES_PER_DS,
                 seed: int = DEFAULT_SEED) -> Dict[int, List[NetworkNode]]:

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

    if rng is None:
        rng = np.random.default_rng()

    evse = node.evse_count


    base_util = 0.65 if node.network_type == "caltech" else 0.50
    daily = 0.15 * np.sin(step * 0.01)
    utilization = float(np.clip(base_util + daily + rng.normal(0, 0.05), 0.05, 0.98))

    active_evse = max(1, int(round(evse * utilization)))
    mean_soc = float(np.clip(rng.uniform(0.30, 0.80), 0.0, 1.0))
    demand_factor = float(np.clip(utilization + rng.normal(0, 0.05), 0.05, 1.0))


    per_evse_kw = 6.6
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

    assert layout_summary(build_layout())["totals"] == t, "layout not deterministic!"
    print(" layout is deterministic for the fixed seed")
