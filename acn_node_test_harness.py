

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from acn_network_layout import build_layout, NetworkNode, EVSE_COUNTS


@dataclass
class NodeAttackTarget:
    ttack_type: str
    ds_id: int
    node_id: int
    network_type: str
    evse_count: int
    magnitude: float = 0.7
    duration: float = 30.0
    stealth_level: float = 0.7


@dataclass
class NodeAttackResult:
    target: NodeAttackTarget
    executed: bool = False
    physical_impact: float = 0.0
    fdi_detected: bool = False
    fdi_score: float = 0.0
    note: str = ""


def resolve_targets(deployment_plan: Dict[str, Any],
                    num_systems: int = 6,
                    n_nodes: int = 10,
                    node_seed: int = 42) -> List[NodeAttackTarget]:

    layout = build_layout(num_systems=num_systems, n_nodes=n_nodes, seed=node_seed)
    targets: List[NodeAttackTarget] = []

    for dep in deployment_plan.get("deployments", []):
        ds = int(dep.get("target_system", 1))
        node_id = int(dep.get("target_node", 0))
        nodes = layout.get(ds, [])
        if not nodes:
            continue
        node = nodes[node_id % len(nodes)]
        targets.append(NodeAttackTarget(
            attack_type=dep.get("attack_type", "voltage_manipulation"),
            ds_id=ds,
            node_id=node.node_id,
            network_type=node.network_type,
            evse_count=node.evse_count,
            magnitude=float(dep.get("magnitude", 0.7)),
            duration=float(dep.get("duration", 30.0)),
            stealth_level=float(dep.get("stealth_level", 0.7)),
        ))
    return targets


def attach_dqn_nodes(deployment_plan: Dict[str, Any],
                     attack_coordinator,
                     num_systems: int = 6) -> Dict[str, Any]:

    if not getattr(attack_coordinator, "node_level", False):
        deployment_plan.setdefault("_notes", []).append(
            "attach_dqn_nodes: coordinator is system-level; target_node not set")
        return deployment_plan

    for dep in deployment_plan.get("deployments", []):
        at = dep.get("attack_type")
        ds = int(dep.get("target_system", 1))
        dqn_env = attack_coordinator.environments.get(f"dqn_{at}")
        dqn_agent = attack_coordinator.dqn_agents.get(at)
        if dqn_env is None or dqn_agent is None:
            continue
        try:

            dqn_env.continuous_env.forced_target_system = [ds]
            obs, _ = dqn_env.reset()
            action, _ = dqn_agent.predict(obs, deterministic=True)
            dep["target_node"] = int(action) % attack_coordinator.n_nodes
        except Exception as e:
            dep.setdefault("note", "")
            dep["note"] += f" [dqn-node query failed: {e}]"
    return deployment_plan


def _acn_available() -> bool:
    try:
        import acnportal  # noqa: F401
        return True
    except Exception:
        return False


def run_acn_validation(deployment_plan: Dict[str, Any],
                       num_systems: int = 6,
                       n_nodes: int = 10,
                       node_seed: int = 42,
                       duration_s: float = 3600.0) -> List[NodeAttackResult]:

    targets = resolve_targets(deployment_plan, num_systems, n_nodes, node_seed)
    results: List[NodeAttackResult] = []

    if not _acn_available():
        for t in targets:
            results.append(NodeAttackResult(
                target=t, executed=False,
                note="acnportal/acnsim not available — plan resolved but not executed"))
        return results


    from acn_sim_interface import ACNSimZone
    try:
        from acn.fdi_anomaly_detection import score_network as _fdi_score  # noqa
    except Exception:
        _fdi_score = None

    for t in targets:
        res = NodeAttackResult(target=t)
        try:

            n_evses = EVSE_COUNTS[t.network_type]
            zone = ACNSimZone(ds_id=t.ds_id, n_evses=n_evses)


            res.note = ("zone constructed; wire CMS init + attack injection + "
                        "FDI scoring at INTEGRATION POINTS A/B/C to execute")
            res.executed = False
        except Exception as e:
            res.note = f"ACN execution error: {e}"
        results.append(res)

    return results


def summarize(results: List[NodeAttackResult]) -> Dict[str, Any]:

    executed = [r for r in results if r.executed]
    by_type = {"caltech": [], "jpl": []}
    for r in results:
        by_type[r.target.network_type].append(r)
    return {
        "n_targets": len(results),
        "n_executed": len(executed),
        "detected": sum(1 for r in executed if r.fdi_detected),
        "mean_impact": (sum(r.physical_impact for r in executed) / len(executed)
                        if executed else 0.0),
        "caltech_targets": len(by_type["caltech"]),
        "jpl_targets": len(by_type["jpl"]),
    }


if __name__ == "__main__":

    demo_plan = {"deployments": [
        {"attack_type": "voltage_manipulation", "target_system": 1, "target_node": 3},
        {"attack_type": "current_injection", "target_system": 4, "target_node": 7},
        {"attack_type": "power_disruption", "target_system": 6},
    ]}
    for t in resolve_targets(demo_plan):
        print(f"  {t.attack_type:22s} DS{t.ds_id} node{t.node_id:02d} "
              f" {t.network_type} ({t.evse_count} EVSEs) mag={t.magnitude}")
    print("resolve_targets OK")
