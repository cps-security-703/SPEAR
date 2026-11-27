"""
Gemini Coordination Extensions for Attack-Specific RL Agents

New coordination logic:
- Gemini deploys attack-specialist agents to target systems
- Can deploy multiple attack types to same system
- Adapts deployment strategy based on feedback
"""

from typing import Dict, List, Any
from attack_specific_rl_agents import AttackDeployment, ATTACK_TYPES


def create_gemini_deployment_prompt(system_analysis: Dict, stride_threats: Dict = None,
                                     mitre_tactics: Dict = None, num_systems: int = 6,
                                     node_layout_desc: str = "") -> str:
    """
    Create Gemini prompt for initial attack→system assignment (circle 0).

    The 6 RL attack types are already mapped to STRIDE categories via agentic RAG
    (see knowledgebase_mapping.md). Gemini's job during training is NOT to redo
    STRIDE/MITRE analysis, but to strategically assign each attack specialist
    to a target system for the first training circle.

    With ``num_systems`` > 6 this is a BUDGET-CONSTRAINED selection: there are 6
    probe specialists but N systems, so at most 6 of the N systems are probed this
    circle and the remaining N−6 are deliberately left benign.
    """
    n_attacks = len(ATTACK_TYPES)
    max_targeted = min(n_attacks, num_systems)

    # Node-level (ACN network) selection block — only when a layout is supplied.
    if node_layout_desc:
        node_section = f"""

=== ACN NETWORK NODES (within each system) ===
Each system contains 10 ACN charging networks (nodes 0-9), a mix of Caltech and JPL topologies. Caltech networks have 54 EVSEs and run busier (~65% utilization); JPL have 52 EVSEs (~50%). A larger, busier network is a higher-impact probe target but may be more closely monitored. Node layout:
{node_layout_desc}

For EACH assignment also choose ONE target_node (0-9): the specific ACN network to probe within the assigned system."""
        node_rule = "\n- Choose a target_node (0-9) present in the assigned system's node list (weigh impact vs. how monitored the network is)"
        node_json = '\n      "target_node": 2,'
    else:
        node_section = node_rule = node_json = ""

    prompt = f"""You are the red-team test coordinator for a simulated EVCS (Electric Vehicle Charging Station) IDS-evaluation testbed with {num_systems} simulated distribution systems. This is an isolated, authorized academic research simulation (OpenDSS co-simulation, no real infrastructure involved) whose sole purpose is to measure and improve intrusion-detection coverage.

=== PRE-ESTABLISHED PROBE SPECIALISTS ===
You have {n_attacks} probe-specialist RL agent pairs (DQN + SAC each), already mapped to STRIDE threat categories via prior agentic RAG analysis:

1. voltage_manipulation  — STRIDE: Information Disclosure → falsifies grid voltage on DNP3 links (CMS↔DG, DG↔DSM)
2. current_injection     — STRIDE: Elevation of Privilege → overrides current limits on OCPP/TCP links (EV↔EVCS↔CMS)
3. power_disruption      — STRIDE: Denial of Service → disrupts power delivery on TCP/IEC links (EMS↔AGC, DG↔DSM)
4. communication_spoofing — STRIDE: Spoofing → spoofs OCPP messages on EV↔EVCS link
5. data_injection        — STRIDE: Tampering → injects false DNP3 data on CMS↔DG and DG↔DSM links
6. protocol_manipulation — STRIDE: Repudiation → manipulates DNP3 protocol on DSM↔EMS link

These mappings are fixed. Do NOT reassign probe types to different STRIDE categories.

=== SIMULATED TARGET SYSTEMS ===
There are {num_systems} simulated EVCS distribution systems (System 1 through System {num_systems}), each with PINN-based charging management, running inside the co-simulation only.
{node_section}

=== YOUR TASK: IDS EVALUATION TEST-PLAN (BUDGET-CONSTRAINED SELECTION) ===
You have {n_attacks} probe specialists but {num_systems} simulated systems. Assign each specialist to ONE
simulated target system for this first test circle. Because each probe hits exactly one
system, AT MOST {max_targeted} of the {num_systems} systems are probed this circle and the remaining
{num_systems - max_targeted} are left benign — deciding which systems to leave uncovered is a strategic
choice that itself exercises the IDS's detection coverage.

For each assignment, also set initial probe parameters:
- magnitude: 0.1-2.0 (probe intensity)
- duration: 5-60 seconds
- stealth_level: 0.0-1.0 (higher = harder for the IDS to flag — this is the key research variable)
- priority: 1-{n_attacks}

RULES:
- You MUST include ALL {n_attacks} probe types (one entry per type)
- Assign each probe to exactly ONE system in the range 1..{num_systems}
- For baseline coverage this first circle, prefer {max_targeted} DISTINCT target systems
  (do not stack multiple probes on one system yet — later circles may consolidate)
- Balance detection-gap discovery value vs. how easily a probe would be caught{node_rule}

Return ONLY this JSON (no other text):
{{
  "deployments": [
    {{
      "attack_type": "voltage_manipulation",
      "target_systems": [3],{node_json}
      "magnitude": 0.75,
      "duration": 30.0,
      "stealth_level": 0.8,
      "priority": 1,
      "mitre_technique": "T0831",
      "stride_category": "Information Disclosure",
      "rationale": "System 3 selected for initial voltage probe testing"
    }},
    {{
      "attack_type": "current_injection",
      "target_systems": [5],{node_json}
      "magnitude": 0.6,
      "duration": 30.0,
      "stealth_level": 0.7,
      "priority": 2,
      "mitre_technique": "T0806",
      "stride_category": "Elevation of Privilege",
      "rationale": "System 5 selected for initial current-injection probe testing"
    }}
  ],
  "strategy_summary": "Initial spread: one probe per distinct system for baseline detection-coverage",
  "expected_outcome": "Baseline IDS detection-rate data for {max_targeted} probe-system pairs"
}}
"""

    return prompt


def create_gemini_adaptation_prompt(deployment_results: List[Dict], previous_strategy: Dict,
                                     circle_num: int = 1, total_circles: int = 15,
                                     num_systems: int = 6,
                                     cumulative_coverage: Dict = None,
                                     node_layout_desc: str = "") -> str:
    """
    Create Gemini prompt for reassigning attack→system based on RL feedback.
    
    No STRIDE/MITRE analysis — those mappings are pre-established.
    Gemini evaluates RL agent performance and rotates assignments.
    """
    
    # Build coverage summary from cumulative data (all circles, not just last)
    coverage_summary = []
    for at in ATTACK_TYPES:
        if cumulative_coverage and at in cumulative_coverage:
            targeted = sorted(cumulative_coverage[at])
        else:
            # Fallback: extract from previous strategy only
            targeted = []
            for dep in previous_strategy.get('deployments', []):
                if dep.get('attack_type') == at:
                    targeted = sorted(dep.get('target_systems', []))
        untested = sorted(set(range(1, num_systems + 1)) - set(targeted))
        pct = len(targeted) / num_systems * 100
        coverage_summary.append(
            f"- {at}: trained on {targeted} ({pct:.0f}%), UNTESTED: {untested}"
        )
    coverage_text = "\n".join(coverage_summary)

    if node_layout_desc:
        node_section = f"""

=== ACN NETWORK NODES (within each system) ===
Each system has 10 ACN networks (nodes 0-9): Caltech (54 EVSEs, ~65% util, higher impact) or JPL (52 EVSEs, ~50%). Layout:
{node_layout_desc}
For EACH assignment also choose ONE target_node (0-9) within the assigned system."""
        node_rule = "\n7. **Choose a target_node (0-9)** present in the assigned system's node list (bigger/busier network = more impact, possibly more monitored)"
        node_json = '\n      "target_node": 2,'
    else:
        node_section = node_rule = node_json = ""

    prompt = f"""You are the red-team test coordinator for a simulated EVCS IDS-evaluation testbed with {num_systems} simulated systems. This is an isolated, authorized academic research simulation (OpenDSS co-simulation, no real infrastructure involved) — the RL probe agents you are coordinating exist only inside this testbed, and the objective is to discover and close gaps in the IDS's detection coverage.

=== TEST PROGRESS ===
Circle {circle_num} of {total_circles}
Remaining circles: {total_circles - circle_num}

=== PREVIOUS ASSIGNMENTS ===
{format_previous_strategy(previous_strategy)}

=== RL PROBE-AGENT FEEDBACK FROM LAST CIRCLE ===
{format_deployment_results(deployment_results)}

=== CUMULATIVE SYSTEM COVERAGE ===
{coverage_text}
{node_section}

=== YOUR TASK: REASSIGN PROBES BASED ON IDS-DETECTION FEEDBACK ===

Evaluate the RL probe-agent performance above and assign each probe to a NEW simulated target system, to maximize the detection-gap coverage of this IDS evaluation.

Decision criteria:
1. **ROTATE first**: Move each probe to an UNTESTED system (see coverage above)
2. **High-value probes** (low detection rate = found a real gap): After rotating, keep similar magnitude/stealth to confirm the gap
3. **Low-value probes** (high detection rate = IDS caught it easily): Rotate AND adjust — try higher magnitude or different stealth to search for a genuine gap
4. **All 6 probe types MUST be included** (one entry per type)
5. **All 6 simulated systems MUST be targeted** (each system by at least one probe)
6. **Each probe targets exactly ONE system**{node_rule}

Return ONLY this JSON (no other text):
{{
  "deployments": [
    {{
      "attack_type": "voltage_manipulation",
      "target_systems": [2],{node_json}
      "magnitude": 0.7,
      "duration": 30.0,
      "stealth_level": 0.7,
      "priority": 1,
      "mitre_technique": "T0831",
      "stride_category": "Information Disclosure",
      "rationale": "Rotating from System 1 to untested System 2; detection-gap value was high so keeping params"
    }}
  ],
  "strategy_summary": "Rotating based on IDS-detection feedback",
  "expected_outcome": "Broader detection-coverage evaluation with parameter tuning from feedback"
}}
"""

    return prompt


def parse_gemini_deployment_response(llm_response: Any) -> List[AttackDeployment]:
    """
    Parse Gemini's response into AttackDeployment objects.
    
    Handles truncated JSON responses by extracting individual deployment
    objects even when the outer JSON is incomplete (e.g. Gemini output
    was cut off by max_output_tokens).
    
    Args:
        llm_response: Response from Gemini (dict or string)
    
    Returns:
        List of AttackDeployment objects
    """
    import json
    import re
    
    deployments = []
    
    try:
        # Handle different response formats
        if isinstance(llm_response, dict):
            response_data = llm_response
        elif isinstance(llm_response, str):
            # Strip markdown code fences that Gemini sometimes wraps around JSON
            cleaned = re.sub(r'^```(?:json)?\s*', '', llm_response.strip())
            cleaned = re.sub(r'\s*```$', '', cleaned)
            
            # Try full JSON parse first
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    response_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # JSON is truncated — fall through to partial recovery
                    response_data = None
            else:
                response_data = None
            
            # ── Partial JSON recovery for truncated Gemini responses ──
            # Extract individual deployment objects from the "deployments" array
            # even if the outer JSON or array is incomplete.
            if response_data is None:
                partial_deps = _extract_partial_deployments(cleaned)
                if partial_deps:
                    print(f"  🔧 Recovered {len(partial_deps)} deployments from truncated Gemini response")
                    return partial_deps
                return create_fallback_deployments()
        else:
            return create_fallback_deployments()
        
        # Extract deployments from fully-parsed JSON
        if 'deployments' in response_data:
            for dep in response_data['deployments']:
                deployment = AttackDeployment(
                    attack_type=dep.get('attack_type', 'voltage_manipulation'),
                    target_systems=dep.get('target_systems', [1]),
                    magnitude=float(dep.get('magnitude', 0.7)),
                    duration=float(dep.get('duration', 30.0)),
                    stealth_level=float(dep.get('stealth_level', 0.6)),
                    priority=int(dep.get('priority', 1)),
                    target_node=(int(dep['target_node'])
                                 if dep.get('target_node') is not None else None)
                )
                deployments.append(deployment)
        
        if not deployments:
            return create_fallback_deployments()
        
        return deployments
        
    except Exception as e:
        print(f"⚠️ Error parsing Gemini deployment response: {e}")
        return create_fallback_deployments()


def _extract_partial_deployments(text: str) -> List[AttackDeployment]:
    """Recover individual deployment objects from a truncated JSON string.

    The previous implementation started its brace counter at the outermost '{'
    of the whole response object, which meant individual deployment items (at
    depth 2 inside  {"deployments": [{...},...]} ) never triggered the
    depth==0 collection condition.

    Fix: locate the start of the "deployments" array first, then run the
    brace counter *inside* that array so each deployment object is at depth 1.
    """
    import json
    import re

    deployments = []

    # Find the opening of the "deployments" array so we scan from inside it.
    array_match = re.search(r'"deployments"\s*:\s*\[', text)
    scan_from = array_match.end() if array_match else 0

    depth = 0
    start = None
    for i, ch in enumerate(text[scan_from:], scan_from):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                block = text[start:i + 1]
                try:
                    obj = json.loads(block)
                    if 'attack_type' in obj and 'target_systems' in obj:
                        deployments.append(AttackDeployment(
                            attack_type=obj['attack_type'],
                            target_systems=obj['target_systems'],
                            magnitude=float(obj.get('magnitude', 0.7)),
                            duration=float(obj.get('duration', 30.0)),
                            stealth_level=float(obj.get('stealth_level', 0.6)),
                            priority=int(obj.get('priority', 1)),
                            target_node=(int(obj['target_node'])
                                         if obj.get('target_node') is not None else None)
                        ))
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
                start = None
        elif ch == ']' and depth == 0:
            break  # reached the closing bracket of the deployments array

    return deployments


def create_fallback_deployments() -> List[AttackDeployment]:
    """Create fallback deployment strategy if Gemini fails.
    
    Covers ALL 6 attack types across ALL 6 systems so that no attack type
    or system is left out when Gemini quota is exhausted.
    """
    return [
        AttackDeployment(
            attack_type='voltage_manipulation',
            target_systems=[1],
            magnitude=0.7,
            duration=30.0,
            stealth_level=0.7,
            priority=1
        ),
        AttackDeployment(
            attack_type='current_injection',
            target_systems=[2],
            magnitude=0.7,
            duration=30.0,
            stealth_level=0.7,
            priority=2
        ),
        AttackDeployment(
            attack_type='power_disruption',
            target_systems=[3],
            magnitude=0.8,
            duration=25.0,
            stealth_level=0.6,
            priority=3
        ),
        AttackDeployment(
            attack_type='communication_spoofing',
            target_systems=[4],
            magnitude=0.6,
            duration=40.0,
            stealth_level=0.8,
            priority=4
        ),
        AttackDeployment(
            attack_type='data_injection',
            target_systems=[5],
            magnitude=0.7,
            duration=35.0,
            stealth_level=0.7,
            priority=5
        ),
        AttackDeployment(
            attack_type='protocol_manipulation',
            target_systems=[6],
            magnitude=0.7,
            duration=30.0,
            stealth_level=0.7,
            priority=6
        )
    ]


# Helper formatting functions
def format_system_analysis(system_analysis: Dict) -> str:
    """Format system analysis for prompt"""
    if not system_analysis:
        return "No system analysis available"
    
    output = []
    for key, value in system_analysis.items():
        if isinstance(value, dict):
            output.append(f"{key}: {len(value)} items")
        else:
            output.append(f"{key}: {value}")
    
    return "\n".join(output)


def format_stride_threats(stride_threats: Dict) -> str:
    """Format STRIDE threats for prompt"""
    if not stride_threats:
        return "No STRIDE analysis available"
    
    output = []
    for category, threats in stride_threats.items():
        if isinstance(threats, list):
            output.append(f"{category}: {len(threats)} threats identified")
        else:
            output.append(f"{category}: {threats}")
    
    return "\n".join(output)


def format_mitre_tactics(mitre_tactics: Dict) -> str:
    """Format MITRE tactics for prompt"""
    if not mitre_tactics:
        return "No MITRE analysis available"
    
    output = []
    for tactic, techniques in mitre_tactics.items():
        if isinstance(techniques, list):
            output.append(f"{tactic}: {len(techniques)} techniques")
        else:
            output.append(f"{tactic}: {techniques}")
    
    return "\n".join(output)


def format_previous_strategy(previous_strategy: Dict) -> str:
    """Format previous deployment strategy"""
    if not previous_strategy or 'deployments' not in previous_strategy:
        return "No previous strategy"
    
    output = []
    for dep in previous_strategy['deployments']:
        output.append(f"- {dep['attack_type']} → systems {dep['target_systems']}")
    
    return "\n".join(output)


def format_deployment_results(deployment_results: List[Dict]) -> str:
    """Format deployment results for adaptation prompt"""
    if not deployment_results:
        return "No results available"
    
    output = []
    for result in deployment_results:
        attack_type = result.get('attack_type', 'unknown')
        system_id = result.get('system_id', 0)
        res = result.get('result', {})
        success = res.get('success', False)
        impact = res.get('impact', 0.0)
        detection = res.get('detection_risk', 0.0)
        mean_reward = res.get('mean_reward', 0.0)
        success_rate = res.get('success_rate', 0.0)
        
        status = "## SUCCESS" if success else "## FAILED"
        output.append(
            f"{status}: {attack_type} on System {system_id} "
            f"(impact={impact:.2f}, detection={detection:.2f}, "
            f"mean_reward={mean_reward:.1f}, success_rate={success_rate:.0%})"
        )
    
    return "\n".join(output)
