"""
Gemini Coordination Extensions for Attack-Specific RL Agents

"""

from typing import Dict, List, Any
from attack_specific_rl_agents import AttackDeployment, ATTACK_TYPES


def create_gemini_deployment_prompt(system_analysis: Dict, stride_threats: Dict = None, mitre_tactics: Dict = None) -> str:
    """
    Create Gemini prompt for initial attack→system assignment (circle 0).
    
    The 6 RL attack types are already mapped to STRIDE categories via agentic RAG
    (see knowledgebase_mapping.md). Gemini's job during training is NOT to redo
    STRIDE/MITRE analysis, but to strategically assign each attack specialist
    to a target system for the first training circle.
    """
    
    prompt = f"""You are coordinating a vulnerability discovery simulation for an EVCS (Electric Vehicle Charging Station) network with 6 distribution systems.

=== PRE-ESTABLISHED ATTACK SPECIALISTS ===
You have 6 attack-specialist RL agent pairs (DQN + SAC each), already mapped to STRIDE threat categories via prior agentic RAG analysis:

1. voltage_manipulation  — STRIDE: Information Disclosure → falsifies grid voltage on DNP3 links (CMS↔DG, DG↔DSM)
2. current_injection     — STRIDE: Elevation of Privilege → overrides current limits on OCPP/TCP links (EV↔EVCS↔CMS)
3. power_disruption      — STRIDE: Denial of Service → disrupts power delivery on TCP/IEC links (EMS↔AGC, DG↔DSM)
4. communication_spoofing — STRIDE: Spoofing → spoofs OCPP messages on EV↔EVCS link
5. data_injection        — STRIDE: Tampering → injects false DNP3 data on CMS↔DG and DG↔DSM links
6. protocol_manipulation — STRIDE: Repudiation → manipulates DNP3 protocol on DSM↔EMS link

These mappings are fixed. Do NOT reassign attack types to different STRIDE categories.

=== TARGET SYSTEMS ===
There are 6 EVCS distribution systems (System 1 through System 6), each with PINN-based charging management.

=== YOUR TASK ===
Assign each attack specialist to ONE target system for this first training circle.
Your goal: spread attacks across all 6 systems so each system is tested by at least one attack type.

For each assignment, also set initial attack parameters:
- magnitude: 0.1-2.0 (attack intensity)
- duration: 5-60 seconds
- stealth_level: 0.0-1.0 (higher = more stealthy)
- priority: 1-6

RULES:
- You MUST include ALL 6 attack types (one entry per type)
- You MUST target ALL 6 systems (each system targeted by at least one attack)
- Each attack targets exactly ONE system in this circle
- Balance impact vs detection risk

Return ONLY this JSON (no other text):
{{
  "deployments": [
    {{
      "attack_type": "voltage_manipulation",
      "target_systems": [3],
      "magnitude": 0.75,
      "duration": 30.0,
      "stealth_level": 0.8,
      "priority": 1,
      "mitre_technique": "T0831",
      "stride_category": "Information Disclosure",
      "rationale": "System 3 selected for initial voltage testing"
    }},
    {{
      "attack_type": "current_injection",
      "target_systems": [5],
      "magnitude": 0.6,
      "duration": 30.0,
      "stealth_level": 0.7,
      "priority": 2,
      "mitre_technique": "T0806",
      "stride_category": "Elevation of Privilege",
      "rationale": "System 5 selected for initial current injection testing"
    }}
  ],
  "strategy_summary": "Initial spread: one attack per system for baseline coverage",
  "expected_outcome": "Baseline performance data for all 6 attack-system pairs"
}}
"""
    
    return prompt


def create_gemini_adaptation_prompt(deployment_results: List[Dict], previous_strategy: Dict,
                                     circle_num: int = 1, total_circles: int = 15,
                                     num_systems: int = 6,
                                     cumulative_coverage: Dict = None) -> str:
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
    
    prompt = f"""You are coordinating RL attack agent training on an EVCS network with {num_systems} systems.

=== TRAINING PROGRESS ===
Circle {circle_num} of {total_circles}
Remaining circles: {total_circles - circle_num}

=== PREVIOUS ASSIGNMENTS ===
{format_previous_strategy(previous_strategy)}

=== RL AGENT FEEDBACK FROM LAST CIRCLE ===
{format_deployment_results(deployment_results)}

=== CUMULATIVE SYSTEM COVERAGE ===
{coverage_text}

=== YOUR TASK: REASSIGN ATTACKS BASED ON RL FEEDBACK ===

Evaluate the RL agent performance above and assign each attack to a NEW target system.

Decision criteria:
1. **ROTATE first**: Move each attack to an UNTESTED system (see coverage above)
2. **High-reward attacks**: After rotating, keep similar magnitude/stealth
3. **Low-reward attacks**: Rotate AND adjust — try higher magnitude or different stealth
4. **All 6 attack types MUST be included** (one entry per type)
5. **All 6 systems MUST be targeted** (each system by at least one attack)
6. **Each attack targets exactly ONE system**

Return ONLY this JSON (no other text):
{{
  "deployments": [
    {{
      "attack_type": "voltage_manipulation",
      "target_systems": [2],
      "magnitude": 0.7,
      "duration": 30.0,
      "stealth_level": 0.7,
      "priority": 1,
      "mitre_technique": "T0831",
      "stride_category": "Information Disclosure",
      "rationale": "Rotating from System 1 to untested System 2; reward was high so keeping params"
    }}
  ],
  "strategy_summary": "Rotating based on RL feedback",
  "expected_outcome": "Broader coverage with parameter tuning from feedback"
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
                    priority=int(dep.get('priority', 1))
                )
                deployments.append(deployment)
        
        if not deployments:
            return create_fallback_deployments()
        
        return deployments
        
    except Exception as e:
        print(f"# Error parsing Gemini deployment response: {e}")
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
                            priority=int(obj.get('priority', 1))
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
