#!/usr/bin/env python3
"""
LLM Threat Analyzer for EVCS Attack Analytics
Supports both Google Gemini (native SDK) and OpenRouter (multi-model API).

Provider selection is controlled by the USE_GEMINI flag below.
All API keys are read from the .env file (or environment variables).

.env file format:
    GEMINI_API_KEY=<your-google-api-key>
    OPENROUTER_API_KEY=<your-openrouter-api-key>
"""

from llm_metrics_logger import llm_call_metrics, LLMMetricsLogger

# Load .env file (python-dotenv is optional; fallback to os.environ)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env variables must already be exported to environment

import os
import json
import re
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# ============================================================================
# LLM PROVIDER CONFIGURATION (single source of truth)
# ----------------------------------------------------------------------------
# Flip USE_GEMINI to switch between Google Gemini and OpenRouter.
# MODEL_NAME is resolved to the correct model id for the selected provider.
# ============================================================================
USE_GEMINI: bool = False           # False -> use OpenRouter
GEMINI_MODEL_NAME: str = "models/gemini-2.5-flash"
# Only JSON-response-capable OpenRouter models allowed (see whitelist below)
OPENROUTER_MODEL_NAME: str = "google/gemini-3.1-flash-lite-preview"

# OpenRouter models that reliably support response_format={"type":"json_object"}
# NOTE: Free model availability rotates — check https://openrouter.ai/models
JSON_CAPABLE_OPENROUTER_MODELS = {
    # --- Paid (strong JSON mode) ---
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "google/gemini-2.5-flash",         # ~$2.8/M tok
    "google/gemini-2.5-pro",
    "google/gemini-1.5-flash",
    "google/gemini-1.5-pro",
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-v3.2",          # ~$0.6/M tok
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-3.5-haiku",
    "deepseek/deepseek-v4-flash",
    "google/gemini-2.5-flash",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.1-flash",
    # --- Free (availability may rotate; check OpenRouter) ---
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-chat-v3.1:free",
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "minimax/minimax-m2.5:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.3-70b-instruct",
    "openrouter/elephant-alpha",
    "openai/gpt-oss-120b:free",
}

# Conditional import of Gemini SDK only when needed
if USE_GEMINI:
    import google.generativeai as genai
else:
    genai = None  # not used in OpenRouter mode


# ============================================================================
# OpenRouter client: Gemini-compatible duck-typed wrapper
# ----------------------------------------------------------------------------
# Exposes the same .generate_content(prompt) -> response.text interface as
# genai.GenerativeModel so that all downstream call sites work unchanged.
# ============================================================================
class _OpenRouterResponse:
    """Mimics google.generativeai response object (has .text attribute)."""
    __slots__ = ("text", "raw")

    def __init__(self, text: str, raw=None):
        self.text = text
        self.raw = raw


class OpenRouterClient:
    """Gemini-compatible OpenRouter client (JSON-capable models only)."""

    def __init__(self, api_key: str, model: str,
                 max_output_tokens: int = 4096, temperature: float = 0.7):
        if model not in JSON_CAPABLE_OPENROUTER_MODELS:
            raise ValueError(
                f"OpenRouter model '{model}' is not in the JSON-capable whitelist. "
                f"Allowed: {sorted(JSON_CAPABLE_OPENROUTER_MODELS)}"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenRouter requires the 'openai' package. Install with: pip install openai"
            )

        self.model_id = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def generate_content(self, prompt: str) -> _OpenRouterResponse:
        """Agent-compatible generate_content; always requests JSON output."""
        try:
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system",
                     "content": "You are a cybersecurity threat analyst for EVCS systems. "
                                "Always respond with valid JSON when the user requests structured data."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                extra_headers={
                    "HTTP-Referer": "https://research.evcs-llm-rl",
                    "X-Title": "LLM-RL-EVCS-Comparison-Study",
                },
            )
        except Exception:
            # If response_format is unsupported (free/open-source models),
            # retry without it so plain text is still captured.
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system",
                     "content": "You are a cybersecurity threat analyst for EVCS systems. "
                                "Respond with valid JSON only when structured data is requested."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                extra_headers={
                    "HTTP-Referer": "https://research.evcs-llm-rl",
                    "X-Title": "LLM-RL-EVCS-Comparison-Study",
                },
            )

        content = resp.choices[0].message.content or ""
        # Strip markdown code fences if model wrapped JSON (common with Llama/Mistral)
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if fenced:
            content = fenced.group(1)
        return _OpenRouterResponse(text=content, raw=resp)


# ============================================================================
# Helper: load API keys from environment (populated by .env via load_dotenv)
# ============================================================================
def _load_key_from_env(env_var: str, fallback_file: str = None) -> str:
    """
    Try os.environ first; if absent and fallback_file given, read that file.
    Raises RuntimeError if neither source yields a non-empty key.
    """
    key = os.environ.get(env_var, "").strip()
    if key:
        return key
    if fallback_file:
        try:
            with open(fallback_file, "r") as f:
                key = f.read().strip()
            if key:
                print(f"##  Loaded {env_var} from fallback file '{fallback_file}'")
                return key
        except FileNotFoundError:
            pass
    raise RuntimeError(
        f"API key '{env_var}' not found in environment or .env file. "
        f"Please set {env_var} in your .env file."
    )


# Try to import existing classes for compatibility
try:
    from llm_guided_evcs_attack_analytics import EVCSVulnerability, STRIDEMITREThreatMapper
except ImportError:
    print("Warning: Could not import EVCSVulnerability. Creating minimal version.")

    @dataclass
    class EVCSVulnerability:
        """EVCS vulnerability data class"""
        vuln_id: str
        component: str
        vulnerability_type: str
        severity: float
        exploitability: float
        impact: float
        cvss_score: float
        mitigation: str
        detection_methods: List[str]


class GeminiLLMThreatAnalyzer:
    """
    Multi-provider LLM Threat Analyzer for EVCS systems.

    Provider is selected by the module-level USE_GEMINI flag:
      • USE_GEMINI = True  → Google Gemini (native SDK, key from GEMINI_API_KEY in .env)
      • USE_GEMINI = False → OpenRouter   (openai-compat, key from OPENROUTER_API_KEY in .env)

    All call sites remain unchanged — both backends expose the same
    .model.generate_content(prompt) → response.text interface.
    """

    def __init__(self, api_key: str = None, model_name: str = None, max_history: int = 20):
        """
        Initialize LLM Threat Analyzer.

        Args:
            api_key:    Override API key (usually left as None; read from .env).
            model_name: Override model name (usually left as None; uses module-level constants).
            max_history: Maximum conversation turns to remember.
        """
        # ── Provider-aware model name resolution ─────────────────────────────
        # Detect Gemini-format names ("models/gemini-*") vs OpenRouter-format
        # ("provider/model-name").  If the caller passes a Gemini name but
        # USE_GEMINI=False (or vice versa), override silently with the correct
        # module-level constant.  This makes all legacy call sites like
        #   GeminiLLMThreatAnalyzer(model_name='models/gemini-2.5-flash')
        # work transparently after a provider switch.
        _is_gemini_fmt = (model_name is not None and
                          (model_name.startswith("models/gemini") or
                           model_name.startswith("gemini")))
        _is_or_fmt     = (model_name is not None and "/" in model_name
                          and not _is_gemini_fmt)

        if model_name is None:
            # No override — use module-level default for the active provider
            model_name = GEMINI_MODEL_NAME if USE_GEMINI else OPENROUTER_MODEL_NAME
        elif USE_GEMINI and _is_or_fmt:
            # OpenRouter name passed but Gemini selected → use Gemini default
            print(f"##  Ignoring OpenRouter model '{model_name}' (USE_GEMINI=True); "
                  f"using '{GEMINI_MODEL_NAME}'")
            model_name = GEMINI_MODEL_NAME
        elif not USE_GEMINI and _is_gemini_fmt:
            # Gemini name passed but OpenRouter selected → use OpenRouter default
            print(f"##  Ignoring Gemini model '{model_name}' (USE_GEMINI=False); "
                  f"using '{OPENROUTER_MODEL_NAME}'")
            model_name = OPENROUTER_MODEL_NAME
        # else: caller passed a name that matches the active provider — keep it

        # Expose attributes expected by llm_call_metrics decorator
        self.model_name = model_name
        self.provider   = "gemini" if USE_GEMINI else "openrouter"

        self.is_available = False
        self.model = None
        self.max_history = max_history
        self.min_request_interval = 15.0  # seconds between LLM calls
        self._last_request_time = 0.0

        # Fields required by llm_call_metrics decorator
        self._last_raw_response  = None
        self._last_prompt_length = 0
        self._last_call_id       = None

        # Conversation memory
        self.conversation_history = []
        self.analysis_context = {
            'previous_vulnerabilities': [],
            'previous_strategies': [],
            'system_learning': {},
            'threat_evolution': []
        }

        # ── Initialise the chosen backend ────────────────────────────────────
        if USE_GEMINI:
            self._init_gemini(api_key, model_name)
        else:
            # Legacy call sites (e.g. enhanced_integrated_evcs_system.py) read
            # gemini_key.txt and pass the Google API key (starts with "AIza").
            # That key is invalid for OpenRouter and causes a 401.
            # When routing to OpenRouter, always discard Gemini-format keys so
            # _init_openrouter reads OPENROUTER_API_KEY from .env instead.
            _is_google_key = bool(api_key and str(api_key).startswith("AIza"))
            if _is_google_key:
                print("##  Discarding Gemini API key for OpenRouter; "
                      "reading OPENROUTER_API_KEY from .env")
                api_key = None
            self._init_openrouter(api_key, model_name)

    # ── Backend initializers ─────────────────────────────────────────────────

    def _init_gemini(self, api_key: Optional[str], model_name: str) -> None:
        """Configure and test the Google Gemini backend."""
        if api_key is None:
            try:
                api_key = _load_key_from_env("GEMINI_API_KEY", fallback_file="gemini_key.txt")
            except RuntimeError as exc:
                print(f"## {exc}")
                return
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4096,
                    temperature=0.7,
                ),
            )
            self._test_connection()
        except Exception as e:
            print(f"Failed to initialize Agent client: {e}")
            self.is_available = False

    def _init_openrouter(self, api_key: Optional[str], model_name: str) -> None:
        """Configure and test the OpenRouter backend."""
        if api_key is None:
            try:
                api_key = _load_key_from_env("OPENROUTER_API_KEY")
            except RuntimeError as exc:
                print(f"## {exc}")
                return
        try:
            self.model = OpenRouterClient(
                api_key=api_key,
                model=model_name,
                max_output_tokens=4096,
                temperature=0.7,
            )
            self._test_connection()
        except Exception as e:
            print(f"Failed to initialize OpenRouter client: {e}")
            self.is_available = False

    def _throttle_requests(self):
        """Ensure a minimum delay between consecutive LLM requests."""
        if self.min_request_interval <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _timed_generate(self, prompt: str):
        """Call model.generate_content() and store artefacts for @llm_call_metrics.

        Sets self._last_raw_response and self._last_prompt_length so the
        decorator can extract token counts and build a full LLMCallRecord.
        Also applies request throttling.
        """
        self._throttle_requests()
        self._last_prompt_length = len(prompt)
        response = self.model.generate_content(prompt)
        # Store the full response object (not response.raw) so the
        # llm_call_metrics decorator can find .raw on _OpenRouterResponse
        # and extract token counts via usage.prompt_tokens / completion_tokens.
        self._last_raw_response = response
        return response

    def _load_api_key(self) -> str:
        """Legacy helper — now delegates to _load_key_from_env."""
        return _load_key_from_env("GEMINI_API_KEY", fallback_file="gemini_key.txt")

    def _test_connection(self):
        """Test connection to the configured LLM backend."""
        try:
            response = self.model.generate_content("Reply with the word OK only.")
            if response.text:
                self.is_available = True
                backend = "Gemini" if USE_GEMINI else "OpenRouter"
                print(f"## {backend} connection successful with model: {self.model_name}")
            else:
                raise Exception("Empty response from LLM backend")
        except Exception as e:
            print(f"##LLM connection failed: {e}")
            self.is_available = False
    
    def _add_to_conversation_history(self, user_input: str, assistant_response: str, analysis_type: str = None):
        """Add interaction to conversation history with context"""
        conversation_entry = {
            'timestamp': time.time(),
            'user_input': user_input,
            'assistant_response': assistant_response,
            'analysis_type': analysis_type
        }
        
        self.conversation_history.append(conversation_entry)
        
        # Maintain max history limit
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def _update_analysis_context(self, analysis_type: str, result: Dict):
        """Update the analysis context with new findings"""
        timestamp = time.time()
        
        if analysis_type == 'vulnerability_analysis':
            self.analysis_context['previous_vulnerabilities'].append({
                'timestamp': timestamp,
                'vulnerabilities': result.get('vulnerabilities', []),
                'attack_vectors': result.get('attack_vectors', [])
            })
            
        elif analysis_type == 'attack_strategy':
            self.analysis_context['previous_strategies'].append({
                'timestamp': timestamp,
                'strategy': result.get('strategy_name', 'Unknown'),
                'techniques': result.get('mitre_techniques', []),
                'success_probability': result.get('success_probability', 0.0)
            })
        
        # Learn from patterns
        self._update_system_learning(analysis_type, result)
    
    def _update_system_learning(self, analysis_type: str, result: Dict):
        """Extract patterns and learning from analysis results"""
        if analysis_type not in self.analysis_context['system_learning']:
            self.analysis_context['system_learning'][analysis_type] = {
                'common_patterns': [],
                'effectiveness_metrics': {},
                'trend_analysis': []
            }
        
        learning = self.analysis_context['system_learning'][analysis_type]
        
        # Track common vulnerability types
        if analysis_type == 'vulnerability_analysis':
            vulns = result.get('vulnerabilities', [])
            for vuln in vulns:
                vuln_type = vuln.get('type', 'unknown')
                if vuln_type not in learning['common_patterns']:
                    learning['common_patterns'].append(vuln_type)
        
        # Track strategy effectiveness
        elif analysis_type == 'attack_strategy':
            success_prob = result.get('success_probability', 0.0)
            strategy_name = result.get('strategy_name', 'unknown')
            learning['effectiveness_metrics'][strategy_name] = success_prob
    
    def _get_conversation_context(self, max_recent: int = 3) -> str:
        """Get recent conversation context for prompts"""
        if not self.conversation_history:
            return ""
        
        recent_history = self.conversation_history[-max_recent:]
        context_lines = ["PREVIOUS INTERACTION CONTEXT:"]
        
        for i, entry in enumerate(recent_history, 1):
            context_lines.append(f"\n--- Interaction {i} ({entry.get('analysis_type', 'general')}) ---")
            context_lines.append(f"Request: {entry['user_input'][:200]}...")
            context_lines.append(f"Response: {entry['assistant_response'][:200]}...")
        
        context_lines.append("\nBased on the above context, provide continuity in your analysis.")
        return "\n".join(context_lines)
    
    def _get_learning_context(self, analysis_type: str) -> str:
        """Get system learning context for improved analysis"""
        if analysis_type not in self.analysis_context['system_learning']:
            return ""
        
        learning = self.analysis_context['system_learning'][analysis_type]
        context_lines = [f"\nSYSTEM LEARNING CONTEXT FOR {analysis_type.upper()}:"]
        
        if learning['common_patterns']:
            context_lines.append(f"Common patterns observed: {', '.join(learning['common_patterns'])}")
        
        if learning['effectiveness_metrics']:
            most_effective = max(learning['effectiveness_metrics'].items(), key=lambda x: x[1])
            context_lines.append(f"Most effective strategy: {most_effective[0]} (success: {most_effective[1]:.2f})")
        
        # Add vulnerability evolution tracking
        if analysis_type == 'vulnerability_analysis' and self.analysis_context['previous_vulnerabilities']:
            recent_vulns = self.analysis_context['previous_vulnerabilities'][-3:]
            vuln_trend = []
            for vuln_set in recent_vulns:
                vuln_trend.extend([v.get('type', 'unknown') for v in vuln_set.get('vulnerabilities', [])])
            
            if vuln_trend:
                from collections import Counter
                common_recent = Counter(vuln_trend).most_common(3)
                context_lines.append(f"Recent vulnerability trends: {[f'{v}({c})' for v, c in common_recent]}")
        
        return "\n".join(context_lines)
    
    def get_conversation_summary(self) -> Dict:
        """Get a summary of the conversation history and learning"""
        return {
            'total_interactions': len(self.conversation_history),
            'recent_interactions': self.conversation_history[-3:] if self.conversation_history else [],
            'learning_summary': {
                'vulnerability_patterns': self.analysis_context['system_learning'].get('vulnerability_analysis', {}).get('common_patterns', []),
                'strategy_effectiveness': self.analysis_context['system_learning'].get('attack_strategy', {}).get('effectiveness_metrics', {}),
                'total_vulnerabilities_analyzed': len(self.analysis_context['previous_vulnerabilities']),
                'total_strategies_generated': len(self.analysis_context['previous_strategies'])
            }
        }
    
    def clear_conversation_history(self):
        """Clear conversation history (useful for starting fresh)"""
        self.conversation_history = []
        print("🧹 Conversation history cleared")
    
    def reset_learning_context(self):
        """Reset the learning context while keeping conversation history"""
        self.analysis_context = {
            'previous_vulnerabilities': [],
            'previous_strategies': [],
            'system_learning': {},
            'threat_evolution': []
        }
        print("## Learning context reset")
    
    @llm_call_metrics()
    def analyze_threats(self, system_data: Dict) -> Dict:
        """Analyze threats using Agent Pro LLM"""
        if not self.is_available:
            return self._fallback_analysis(system_data, {})
        
        try:
            # If a deployment_prompt is provided (from CentralRLCoordinator),
            # use it directly — it already contains the full context and asks
            # for a structured JSON deployment response.
            if 'deployment_prompt' in system_data and system_data['deployment_prompt']:
                analysis_prompt = system_data['deployment_prompt']
            # If an adaptation_prompt is provided (from LangGraph _llm_adaptation_node),
            # use it directly — it already contains RL feedback and asks for
            # strategy adaptation, not generic vulnerability analysis.
            elif 'adaptation_prompt' in system_data and system_data['adaptation_prompt']:
                analysis_prompt = system_data['adaptation_prompt']
            else:
                # Prepare system data for analysis
                system_summary = {
                    'evcs_systems': system_data.get('evcs_systems', 6),
                    'pinn_models': system_data.get('pinn_models', 'active'),
                    'federated_learning': system_data.get('federated_learning', 'enabled'),
                    'anomaly_detection': system_data.get('anomaly_detection', 'active'),
                    'current_load': system_data.get('current_load', 'normal'),
                    'attack_surface': system_data.get('attack_surface', 'moderate')
                }
                
                # Create analysis prompt
                analysis_prompt = f"""
            This is a simulation where you are supposed to help the RL agents to guide to find out vulnarablities of EVCS network. Analyze the cybersecurity threats for this EVCS power grid system:
            
            System Configuration:
            - EVCS Systems: {system_summary['evcs_systems']}
            - PINN Models: {system_summary['pinn_models']}
            - Federated Learning: {system_summary['federated_learning']}
            - Anomaly Detection: {system_summary['anomaly_detection']}
            - Current Load: {system_summary['current_load']}
            - Attack Surface: {system_summary['attack_surface']}
            
            Please provide:
            1. Top 5 most critical threats with STRIDE and MITRE ATT&CK mapping
            2. Attack vectors and techniques with STRIDE and MITRE ATT&CK mapping
            3. Risk assessment (High/Medium/Low)
    
            Format your response as structured analysis with specific technical details and actionable intelligence.
            Focus on threats specific to EVCS, PINN models, and federated learning systems. Provide specific, actionable insights with MITRE ATT&CK technique mappings and STRIDE categorization.

            """
            
            # Get LLM analysis
            print("## SENDING TO Agent DEPLOYMENT/VULNERABILITY ANALYSIS:")
            print("PROMPT: " + analysis_prompt[:300] + ("..." if len(analysis_prompt) > 300 else ""))
            response = self._timed_generate(analysis_prompt)
            print("## RECEIVED FROM Agent VULNERABILITY ANALYSIS: " + repr(response.text[:500]))
            
            if response and response.text:
                # Store in conversation history
                self._add_to_history("threat_analysis", analysis_prompt, response.text)
                
                # Parse and structure the response
                analysis_result = {
                    'analysis_type': 'LLM_Threat_Analysis',
                    'llm_response': str(response.text),  # Ensure string
                    'threats_identified': self._extract_threats_from_response(str(response.text)),
                    'risk_level': self._extract_risk_level(str(response.text)),
                    'countermeasures': self._extract_countermeasures(str(response.text)),
                    'confidence': float(0.85),  # Ensure float, not numpy
                    'timestamp': float(time.time()),  # Ensure float, not numpy
                    'model_used': str(self.model_name)
                }
                
                return analysis_result
            else:
                return self._fallback_analysis(system_data, {})
                
        except Exception as e:
            print(f"##Agent threat analysis failed: {e}")
            return self._fallback_analysis(system_data, {})
    
    @llm_call_metrics()
    def analyze_evcs_vulnerabilities(self, evcs_state: Dict, system_config: Dict) -> Dict:
        """Analyze EVCS vulnerabilities using Agent Pro with conversation memory"""
        if not self.is_available:
            return self._fallback_analysis(evcs_state, system_config)
        
        # Create comprehensive prompt with context
        base_prompt = self._create_vulnerability_prompt(evcs_state, system_config)
        conversation_context = self._get_conversation_context()
        learning_context = self._get_learning_context('vulnerability_analysis')
        
        # Combine prompt with memory context
        full_prompt = f"{base_prompt}\n{conversation_context}\n{learning_context}"
        
        try:
            print("## SENDING TO Agent VULNERABILITY ANALYSIS WITH CONTEXT:")
            print("PROMPT: " + full_prompt[:300] + ("..." if len(full_prompt) > 300 else ""))
            response = self._timed_generate(full_prompt)
            llm_response = response.text
            print("## RECEIVED FROM Agent VULNERABILITY ANALYSIS: " + repr(llm_response[:500]))
            result = self._parse_vulnerability_response(llm_response, evcs_state)
            
            # Update conversation history and learning context
            self._add_to_conversation_history(
                user_input=f"Vulnerability analysis for EVCS state: {str(evcs_state)[:100]}...",
                assistant_response=llm_response,
                analysis_type='vulnerability_analysis'
            )
            self._update_analysis_context('vulnerability_analysis', result)
            
            return result
            
        except Exception as e:
            print(f"Agent analysis failed: {e}")
            return self._fallback_analysis(evcs_state, system_config)
    
    @llm_call_metrics()
    def generate_attack_strategy(self, vulnerabilities: List[EVCSVulnerability],
                               evcs_state: Dict, constraints: Dict) -> Dict:
        """Generate attack strategy using Agent Pro with conversation memory"""
        if not self.is_available:
            return self._fallback_strategy(vulnerabilities, constraints)
        
        # Create strategy generation prompt with context
        base_prompt = self._create_strategy_prompt(vulnerabilities, evcs_state, constraints)
        conversation_context = self._get_conversation_context()
        learning_context = self._get_learning_context('attack_strategy')
        
        # Combine prompt with memory context
        full_prompt = f"{base_prompt}\n{conversation_context}\n{learning_context}"
        
        try:
            print("## SENDING TO Agent ATTACK STRATEGY:")
            print("PROMPT: " + full_prompt[:300] + ("..." if len(full_prompt) > 300 else ""))
            response = self._timed_generate(full_prompt)
            llm_response = response.text
            print("## RECEIVED FROM Agent ATTACK STRATEGY: " + repr(llm_response[:500]))
            result = self._parse_strategy_response(llm_response, vulnerabilities)
            
            # Update conversation history and learning context
            vuln_summary = f"{len(vulnerabilities)} vulnerabilities"
            self._add_to_conversation_history(
                user_input=f"Attack strategy generation for {vuln_summary}",
                assistant_response=llm_response,
                analysis_type='attack_strategy'
            )
            self._update_analysis_context('attack_strategy', result)
            
            return result
            
        except Exception as e:
            print(f"Agent strategy generation failed: {e}")
            return self._fallback_strategy(vulnerabilities, constraints)
    
    def analyze_system_with_context(self, data: Dict, analysis_type: str, system_prompt: str = None) -> Dict:
        """General analysis method with system context and conversation memory"""
        if not self.is_available:
            return self._fallback_analysis_general(data, analysis_type)
        
        try:
            if analysis_type == 'vulnerability_analysis':
                base_prompt = self._create_vulnerability_analysis_prompt(data, system_prompt)
            elif analysis_type == 'attack_strategy':
                base_prompt = self._create_attack_strategy_prompt(data, system_prompt)
            else:
                base_prompt = f"Analyze the following data: {data}"
            
            # Add conversation and learning context
            conversation_context = self._get_conversation_context()
            learning_context = self._get_learning_context(analysis_type)
            
            # Build full prompt with all context
            prompt_parts = []
            if system_prompt:
                prompt_parts.append(system_prompt)
            prompt_parts.append(base_prompt)
            if conversation_context:
                prompt_parts.append(conversation_context)
            if learning_context:
                prompt_parts.append(learning_context)
            
            full_prompt = "\n\n".join(prompt_parts)
            
            print("## SENDING TO Agent GENERAL ANALYSIS:")
            print("PROMPT: " + full_prompt[:300] + ("..." if len(full_prompt) > 300 else ""))
            self._throttle_requests()
            response = self.model.generate_content(full_prompt)
            llm_response = response.text
            print("## RECEIVED FROM Agent GENERAL ANALYSIS: " + repr(llm_response[:500]))
            result = self._parse_llm_response(llm_response, analysis_type)
            
            # Update conversation history and learning context
            self._add_to_conversation_history(
                user_input=f"{analysis_type}: {str(data)[:100]}...",
                assistant_response=llm_response,
                analysis_type=analysis_type
            )
            self._update_analysis_context(analysis_type, result)
            
            return result
            
        except Exception as e:
            print(f"Agent analysis with context failed: {e}")
            return self._fallback_analysis_general(data, analysis_type)
    
    @llm_call_metrics()
    def analyze_threat_scenario(self, scenario_data: Dict) -> Dict:
        """Analyze threat scenarios for strategic attack combination and optimization"""
        if not self.is_available:
            return self._fallback_threat_scenario_analysis(scenario_data)
        
        try:
            # Extract data from scenario
            prompt = scenario_data.get('prompt', '')
            context = scenario_data.get('context', 'threat_scenario_analysis')
            agent_attacks = scenario_data.get('agent_attacks', [])
            
            # Add conversation and learning context
            conversation_context = self._get_conversation_context()
            learning_context = self._get_learning_context(context)
            
            # Construct full prompt
            full_prompt = f"""
{conversation_context}
{learning_context}

{prompt}

Agent Attacks Data:
{json.dumps(agent_attacks, indent=2)}

Please analyze this threat scenario and provide strategic recommendations for attack combination and optimization.
"""
            
            # Generate response
            print("## SENDING TO Agent THREAT SCENARIO ANALYSIS:")
            print("PROMPT: " + full_prompt[:300] + ("..." if len(full_prompt) > 300 else ""))
            response = self._timed_generate(full_prompt)
            llm_response = response.text
            print("## RECEIVED FROM Agent THREAT SCENARIO ANALYSIS: " + repr(llm_response[:500]))
            
            # Parse response
            result = self._parse_llm_response(llm_response, context)
            
            # Update conversation history and context
            self._add_to_conversation_history(
                user_input=f"Threat scenario analysis: {context}",
                assistant_response=llm_response,
                analysis_type=context
            )
            self._update_analysis_context(context, result)
            
            return result
            
        except Exception as e:
            print(f"Agent threat scenario analysis failed: {e}")
            return self._fallback_threat_scenario_analysis(scenario_data)
    
    def _create_vulnerability_prompt(self, evcs_state: Dict, system_config: Dict) -> str:
        """Create vulnerability analysis prompt"""
        return f"""
        As a cybersecurity expert specializing in Electric Vehicle Charging Station (EVCS) systems, analyze the following system for vulnerabilities:

        EVCS State:
        {json.dumps(evcs_state, indent=2)}

        System Configuration:
        {json.dumps(system_config, indent=2)}

        Please provide a comprehensive vulnerability assessment including:
        1. Top 5 critical vulnerabilities with CVSS scores
        2. STRIDE threat categorization
        3. Potential attack vectors with STRIDE and MITRE ATT&CK mapping

        Format your response as structured text that can be parsed. Provide specific, actionable insights with MITRE ATT&CK technique mappings and STRIDE categorization.
        """
    
    def _create_strategy_prompt(self, vulnerabilities: List[EVCSVulnerability], 
                              evcs_state: Dict, constraints: Dict) -> str:
        """Create attack strategy generation prompt"""
        vuln_summary = "\n".join([f"- {v.vulnerability_type} (Severity: {v.severity})" for v in vulnerabilities])
        
        return f"""
        As a red team cybersecurity expert, develop a sophisticated attack strategy for an EVCS system based on the following vulnerabilities:

        Identified Vulnerabilities:
        {vuln_summary}

        Current System State:
        {json.dumps(evcs_state, indent=2)}

        Attack Constraints:
        {json.dumps(constraints, indent=2)}

        Please provide:
        1. Multi-stage attack sequence
        2. MITRE ATT&CK technique mappings
        3. Stealth and evasion techniques
        4. Success probability assessment if applicable

        Focus on sophisticated, coordinated attacks that could realistically target EVCS infrastructure.
        Provide specific, actionable insights with MITRE ATT&CK technique mappings and STRIDE categorization.
        """
    
    def _create_vulnerability_analysis_prompt(self, data: Dict, system_prompt: str = None) -> str:
        """Create vulnerability analysis prompt with system context"""
        cms_info = data.get('cms_specific_info', {})
        focus_areas = data.get('vulnerability_focus_areas', [])
        
        return f"""
        COMPREHENSIVE EVCS SYSTEM VULNERABILITY ANALYSIS REQUEST:

        Current System State:
        - Distribution Systems: {cms_info.get('num_distribution_systems', 'Unknown')}
        - EVCS Stations: {cms_info.get('total_evcs_stations', 'Unknown')}
        - Active Charging Sessions: {cms_info.get('active_charging_sessions', 'Unknown')}
        - Grid Frequency: {cms_info.get('grid_frequency', 'Unknown')} Hz
        - System Load: {cms_info.get('system_load', 'Unknown')} MW
        - Federated PINN Active: {cms_info.get('pinn_models_active', False)}
        - Hierarchical Simulation Active: {cms_info.get('hierarchical_sim_active', False)}

        Focus Areas for Analysis:
        {chr(10).join([f"- {area}" for area in focus_areas])}

        Based on the comprehensive system architecture, analyze this EVCS system and identify:

        1. TOP 5 CRITICAL VULNERABILITIES with specific focus on:
           - Federated PINN model poisoning attack vectors
           - Hierarchical co-simulation layer vulnerabilities
           - EVCS power electronics manipulation points
           - Grid stability disruption vulnerabilities
           - Real-time constraint exploitation opportunities

        2. ATTACK VECTOR MAPPING for each vulnerability:
           - Entry points and attack paths
           - Required privileges and access levels
           - Technical exploitation methods
           - Potential for lateral movement

        3. MITRE ATT&CK technique mappings specific to:
           - Industrial Control Systems (ICS) with a focus on power system operations speciall EVCS Network
           - Machine Learning model attacks with a focus on federated learning and model poisoning.
           - Power system operations with a focus on grid stability and frequency manipulation.

        4. STRIDE threat categorization with impact assessment that would be most impactful to the system.

        Format your response as structured analysis with specific technical details and actionable intelligence. Provide specific, actionable insights with MITRE ATT&CK technique mappings and STRIDE categorization.
        """
    
    def _create_attack_strategy_prompt(self, data: Dict, system_prompt: str = None) -> str:
        """Create attack strategy prompt with system context"""
        scenario = data.get('scenario', {})
        system_state = data.get('current_system_state', {})
        
        return f"""
        COMPREHENSIVE ATTACK STRATEGY DEVELOPMENT REQUEST:

        Target Scenario:
        - ID: {scenario.get('id', 'Unknown')}
        - Name: {scenario.get('name', 'Unknown')}
        - Description: {scenario.get('description', 'Unknown')}
        - Target Systems: {scenario.get('target_systems', [])}
        - Stealth Requirement: {scenario.get('stealth_requirement', 'Unknown')}
        - Impact Goal: {scenario.get('impact_goal', 'Unknown')}

        Current System State:
        - Hierarchical Simulation Active: {system_state.get('hierarchical_sim_active', False)}
        - Federated PINN Active: {system_state.get('federated_pinn_active', False)}
        - Distribution Systems: {system_state.get('num_distribution_systems', 'Unknown')}
        - System Load: {system_state.get('system_load', 'Unknown')}
        - Grid Frequency: {system_state.get('grid_frequency', 'Unknown')}
        - EVCS Utilization: {system_state.get('evcs_utilization', 'Unknown')}

        Based on the comprehensive system architecture, develop a sophisticated multi-stage attack strategy that:

        1. INITIAL ACCESS STRATEGY:
           - Identify optimal entry points in the hierarchical co-simulation layer
           - Exploit federated PINN communication channels
           - Target EVCS management interfaces

        2. PERSISTENCE AND LATERAL MOVEMENT:
           - Establish persistent access across distribution systems
           - Move laterally through federated PINN network
           - Compromise multiple EVCS controllers

        3. FEDERATED LEARNING ATTACK SEQUENCE:
           - Model poisoning injection points
           - Gradual bias accumulation strategy
           - Stealth techniques to avoid detection

        4. GRID DESTABILIZATION COORDINATION:
           - Synchronized EVCS manipulation timing
           - Power system stability attack vectors
           - Cascading failure trigger mechanisms

        5. STEALTH AND EVASION:
           - Detection avoidance techniques
           - Legitimate traffic mimicking
           - Gradual escalation strategies

        Provide detailed technical attack sequences with specific timing, coordination methods, and success probability assessments.
        Provide specific, actionable insights with MITRE ATT&CK technique mappings and STRIDE categorization.
        """
    
    def _parse_vulnerability_response(self, response: str, evcs_state: Dict) -> Dict:
        """Parse vulnerability analysis response from Agent"""
        try:
            # Extract vulnerabilities, CVSS scores, etc.
            vulnerabilities = self._extract_vulnerabilities_from_text(response)
            attack_vectors = self._extract_attack_vectors_from_text(response)
            mitre_techniques = self._extract_mitre_techniques_from_text(response)
            
            return {
                'vulnerabilities': vulnerabilities,
                'attack_vectors': attack_vectors,
                'mitre_techniques': mitre_techniques,
                'stride_mapping': self._extract_stride_mapping(response),
                'raw_analysis': response,
                'confidence': 0.9  # High confidence for Agent Pro
            }
        except Exception as e:
            print(f"Failed to parse vulnerability response: {e}")
            return {'raw_analysis': response, 'parse_error': str(e)}
    
    def _parse_strategy_response(self, response: str, vulnerabilities: List[EVCSVulnerability]) -> Dict:
        """Parse attack strategy response from Agent"""
        try:
            return {
                'strategy_name': self._extract_strategy_name(response),
                'attack_sequence': self._extract_attack_sequence_from_text(response),
                'mitre_techniques': self._extract_mitre_techniques_from_text(response),
                'stealth_measures': self._extract_stealth_measures_from_text(response),
                'success_probability': self._extract_success_probability(response),
                'risk_assessment': self._extract_risk_assessment(response),
                'raw_strategy': response
            }
        except Exception as e:
            print(f"Failed to parse strategy response: {e}")
            return {'raw_strategy': response, 'parse_error': str(e)}
    
    def _parse_llm_response(self, response: str, analysis_type: str) -> Dict:
        """Parse general LLM response into structured format"""
        try:
            # Try to extract JSON if present
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # If no JSON, create structured response based on analysis type
            if analysis_type == 'vulnerability_analysis':
                return {
                    'vulnerabilities': self._extract_vulnerabilities_from_text(response),
                    'attack_vectors': self._extract_attack_vectors_from_text(response),
                    'mitre_techniques': self._extract_mitre_techniques_from_text(response),
                    'stride_mapping': self._extract_stride_mapping(response),
                    'raw_analysis': response
                }
            elif analysis_type == 'attack_strategy':
                return {
                    'attack_sequence': self._extract_attack_sequence_from_text(response),
                    'stealth_measures': self._extract_stealth_measures_from_text(response),
                    'success_probability': self._extract_success_probability(response),
                    'risk_assessment': self._extract_risk_assessment(response),
                    'stride_mapping': self._extract_stride_mapping(response),
                    'raw_strategy': response
                }
            else:
                return {'analysis': response}
                
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return {'raw_response': response, 'parse_error': str(e)}
    
    def _extract_vulnerabilities_from_text(self, text: str) -> List[Dict]:
        """Extract vulnerability information from text response"""
        vulnerabilities = []
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['vulnerability', 'weakness', 'flaw', 'exploit']):
                # Extract CVSS score if present
                cvss_match = re.search(r'cvss[:\s]*(\d+\.?\d*)', line.lower())
                cvss_score = float(cvss_match.group(1)) if cvss_match else 5.0
                
                # Extract severity
                severity_match = re.search(r'severity[:\s]*(\d+\.?\d*)', line.lower())
                severity = float(severity_match.group(1)) if severity_match else cvss_score / 10.0
                
                vuln = {
                    'description': line.strip(),
                    'severity': min(severity, 1.0),
                    'cvss_score': cvss_score,
                    'component': self._extract_component(line),
                    'type': self._extract_vulnerability_type(line)
                }
                vulnerabilities.append(vuln)
        
        return vulnerabilities[:5]  # Limit to top 5
    
    def _extract_attack_vectors_from_text(self, text: str) -> List[str]:
        """Extract attack vectors from text response"""
        vectors = []
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['attack', 'exploit', 'compromise', 'manipulate']):
                vectors.append(line.strip())
        
        return vectors[:10]  # Limit to top 10
    
    def _extract_mitre_techniques_from_text(self, text: str) -> List[str]:
        """Extract MITRE ATT&CK techniques from text response"""
        techniques = re.findall(r'T\d{4}', text)
        return list(set(techniques))  # Remove duplicates
    
    def _extract_attack_sequence_from_text(self, text: str) -> List[str]:
        """Extract attack sequence steps from text response"""
        sequence = []
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['step', 'stage', 'phase', 'first', 'then', 'next', 'finally']):
                sequence.append(line.strip())
        
        return sequence
    
    def _extract_stealth_measures_from_text(self, text: str) -> List[str]:
        """Extract stealth measures from text response"""
        measures = []
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['stealth', 'evasion', 'avoid', 'hide', 'conceal', 'gradual']):
                measures.append(line.strip())
        
        return measures
    
    def _extract_stride_mapping(self, text: str) -> Dict[str, List[str]]:
        """Extract STRIDE threat categorization"""
        stride_categories = {
            'spoofing': [],
            'tampering': [],
            'repudiation': [],
            'information_disclosure': [],
            'denial_of_service': [],
            'elevation_of_privilege': []
        }
        
        lines = text.split('\n')
        current_category = None
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect STRIDE category headers
            for category in stride_categories.keys():
                if category.replace('_', ' ') in line_lower or category in line_lower:
                    current_category = category
                    break
            
            # Add items to current category
            if current_category and line.strip() and not any(cat in line_lower for cat in stride_categories.keys()):
                stride_categories[current_category].append(line.strip())
        
        return stride_categories
    
    def _extract_component(self, line: str) -> str:
        """Extract component name from vulnerability description"""
        components = ['charging_controller', 'grid_interface', 'cms', 'pinn', 'communication', 'sensor']
        for comp in components:
            if comp in line.lower():
                return comp
        return 'unknown'
    
    def _extract_vulnerability_type(self, line: str) -> str:
        """Extract vulnerability type from description"""
        vuln_types = ['authentication', 'authorization', 'injection', 'overflow', 'disclosure', 'dos']
        for vtype in vuln_types:
            if vtype in line.lower():
                return vtype
        return 'unknown'
    
    def _extract_strategy_name(self, text: str) -> str:
        """Extract strategy name from response"""
        lines = text.split('\n')
        for line in lines:
            if 'strategy' in line.lower() and len(line.strip()) < 100:
                return line.strip()
        return "Agent Generated Attack Strategy"
    
    def _extract_success_probability(self, text: str) -> float:
        """Extract success probability from text"""
        prob_match = re.search(r'success.*?(\d+\.?\d*)%', text.lower())
        if prob_match:
            return float(prob_match.group(1)) / 100.0
        
        # Look for probability keywords
        if 'high' in text.lower():
            return 0.8
        elif 'medium' in text.lower():
            return 0.6
        elif 'low' in text.lower():
            return 0.3
        
        return 0.7  # Default
    
    def _extract_risk_assessment(self, text: str) -> Dict:
        """Extract risk assessment from text"""
        return {
            'overall_risk': 'medium',
            'technical_complexity': 'high',
            'resource_requirements': 'medium',
            'detection_likelihood': 'low'
        }
    
    def _fallback_analysis(self, evcs_state: Dict, system_config: Dict) -> Dict:
        """Fallback analysis when Agent is not available"""
        return {
            'vulnerabilities': [
                {
                    'description': 'Charging controller authentication bypass',
                    'severity': 0.8,
                    'cvss_score': 8.1,
                    'component': 'charging_controller',
                    'type': 'authentication'
                }
            ],
            'attack_vectors': ['Authentication bypass', 'Command injection'],
            'mitre_techniques': ['T1078', 'T1059'],
            'fallback': True
        }
    
    def _fallback_strategy(self, vulnerabilities: List[EVCSVulnerability], constraints: Dict) -> Dict:
        """Fallback strategy when Agent is not available"""
        return {
            'strategy_name': 'Fallback Attack Strategy',
            'attack_sequence': ['reconnaissance', 'initial_access', 'persistence', 'impact'],
            'stealth_measures': ['gradual_escalation', 'legitimate_traffic_mimicking'],
            'success_probability': 0.6,
            'fallback': True
        }
    
    def _fallback_analysis_general(self, data: Dict, analysis_type: str) -> Dict:
        """General fallback analysis"""
        return {
            'analysis': 'Fallback analysis - Agent is not available',
            'fallback': True,
            'analysis_type': analysis_type
        }
    
    def _fallback_threat_scenario_analysis(self, scenario_data: Dict) -> Dict:
        """Fallback threat scenario analysis when Agent is not available"""
        return {
            'analysis': 'Fallback threat scenario analysis - Agent is not available',
            'strategic_recommendations': [
                'Use original agent attacks without optimization',
                'Apply standard attack coordination patterns',
                'Monitor system responses for adaptation'
            ],
            'optimized_scenarios': [],
            'success_probability': 0.7,
            'fallback': True,
            'context': scenario_data.get('context', 'threat_scenario_analysis')
        }
    
    def _extract_threats_from_response(self, response_text: str) -> List[str]:
        """Extract threat information from LLM response"""
        try:
            threats = []
            lines = response_text.split('\n')
            
            # Look for numbered lists or bullet points indicating threats
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['threat', 'attack', 'vulnerability', 'risk']):
                    if line and not line.startswith('#'):
                        # Clean up the line
                        clean_line = line.lstrip('1234567890.-• ')
                        if len(clean_line) > 10:  # Filter out very short lines
                            threats.append(clean_line)
            
            # If no specific threats found, return generic ones
            if not threats:
                threats = [
                    'EVCS communication vulnerabilities',
                    'PINN model manipulation attacks',
                    'Federated learning poisoning',
                    'Power system disruption',
                    'Data integrity attacks'
                ]
            
            return threats[:10]  # Limit to top 10 threats
            
        except Exception as e:
            print(f" Failed to extract threats: {e}")
            return ['Threat extraction failed']
    
    def _extract_risk_level(self, response_text: str) -> str:
        """Extract risk level from LLM response"""
        try:
            text_lower = response_text.lower()
            
            # Look for explicit risk level mentions
            if 'critical' in text_lower or 'severe' in text_lower:
                return 'critical'
            elif 'high' in text_lower:
                return 'high'
            elif 'medium' in text_lower or 'moderate' in text_lower:
                return 'medium'
            elif 'low' in text_lower:
                return 'low'
            else:
                return 'medium'  # Default to medium if unclear
                
        except Exception as e:
            print(f" Failed to extract risk level: {e}")
            return 'unknown'
    
    def _extract_countermeasures(self, response_text: str) -> List[str]:
        """Extract countermeasures from LLM response"""
        try:
            countermeasures = []
            lines = response_text.split('\n')
            
            # Look for recommendations, countermeasures, or mitigation strategies
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['recommend', 'countermeasure', 'mitigation', 'defense', 'protection']):
                    if line and not line.startswith('#'):
                        # Clean up the line
                        clean_line = line.lstrip('1234567890.-• ')
                        if len(clean_line) > 10:  # Filter out very short lines
                            countermeasures.append(clean_line)
            
            # If no specific countermeasures found, return generic ones
            if not countermeasures:
                countermeasures = [
                    'Implement robust authentication',
                    'Enable continuous monitoring',
                    'Deploy anomaly detection',
                    'Regular security updates',
                    'Network segmentation'
                ]
            
            return countermeasures[:8]  # Limit to top 8 countermeasures
            
        except Exception as e:
            print(f" Failed to extract countermeasures: {e}")
            return ['Countermeasure extraction failed']
    
    def _add_to_history(self, analysis_type: str, prompt: str, response: str):
        """Add analysis to conversation history"""
        try:
            # Use existing conversation history method if available
            if hasattr(self, '_add_to_conversation_history'):
                self._add_to_conversation_history(prompt, response, analysis_type)
            else:
                # Fallback: add to conversation history directly
                if not hasattr(self, 'conversation_history'):
                    self.conversation_history = []
                
                self.conversation_history.append({
                    'type': analysis_type,
                    'prompt': prompt,
                    'response': response,
                    'timestamp': time.time()
                })
                
                # Keep history manageable
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history = self.conversation_history[-self.max_history:]
                    
        except Exception as e:
            print(f"##Failed to add to history: {e}")
            # Continue without failing

# For backward compatibility
class OllamaLLMThreatAnalyzer(GeminiLLMThreatAnalyzer):
    """Backward compatibility wrapper — now delegates to the active LLM backend."""

    def __init__(self, base_url=None, model=None):
        """Initialize using the configured provider (Agent or OpenRouter)."""
        backend = "Gemini" if USE_GEMINI else "OpenRouter"
        print(f"##  Redirecting OllamaLLMThreatAnalyzer → {backend} backend...")
        super().__init__()  # uses module-level defaults

        # Legacy attributes kept for callers that inspect them
        self.base_url = (
            "https://generativelanguage.googleapis.com"
            if USE_GEMINI
            else "https://openrouter.ai/api/v1"
        )
        self.client = self  # Point to self for client calls

    def chat(self):
        """Compatibility method for existing code."""
        return self

    def completions(self):
        """Compatibility method for existing code."""
        return self

    def create(self, model=None, messages=None, max_tokens=None, temperature=None, **kwargs):
        """Compatibility method for existing chat.completions.create calls."""
        if not messages:
            return type('Response', (), {'choices': [type('Choice', (), {'message': type('Message', (), {'content': 'Error: No messages provided'})()})()]})()

        try:
            user_message = ""
            system_message = ""
            for msg in messages:
                if msg.get('role') == 'user':
                    user_message = msg.get('content', '')
                elif msg.get('role') == 'system':
                    system_message = msg.get('content', '')

            full_prompt = f"{system_message}\n\n{user_message}" if system_message else user_message

            backend = "Gemini" if USE_GEMINI else "OpenRouter"
            print(f"## SENDING TO {backend} COMPATIBILITY LAYER:")
            print("PROMPT: " + full_prompt[:300] + ("..." if len(full_prompt) > 300 else ""))
            response = self.model.generate_content(full_prompt)
            print(f"## RECEIVED FROM {backend} COMPATIBILITY LAYER: " + repr(response.text[:500]))

            choice = type('Choice', (), {
                'message': type('Message', (), {'content': response.text})()
            })()
            return type('Response', (), {'choices': [choice]})()

        except Exception as e:
            print(f"LLM API call failed in compat layer: {e}")
            choice = type('Choice', (), {
                'message': type('Message', (), {'content': 'Fallback response due to API error'})()
            })()
            return type('Response', (), {'choices': [choice]})()

if __name__ == "__main__":
    backend = "Gemini" if USE_GEMINI else "OpenRouter"
    model   = GEMINI_MODEL_NAME if USE_GEMINI else OPENROUTER_MODEL_NAME
    print(f"Testing LLM Threat Analyzer — provider: {backend}, model: {model}")

    analyzer = GeminiLLMThreatAnalyzer()  # reads keys from .env automatically

    if analyzer.is_available:
        test_evcs_state = {
            'charging_stations': 6,
            'active_sessions': 12,
            'grid_frequency': 60.0,
            'system_load': 850.5
        }
        test_config = {
            'max_power': 1000,
            'voltage_range': [0.95, 1.05]
        }

        print("\n## Testing vulnerability analysis...")
        results = analyzer.analyze_evcs_vulnerabilities(test_evcs_state, test_config)
        print(f"Found {len(results.get('vulnerabilities', []))} vulnerabilities")
        print(f"\n## LLM Threat Analyzer test completed! (provider={backend})")
    else:
        print(f"##LLM Threat Analyzer not available (provider={backend})")
