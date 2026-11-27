import google.generativeai as genai
from typing import List, Dict, Optional
from loguru import logger

from config import config
from vector_db import ChromaDBManager, DocumentEmbedder

class GeminiRAG:
    """
    RAG system using Gemini for vulnerability analysis
    Integrates with the vector database for context retrieval
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GOOGLE_API_KEY
        
        if not self.api_key:
            logger.warning("No Gemini API key provided. Set GOOGLE_API_KEY in .env file")
        else:
            genai.configure(api_key=self.api_key)
            logger.info("Gemini API configured")
        
        self.db_manager = ChromaDBManager()
        self.embedder = DocumentEmbedder()
        
        # Use Gemini 1.5 Flash (gemini-pro is deprecated)
        model_name = config.GEMINI_MODEL
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Initialized Gemini RAG system with {model_name}")
    
    def retrieve_context(
        self,
        query: str,
        n_results: int = 5,
        severity_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant context from vector database
        
        Args:
            query: Query string
            n_results: Number of results to retrieve
            severity_filter: Filter by severity
            doc_type_filter: Filter by document type
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Retrieving context for query: {query[:100]}...")
        
        query_embedding = self.embedder.embed_text(query)
        
        where_filter = {}
        if severity_filter:
            where_filter["severity"] = severity_filter
        if doc_type_filter:
            where_filter["type"] = doc_type_filter
        
        results = self.db_manager.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        context_docs = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'content': results['documents'][0][i]
                }
                context_docs.append(doc)
        
        logger.info(f"Retrieved {len(context_docs)} context documents")
        return context_docs
    
    def format_context(self, context_docs: List[Dict]) -> str:
        """
        Format context documents for Gemini prompt
        
        Args:
            context_docs: List of context documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(context_docs, 1):
            metadata = doc['metadata']
            content = doc['content']
            
            context_part = f"""
Document {i}:
- ID: {doc['id']}
- Type: {metadata.get('type', 'unknown')}
- Title: {metadata.get('title', 'N/A')}
- Severity: {metadata.get('severity', 'N/A')}
- CVSS Score: {metadata.get('cvss_score', 'N/A')}
- STRIDE Categories: {metadata.get('stride_categories', '[]')}
- MITRE Techniques: {metadata.get('mitre_techniques', '[]')}
- Affected Systems: {metadata.get('affected_systems', '[]')}
- Content: {content[:500]}...
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def analyze_vulnerability(
        self,
        system_description: str,
        attack_scenario: Optional[str] = None,
        n_context_docs: int = 10
    ) -> str:
        """
        Analyze vulnerabilities for a given system using RAG
        
        Args:
            system_description: Description of the system to analyze
            attack_scenario: Optional specific attack scenario
            n_context_docs: Number of context documents to retrieve
            
        Returns:
            Gemini's vulnerability analysis
        """
        logger.info("Performing vulnerability analysis with RAG")
        
        query = system_description
        if attack_scenario:
            query += f" {attack_scenario}"
        
        context_docs = self.retrieve_context(query, n_results=n_context_docs)
        context_text = self.format_context(context_docs)
        
        prompt = self._create_vulnerability_analysis_prompt(
            system_description,
            attack_scenario,
            context_text
        )
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating analysis: {e}"
    
    def suggest_rl_attack_strategies(
        self,
        system_description: str,
        objective: str,
        n_context_docs: int = 10
    ) -> str:
        """
        Suggest attack strategies for RL agents
        
        Args:
            system_description: Description of the target system
            objective: Attack objective
            n_context_docs: Number of context documents to retrieve
            
        Returns:
            Attack strategy suggestions
        """
        logger.info("Generating RL attack strategy suggestions")
        
        query = f"{system_description} {objective} attack strategies"
        
        context_docs = self.retrieve_context(query, n_results=n_context_docs)
        context_text = self.format_context(context_docs)
        
        prompt = self._create_rl_attack_prompt(
            system_description,
            objective,
            context_text
        )
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating strategies: {e}"
    
    def _create_vulnerability_analysis_prompt(
        self,
        system_description: str,
        attack_scenario: Optional[str],
        context: str
    ) -> str:
        """Create prompt for vulnerability analysis"""
        prompt = f"""You are a cybersecurity expert specializing in critical infrastructure security, 
particularly Electric Vehicle Supply Equipment (EVSE), power grid systems, and industrial control systems.

System to Analyze:
{system_description}

{f"Attack Scenario: {attack_scenario}" if attack_scenario else ""}

Relevant Security Context from Knowledge Base:
{context}

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE:
1. REVIEW ALL PROVIDED CONTEXT DOCUMENTS - Do not skip any document
2. For EACH context document, extract and use relevant information
3. Reference SPECIFIC MITRE ATT&CK technique IDs (e.g., T0866, T0814) from the context
4. **EXTRACT AND CITE CVE IDs EXPLICITLY** when available in the context:
   - Format: CVE-YYYY-NNNNN (Source/Advisory ID)
   - Example: "CVE-2022-3203 (ICSA-22-256-01): OCPP weak authentication"
   - Include CVE IDs in vulnerability descriptions, not just advisory IDs
   - If a document mentions multiple CVEs, list ALL of them
5. Include STRIDE categories mentioned in the context
6. Cross-reference information across multiple context documents
7. Provide DETAILED, step-by-step recommendations with implementation guidance
8. Use technical terminology from the security domain
9. Organize recommendations by PRIORITY (Critical/High/Medium/Low)
10. Reference industry STANDARDS (IEC 62443, NIST, ISO 15118) where applicable
11. When citing ICS-CERT advisories, ALWAYS extract and mention the CVE IDs contained within them

Based on the provided context, perform a COMPREHENSIVE vulnerability analysis using both MITRE ATT&CK for ICS 
and STRIDE threat modeling frameworks. Your analysis MUST include:

1. **Identified Vulnerabilities**:
   - List SPECIFIC vulnerabilities from the context
   - **MUST include CVE IDs when present** in format: CVE-YYYY-NNNNN (Source)
   - For ICS-CERT advisories, extract ALL CVE IDs mentioned
   - Explain how each vulnerability applies to the described system
   - Rate severity (Critical/High/Medium/Low) based on context
   - Include CVSS scores when available

2. **MITRE ATT&CK Mapping**:
   - Map to SPECIFIC MITRE ATT&CK for ICS techniques from the context
   - Use technique IDs (e.g., T0866 - Exploitation of Remote Services)
   - Explain how each technique could be applied to this system
   - Include tactics (Initial Access, Execution, Persistence, etc.)

3. **STRIDE Threat Analysis**:
   - Categorize threats using ALL relevant STRIDE categories:
     * Spoofing: Identity/authentication attacks
     * Tampering: Data/system modification attacks
     * Repudiation: Denial of actions/events
     * Information Disclosure: Data leakage/exposure
     * Denial of Service: Availability attacks
     * Elevation of Privilege: Unauthorized access escalation
   - Provide specific examples for each applicable category

4. **Attack Vectors and Entry Points**:
   - Describe DETAILED attack vectors based on context
   - Identify specific entry points (protocols, interfaces, services)
   - Explain attack prerequisites and required access levels
   - Include attack complexity and likelihood

5. **Impact Assessment**:
   - Evaluate impact on system operations (operational disruption)
   - Assess impact on grid stability and power delivery
   - Consider safety implications for users and infrastructure
   - Estimate financial and reputational damage
   - Use CVSS scores from context if available

6. **Mitigation Strategies** (MUST be prioritized and detailed):
   - Provide SPECIFIC, actionable mitigation measures from the context
   - ORGANIZE BY PRIORITY with clear labels:
     * CRITICAL: Immediate action required
     * HIGH: Address within 1 week
     * MEDIUM: Address within 1 month
     * LOW: Long-term improvements
   - For EACH mitigation, provide:
     * Configuration changes needed
     * Patches or updates required
     * Architectural improvements
     * Implementation steps (numbered, detailed)
     * Estimated effort and resources
   - Reference industry standards (IEC 62443, NIST SP 800-82, ISO 15118, IEEE 1686, etc.)
   - Include verification methods to confirm mitigation effectiveness

7. **Detection and Monitoring**:
   - Suggest SPECIFIC detection methods from the context
   - Recommend monitoring tools and techniques
   - Define alert conditions and thresholds
   - Include logging requirements and SIEM integration
   - Provide indicators of compromise (IoCs)

8. **References**:
   - List all CVE IDs mentioned
   - List all MITRE ATT&CK technique IDs used
   - Include any other relevant references from context

Provide a DETAILED, COMPREHENSIVE, and ACTIONABLE analysis with specific technical details that security 
professionals can immediately use to improve system security and respond to threats.
"""
        return prompt
    
    def _create_rl_attack_prompt(
        self,
        system_description: str,
        objective: str,
        context: str
    ) -> str:
        """Create prompt for RL attack strategy suggestions"""
        prompt = f"""You are a cybersecurity researcher designing attack scenarios for reinforcement learning (RL) 
agents to test the security of critical infrastructure systems.

Target System:
{system_description}

Attack Objective:
{objective}

Relevant Security Context from Knowledge Base:
{context}

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE:
1. REVIEW ALL PROVIDED CONTEXT DOCUMENTS - Do not skip any document
2. Base ALL suggestions on SPECIFIC vulnerabilities and techniques from the context
3. Reference MITRE ATT&CK technique IDs (e.g., T0866, T0814) from EACH relevant context document
4. **EXTRACT AND CITE CVE IDs** when available in format: CVE-YYYY-NNNNN (Source)
5. Use STRIDE categories from the context documents
6. Cross-reference attack patterns across multiple context documents
7. Provide DETAILED, implementable specifications with concrete examples
8. Include technical details that can be directly coded into an RL agent
9. Prioritize attacks by feasibility and impact based on context evidence

Based on the provided context, design COMPREHENSIVE attack strategies for an RL agent. 
Your response MUST include:

1. **Attack Types and Techniques**:
   - List SPECIFIC attack types from the context
   - **Include CVE IDs when present** in format: CVE-YYYY-NNNNN (Source)
   - Explain why each attack is suitable for RL implementation
   - Map to MITRE ATT&CK for ICS techniques (use technique IDs)
   - Categorize using STRIDE framework
   - Prioritize by feasibility and impact

2. **RL Agent Action Space** (be VERY specific):
   - Define discrete actions the agent can take:
     * Network actions (scan, probe, exploit, inject packets)
     * Protocol actions (OCPP commands, Modbus writes, DNP3 manipulation)
     * System actions (authentication attempts, privilege escalation)
     * Timing actions (delay, burst, sustained attack)
   - Specify action parameters and valid ranges
   - Include action dependencies and prerequisites

3. **State Observation Space** (detailed specification):
   - System state variables to observe:
     * Network metrics (latency, packet loss, bandwidth)
     * System metrics (CPU, memory, connections)
     * Security metrics (failed auth, alerts, anomalies)
     * Domain-specific metrics (charging rate, grid frequency, voltage)
   - Observation frequency and granularity
   - State representation format

4. **Reward Function Design** (mathematical specification):
   - Define reward components:
     * Progress toward objective (quantified)
     * Stealth/detection avoidance (penalty for alerts)
     * Resource efficiency (minimize actions/time)
     * Impact maximization (damage/disruption)
   - Provide reward function formula
   - Specify reward shaping and normalization
   - Include terminal rewards for success/failure

5. **Multi-Step Attack Sequences** (detailed scenarios):
   - Describe 3-5 attack sequences from context:
     * Initial access methods
     * Lateral movement steps
     * Privilege escalation techniques
     * Objective achievement actions
     * Persistence mechanisms
   - Map each step to MITRE techniques
   - Specify decision points and branching

6. **Training Environment Specification**:
   - Simulation requirements
   - Network topology and components
   - Defender behavior modeling
   - Realistic constraints and limitations

7. **Success Criteria and Metrics**:
   - Define measurable success conditions
   - Specify performance metrics
   - Include time-to-compromise targets
   - Define stealth metrics (detection rate)

8. **Realistic Constraints**:
   - Network visibility limitations
   - Rate limiting and detection thresholds
   - Authentication requirements
   - Physical/logical access boundaries
   - Defender response capabilities

9. **Defensive Testing Value**:
   - Explain how this helps improve defenses
   - Identify detection opportunities
   - Suggest countermeasures to test

10. **Implementation Guidance**:
    - Recommend RL algorithms (PPO, DQN, A3C, etc.)
    - Suggest training strategies
    - Provide hyperparameter starting points
    - Include evaluation methodology

Provide DETAILED, TECHNICAL, and IMPLEMENTABLE specifications that an RL researcher can directly use 
to develop and train attack agents for security testing purposes.
"""
        return prompt
    
    def interactive_query(self, query: str, n_context_docs: int = 5) -> str:
        """
        Interactive query with RAG
        
        Args:
            query: User query
            n_context_docs: Number of context documents
            
        Returns:
            Response from Gemini
        """
        logger.info(f"Processing interactive query: {query[:100]}...")
        
        context_docs = self.retrieve_context(query, n_results=n_context_docs)
        context_text = self.format_context(context_docs)
        
        prompt = f"""You are a cybersecurity expert assistant with access to a comprehensive knowledge base 
about EVSE, power grid systems, and industrial control system security.

User Query:
{query}

Relevant Context from Knowledge Base:
{context_text}

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE:
1. REVIEW ALL PROVIDED CONTEXT DOCUMENTS - Extract information from each one
2. Use SPECIFIC information from the context (CVEs, MITRE IDs, attack patterns)
3. Reference MITRE ATT&CK technique IDs when found in context
4. **EXTRACT AND CITE CVE IDs EXPLICITLY** when available in format: CVE-YYYY-NNNNN (Source)
5. Include STRIDE categories when present in context
6. Cross-reference information across multiple context documents
7. Provide detailed, technical answers with proper terminology
8. Include actionable recommendations with implementation steps

Based on the provided context, answer the user's query with:
1. SPECIFIC details from EACH relevant context document:
   - **CVE IDs in format: CVE-YYYY-NNNNN (Source)** when present
   - MITRE ATT&CK technique IDs
   - Vulnerabilities and attack patterns
2. Technical explanations using proper security terminology
3. Actionable recommendations with:
   - Prioritization (Critical/High/Medium/Low)
   - Implementation steps (numbered, detailed)
   - Industry standards references (IEC 62443, NIST, ISO 15118)
4. References to relevant frameworks (MITRE ATT&CK, STRIDE, CVSS)
5. Step-by-step guidance for complex procedures
6. Detection and monitoring recommendations

If the context doesn't contain sufficient information, clearly state:
- Which context documents were reviewed
- What specific information is missing
- General guidance based on cybersecurity best practices
Always prioritize accuracy over completeness.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error processing query: {e}"
