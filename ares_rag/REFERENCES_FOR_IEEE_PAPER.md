# References for IEEE Paper - RAG Knowledge Base Sources

Based on the ARES RAG system implementation, here are the references and links for the sources mentioned in the paper (excluding datasets and literature PDFs):

---

## 1. MITRE ATT&CK for ICS

**Description:** Structured threat taxonomy providing adversary tactics, techniques, and procedures for Industrial Control Systems

**Source URL:** 
```
https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/ics-attack/ics-attack.json
```

**Official Website:**
```
https://attack.mitre.org/matrices/ics/
```

**Reference Format:**
```
MITRE Corporation, "MITRE ATT&CK for ICS," [Online]. Available: https://attack.mitre.org/matrices/ics/
```

**In Code:** `config.py` line 27, `collectors/mitre_collector.py`

---

## 2. NVD CVE Database

**Description:** National Vulnerability Database providing Common Vulnerabilities and Exposures

**API Endpoint:**
```
https://services.nvd.nist.gov/rest/json/cves/2.0
```

**Official Website:**
```
https://nvd.nist.gov/
```

**Reference Format:**
```
National Institute of Standards and Technology, "National Vulnerability Database," [Online]. Available: https://nvd.nist.gov/
```

**In Code:** `config.py` line 26, `collectors/nvd_collector.py`

---

## 3. CISA ICS-CERT Advisories

**Description:** Real-world vulnerability disclosures for Industrial Control Systems from the Cybersecurity and Infrastructure Security Agency

**Base URL:**
```
https://www.cisa.gov/uscert/ics/advisories
```

**Specific Advisories Used:**

1. **ICSA-23-166-01** - Siemens SICAM A8000 RTU
   - URL: `https://www.cisa.gov/uscert/ics/advisories/icsa-23-166-01`
   - CVEs: CVE-2023-32730, CVE-2023-32731, CVE-2023-32732
   - CVSS: 9.8 (Critical)

2. **ICSA-22-349-01** - ABB System 800xA
   - URL: `https://www.cisa.gov/uscert/ics/advisories/icsa-22-349-01`

3. **ICSA-22-256-01** - Multiple EVSE Vendors (OCPP vulnerabilities)
   - URL: `https://www.cisa.gov/uscert/ics/advisories/icsa-22-256-01`

4. **ICSA-21-336-01** - Schneider Electric EVlink
   - URL: `https://www.cisa.gov/uscert/ics/advisories/icsa-21-336-01`
   - CVE: CVE-2021-22707 (hard-coded credentials)

5. **ICSA-23-073-01** - Hitachi Energy RTU500
   - URL: `https://www.cisa.gov/uscert/ics/advisories/icsa-23-073-01`

6. **ICSA-22-090-01** - GE Digital SCADA
   - URL: `https://www.cisa.gov/uscert/ics/advisories/icsa-22-090-01`

7. **ICSA-23-129-01** - Rockwell Automation ControlLogix
   - URL: `https://www.cisa.gov/uscert/ics/advisories/icsa-23-129-01`

8. **ICSA-22-174-01** - Mitsubishi Electric MELSEC
   - URL: `https://www.cisa.gov/uscert/ics/advisories/icsa-22-174-01`

**Reference Format:**
```
Cybersecurity and Infrastructure Security Agency, "ICS-CERT Advisories," [Online]. Available: https://www.cisa.gov/uscert/ics/advisories
```

**In Code:** `collectors/ics_cert_collector.py`

---

## 4. STRIDE Threat Modeling Framework

**Description:** Microsoft's threat categorization framework for systematic security analysis

**Original Paper:**
```
L. Hernan, S. Lambert, T. Ostwald, and A. Shostack, "Uncover Security Design Flaws Using The STRIDE Approach," MSDN Magazine, Nov. 2006.
```

**Microsoft Documentation:**
```
https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats
```

**Reference Format:**
```
Microsoft Corporation, "The STRIDE Threat Model," [Online]. Available: https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats
```

**In Code:** `collectors/stride_collector.py` (custom EVSE-specific patterns)

---

## 5. Protocol-Specific Security Analyses

### OCPP (Open Charge Point Protocol)

**Official Specification:**
```
https://www.openchargealliance.org/protocols/ocpp/
```

**Reference Format:**
```
Open Charge Alliance, "Open Charge Point Protocol," [Online]. Available: https://www.openchargealliance.org/protocols/ocpp/
```

### DNP3 (Distributed Network Protocol)

**IEEE Standard:**
```
IEEE Std 1815-2012 (IEEE Standard for Electric Power Systems Communications - Distributed Network Protocol)
```

**Reference Format:**
```
IEEE, "IEEE Standard for Electric Power Systems Communications - Distributed Network Protocol (DNP3)," IEEE Std 1815-2012, 2012.
```

### Modbus Protocol

**Official Specification:**
```
https://modbus.org/specs.php
```

**Reference Format:**
```
Modbus Organization, "Modbus Application Protocol Specification," [Online]. Available: https://modbus.org/specs.php
```

**In Code:** `collectors/protocol_vuln_collector.py`

---

## 6. Embedding Model

**Model:** all-MiniLM-L6-v2

**Source:**
```
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

**Reference Format:**
```
N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in Proc. Conf. Empirical Methods Natural Language Process. (EMNLP), 2019.
```

**In Code:** `config.py` line 19, `vector_db/embedder.py`

---

## 7. Vector Database

**ChromaDB:**
```
https://www.trychroma.com/
```

**Reference Format:**
```
Chroma, "Chroma - the AI-native open-source embedding database," [Online]. Available: https://www.trychroma.com/
```

**In Code:** `vector_db/chroma_manager.py`

---

## 8. Foundation Model

**Gemini 2.5 Flash:**
```
https://ai.google.dev/gemini-api/docs
```

**Reference Format:**
```
Google DeepMind, "Gemini: A Family of Highly Capable Multimodal Models," arXiv:2312.11805, 2023.
```

**In Code:** `config.py` line 16, `gemini_rag.py`

---

## Summary for IEEE Paper

**Recommended Citation Style:**

For the paragraph mentioning the knowledge base sources, you can add these citations:

```latex
The knowledge base integrates structured threat taxonomies (MITRE ATT\&CK for ICS~\cite{mitre_attack_ics}, STRIDE patterns~\cite{stride_microsoft}), real-world vulnerability disclosures (NVD CVE database~\cite{nvd_nist}, ICS-CERT advisories~\cite{cisa_ics_cert}), protocol-specific security analyses (OCPP~\cite{ocpp_spec}, DNP3~\cite{ieee_dnp3}, Modbus~\cite{modbus_spec}, TCP/IP), empirical attack datasets (CICEVSE2024), and peer-reviewed research literature on EVCS and power grid security.
```

**BibTeX Entries:**

```bibtex
@misc{mitre_attack_ics,
  author = {{MITRE Corporation}},
  title = {{MITRE ATT\&CK for ICS}},
  howpublished = {\url{https://attack.mitre.org/matrices/ics/}},
  note = {Accessed: 2026-02-17}
}

@misc{stride_microsoft,
  author = {Microsoft Corporation},
  title = {{The STRIDE Threat Model}},
  howpublished = {\url{https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats}},
  note = {Accessed: 2026-02-17}
}

@misc{nvd_nist,
  author = {{National Institute of Standards and Technology}},
  title = {{National Vulnerability Database}},
  howpublished = {\url{https://nvd.nist.gov/}},
  note = {Accessed: 2026-02-17}
}

@misc{cisa_ics_cert,
  author = {{Cybersecurity and Infrastructure Security Agency}},
  title = {{ICS-CERT Advisories}},
  howpublished = {\url{https://www.cisa.gov/uscert/ics/advisories}},
  note = {Accessed: 2026-02-17}
}

@misc{ocpp_spec,
  author = {{Open Charge Alliance}},
  title = {{Open Charge Point Protocol}},
  howpublished = {\url{https://www.openchargealliance.org/protocols/ocpp/}},
  note = {Accessed: 2026-02-17}
}

@standard{ieee_dnp3,
  author = {{IEEE}},
  title = {{IEEE Standard for Electric Power Systems Communications - Distributed Network Protocol (DNP3)}},
  journal = {IEEE Std 1815-2012},
  year = {2012}
}

@misc{modbus_spec,
  author = {{Modbus Organization}},
  title = {{Modbus Application Protocol Specification}},
  howpublished = {\url{https://modbus.org/specs.php}},
  note = {Accessed: 2026-02-17}
}

@inproceedings{sentence_bert,
  author = {Reimers, Nils and Gurevych, Iryna},
  title = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year = {2019}
}

@misc{gemini_model,
  author = {{Google DeepMind}},
  title = {{Gemini: A Family of Highly Capable Multimodal Models}},
  journal = {arXiv preprint arXiv:2312.11805},
  year = {2023}
}
```
