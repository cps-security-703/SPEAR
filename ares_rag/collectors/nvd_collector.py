import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger
from tqdm import tqdm

from config import config
from schemas import VulnerabilityDocument

class NVDCollector:
    """
    Collector for NVD CVE data focused on EVSE, AGC, Power Systems, and related infrastructure
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = config.NVD_BASE_URL
        self.api_key = api_key or config.NVD_API_KEY
        self.headers = {}
        if self.api_key:
            self.headers['apiKey'] = self.api_key
        
        logger.info(f"Initialized NVDCollector with API key: {'Yes' if self.api_key else 'No'}")
    
    def collect_evse_power_cves(self, start_date: str = "2022-01-01", max_results: int = 100) -> List[Dict]:
        """
        Collect CVEs related to EVSE, AGC, Power Systems, and related infrastructure
        
        Args:
            start_date: Start date for CVE search (ISO format)
            max_results: Maximum number of CVEs to collect
            
        Returns:
            List of raw CVE data from NVD
        """
        logger.info(f"Starting CVE collection from {start_date}, max results: {max_results}")
        logger.info(f"Using API key: {'Yes' if self.api_key else 'No'}")
        logger.info(f"NVD API endpoint: {self.base_url}")
        
        all_cves = []
        seen_cve_ids = set()
        successful_keywords = 0
        
        for keyword in tqdm(config.CVE_KEYWORDS, desc="Searching CVE keywords"):
            try:
                cves = self._search_by_keyword(keyword, start_date)
                
                if cves:
                    successful_keywords += 1
                    logger.debug(f"Found {len(cves)} CVEs for keyword: {keyword}")
                
                for cve in cves:
                    cve_id = cve['cve']['id']
                    if cve_id not in seen_cve_ids:
                        all_cves.append(cve)
                        seen_cve_ids.add(cve_id)
                        
                        if len(all_cves) >= max_results:
                            logger.info(f"Reached max results limit: {max_results}")
                            return all_cves
                
                time.sleep(6 if not self.api_key else 0.6)
                
            except Exception as e:
                logger.error(f"Error fetching CVEs for keyword '{keyword}': {e}")
                continue
        
        logger.info(f"Collected {len(all_cves)} unique CVEs from {successful_keywords}/{len(config.CVE_KEYWORDS)} keywords")
        
        if len(all_cves) == 0:
            logger.warning("No CVEs collected. This might be due to:")
            logger.warning("  1. NVD API endpoint changes")
            logger.warning("  2. Network connectivity issues")
            logger.warning("  3. API rate limiting")
            logger.warning("  4. Invalid date range")
            logger.warning("Consider running with --skip-nvd to use other data sources")
        
        return all_cves
    
    def _search_by_keyword(self, keyword: str, start_date: str) -> List[Dict]:
        """Search NVD by keyword"""
        # NVD API 2.0 uses different parameter names
        params = {
            'keywordSearch': keyword,
            'pubStartDate': f"{start_date}T00:00:00.000",
            'pubEndDate': datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000"),
            'resultsPerPage': 20
        }
        
        try:
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('vulnerabilities', [])
            elif response.status_code == 403:
                logger.warning("Rate limit exceeded, waiting 60 seconds...")
                time.sleep(60)
                return []
            elif response.status_code == 404:
                # 404 might mean no results or API endpoint issue
                logger.debug(f"No results found for keyword: {keyword}")
                return []
            else:
                logger.error(f"NVD API error: {response.status_code} - {response.text[:200]}")
                return []
                
        except Exception as e:
            logger.error(f"Request failed for keyword '{keyword}': {e}")
            return []
    
    def process_cve_to_document(self, cve_data: Dict) -> VulnerabilityDocument:
        """
        Convert NVD CVE format to VulnerabilityDocument schema
        
        Args:
            cve_data: Raw CVE data from NVD API
            
        Returns:
            VulnerabilityDocument instance
        """
        cve = cve_data['cve']
        cve_id = cve['id']
        
        description = self._extract_description(cve)
        cvss_score, severity = self._extract_cvss(cve)
        references = self._extract_references(cve)
        
        stride_categories = self._infer_stride(description)
        affected_systems = self._infer_affected_systems(description)
        attack_vector = self._extract_attack_vector(cve)
        
        embedding_text = f"{cve_id} {description}"
        keywords = self._extract_keywords(description)
        
        document = VulnerabilityDocument(
            doc_id=cve_id,
            type="vulnerability",
            title=cve_id,
            description=description,
            source=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
            date_published=cve.get('published', ''),
            last_updated=cve.get('lastModified', ''),
            stride_categories=stride_categories,
            mitre_tactics=[],
            mitre_techniques=[],
            attack_vector=attack_vector,
            severity=severity,
            affected_systems=affected_systems,
            affected_components=[],
            industry_sectors=["Energy", "Transportation", "Critical Infrastructure"],
            mitigation_strategies=[],
            detection_methods=[],
            defensive_actions=[],
            countermeasures=[],
            prerequisites="",
            impact=description,
            cvss_score=cvss_score,
            exploitability=self._map_cvss_to_exploitability(cvss_score),
            references=references,
            embedding_text=embedding_text,
            keywords=keywords,
            relevance_tags=["cve", "vulnerability", "nvd"]
        )
        
        return document
    
    def _extract_description(self, cve: Dict) -> str:
        """Extract English description from CVE"""
        if 'descriptions' in cve:
            for desc in cve['descriptions']:
                if desc.get('lang') == 'en':
                    return desc.get('value', '')
        return ""
    
    def _extract_cvss(self, cve: Dict) -> tuple[float, str]:
        """Extract CVSS score and severity"""
        cvss_score = 0.0
        severity = "Unknown"
        
        if 'metrics' in cve:
            if 'cvssMetricV31' in cve['metrics'] and cve['metrics']['cvssMetricV31']:
                metric = cve['metrics']['cvssMetricV31'][0]
                cvss_score = metric['cvssData'].get('baseScore', 0.0)
                severity = metric['cvssData'].get('baseSeverity', 'Unknown')
            elif 'cvssMetricV30' in cve['metrics'] and cve['metrics']['cvssMetricV30']:
                metric = cve['metrics']['cvssMetricV30'][0]
                cvss_score = metric['cvssData'].get('baseScore', 0.0)
                severity = metric['cvssData'].get('baseSeverity', 'Unknown')
            elif 'cvssMetricV2' in cve['metrics'] and cve['metrics']['cvssMetricV2']:
                metric = cve['metrics']['cvssMetricV2'][0]
                cvss_score = metric['cvssData'].get('baseScore', 0.0)
                severity = metric.get('baseSeverity', 'Unknown')
        
        return cvss_score, severity
    
    def _extract_references(self, cve: Dict) -> List[str]:
        """Extract reference URLs"""
        references = []
        if 'references' in cve:
            references = [ref.get('url', '') for ref in cve['references'][:5]]
        return references
    
    def _extract_attack_vector(self, cve: Dict) -> str:
        """Extract attack vector from CVSS metrics"""
        if 'metrics' in cve:
            if 'cvssMetricV31' in cve['metrics'] and cve['metrics']['cvssMetricV31']:
                return cve['metrics']['cvssMetricV31'][0]['cvssData'].get('attackVector', 'Network')
            elif 'cvssMetricV30' in cve['metrics'] and cve['metrics']['cvssMetricV30']:
                return cve['metrics']['cvssMetricV30'][0]['cvssData'].get('attackVector', 'Network')
        return "Network"
    
    def _infer_stride(self, description: str) -> List[str]:
        """Infer STRIDE categories from description"""
        stride = []
        desc_lower = description.lower()
        
        stride_patterns = {
            "Spoofing": ['spoof', 'imperson', 'fake identity', 'authentication bypass'],
            "Tampering": ['tamper', 'modify', 'alter', 'inject', 'manipulation'],
            "Repudiation": ['deny', 'log', 'audit', 'trace', 'non-repudiation'],
            "Information Disclosure": ['disclosure', 'leak', 'expose', 'information', 'confidential'],
            "Denial of Service": ['denial', 'dos', 'ddos', 'crash', 'unavailable', 'resource exhaustion'],
            "Elevation of Privilege": ['privilege', 'escalat', 'unauthorized', 'bypass', 'admin']
        }
        
        for category, patterns in stride_patterns.items():
            if any(pattern in desc_lower for pattern in patterns):
                stride.append(category)
        
        return stride if stride else ["Unknown"]
    
    def _infer_affected_systems(self, description: str) -> List[str]:
        """Infer affected systems from description"""
        systems = []
        desc_lower = description.lower()
        
        system_keywords = {
            "EVSE": ['ev charging', 'evse', 'charging station', 'charger', 'electric vehicle supply', "OCPP", "ISO 15118", "CHAdeMO", "CCS", "J1772","charge point", "charging infrastructure", "vehicle-to-grid", "V2G", "V2X","smart charging", "load management"],
            "AGC": ['agc', 'automatic generation control', 'generation control'],
            "DMS": ['dms', 'distribution management', 'distribution system'],
            "CMS": ['charging management', 'cms'],
            "CCMS": ['central charging', 'ccms'],
            "SCADA": ['scada', 'supervisory control'],
            "ICS": ['industrial control', 'ics', 'operational technology'],
            "ADMS": ['adms', 'advanced distribution'],
            "DER": ['der', 'distributed energy', 'microgrid'],
            "Grid": ['power grid', 'smart grid', 'electrical grid', 'substation'],
            "EMS": ['energy management', 'ems'],
            "Protocol": ["Modbus", "DNP3", "IEC 61850", "MQTT", "CoAP", "HTTP", "HTTPS", "TCP", "UDP"],
            "Vendor": ["ABB", "Siemens", "ChargePoint", "EVBox", "Schneider Electric", "Tesla Supercharger", "Electrify America"],

        }
        
        for system, keywords in system_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                systems.append(system)
        
        return systems if systems else ["General"]
    
    def _map_cvss_to_exploitability(self, cvss_score: float) -> str:
        """Map CVSS score to exploitability level"""
        if cvss_score >= 9.0:
            return "Critical"
        elif cvss_score >= 7.0:
            return "High"
        elif cvss_score >= 4.0:
            return "Medium"
        else:
            return "Low"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        common_words = {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 
            'this', 'that', 'with', 'from', 'by', 'as', 'is', 'was', 'are', 'be'
        }
        
        words = text.lower().split()
        keywords = [w.strip('.,;:!?()[]{}') for w in words if len(w) > 4 and w not in common_words]
        
        unique_keywords = list(set(keywords))[:15]
        return unique_keywords
    
    def save_raw_data(self, cves: List[Dict], filename: str = "nvd_cves_raw.json"):
        """Save raw CVE data to file"""
        filepath = config.RAW_DATA_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cves, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(cves)} raw CVEs to {filepath}")
    
    def save_processed_documents(self, documents: List[VulnerabilityDocument], filename: str = "nvd_documents.json"):
        """Save processed documents to file"""
        filepath = config.PROCESSED_DATA_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([doc.model_dump() for doc in documents], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} processed documents to {filepath}")
