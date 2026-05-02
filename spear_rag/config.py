import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    CHROMA_PERSIST_DIR = BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    
    NVD_API_KEY = os.getenv("NVD_API_KEY", "7d4d7d7d-3341-4154-99d2-1c449d6f2968")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "evse_vulnerability_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    MAX_CVE_RESULTS = int(os.getenv("MAX_CVE_RESULTS", "100"))
    MAX_MITRE_TECHNIQUES = int(os.getenv("MAX_MITRE_TECHNIQUES", "60"))
    MAX_STRIDE_PATTERNS = int(os.getenv("MAX_STRIDE_PATTERNS", "40"))
    
    NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    MITRE_STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/ics-attack/ics-attack.json"
    
    TARGET_SYSTEMS = [
        "EVSE", "EV Charging", "Electric Vehicle",
        "AGC", "Automatic Generation Control",
        "DMS", "Distribution Management System",
        "CMS", "Charging Management System",
        "CCMS", "Central Charging Management System",
        "SCADA", "ICS", "Industrial Control System",
        "Power Grid", "Smart Grid", "Substation",
        "DER", "Distributed Energy Resources"
    ]
    
    CVE_KEYWORDS = [
        "SCADA", "ICS", "industrial control",
        "electric vehicle", "EV charging", "EVSE", "charging station",
        "power grid", "smart grid", "substation", "distribution",
        "DER", "distributed energy", "microgrid",
        "EMS", "energy management", "ADMS", "DMS",
        "AGC", "automatic generation control",
        "charging management", "billing system"
    ]
    
    @classmethod
    def create_directories(cls):
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        cls.CHROMA_PERSIST_DIR.mkdir(exist_ok=True)

config = Config()
