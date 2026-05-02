from typing import List, Optional
from loguru import logger
from tqdm import tqdm

from config import config
from schemas import VulnerabilityDocument
from collectors import (
    NVDCollector,
    MITRECollector,
    STRIDECollector,
    CICEVSECollector,
    MITRESTRIDEMapper
)
from collectors.ics_cert_collector import ICSCERTCollector
from collectors.protocol_vuln_collector import ProtocolVulnerabilityCollector
from collectors.pdf_collector import PDFCollector
from vector_db import ChromaDBManager, DocumentEmbedder

class VulnerabilityDBPipeline:
    """
    Main pipeline for creating and populating the vulnerability vector database
    """
    
    def __init__(self):
        logger.info("Initializing Vulnerability DB Pipeline")
        config.create_directories()
        
        self.nvd_collector = NVDCollector()
        self.mitre_collector = MITRECollector()
        self.stride_collector = STRIDECollector()
        self.cicevse_collector = CICEVSECollector()
        self.mitre_stride_mapper = MITRESTRIDEMapper()
        self.ics_cert_collector = ICSCERTCollector()
        self.protocol_collector = ProtocolVulnerabilityCollector()
        self.pdf_collector = PDFCollector(pdf_directory='pdf', chunk_size=1000, chunk_overlap=200)
        
        self.embedder = DocumentEmbedder()
        self.db_manager = ChromaDBManager()
        
        self.all_documents: List[VulnerabilityDocument] = []
    
    def collect_nvd_data(self, start_date: str = "2022-01-01", max_results: int = 100) -> List[VulnerabilityDocument]:
        """
        Collect and process NVD CVE data
        
        Args:
            start_date: Start date for CVE collection
            max_results: Maximum number of CVEs to collect
            
        Returns:
            List of processed VulnerabilityDocument instances
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: Collecting NVD CVE Data")
        logger.info("=" * 80)
        
        raw_cves = self.nvd_collector.collect_evse_power_cves(
            start_date=start_date,
            max_results=max_results
        )
        
        self.nvd_collector.save_raw_data(raw_cves)
        
        documents = []
        for cve_data in tqdm(raw_cves, desc="Processing CVEs"):
            doc = self.nvd_collector.process_cve_to_document(cve_data)
            documents.append(doc)
        
        self.nvd_collector.save_processed_documents(documents)
        
        logger.info(f"Collected and processed {len(documents)} NVD CVE documents")
        return documents
    
    def collect_mitre_data(self) -> List[VulnerabilityDocument]:
        """
        Collect and process MITRE ATT&CK for ICS data
        
        Returns:
            List of processed VulnerabilityDocument instances
        """
        logger.info("=" * 80)
        logger.info("PHASE 2: Collecting MITRE ATT&CK for ICS Data")
        logger.info("=" * 80)
        
        raw_techniques = self.mitre_collector.collect_ics_techniques()
        
        self.mitre_collector.save_raw_data(raw_techniques)
        
        documents = []
        for technique in tqdm(raw_techniques, desc="Processing MITRE techniques"):
            doc = self.mitre_collector.process_technique_to_document(technique)
            documents.append(doc)
        
        self.mitre_collector.save_processed_documents(documents)
        
        logger.info(f"Collected and processed {len(documents)} MITRE technique documents")
        return documents
    
    def collect_stride_patterns(self) -> List[VulnerabilityDocument]:
        """
        Create STRIDE threat patterns for EVSE/Power Systems
        
        Returns:
            List of VulnerabilityDocument instances
        """
        logger.info("=" * 80)
        logger.info("PHASE 3: Creating STRIDE Threat Patterns")
        logger.info("=" * 80)
        
        documents = self.stride_collector.create_evse_stride_patterns()
        
        self.stride_collector.save_processed_documents(documents)
        
        logger.info(f"Created {len(documents)} STRIDE pattern documents")
        return documents
    
    def collect_mitre_stride_mappings(self) -> List[VulnerabilityDocument]:
        """
        Create MITRE-STRIDE mappings for EVSE context
        
        Returns:
            List of VulnerabilityDocument instances
        """
        logger.info("=" * 80)
        logger.info("PHASE 4: Creating MITRE-STRIDE Mappings")
        logger.info("=" * 80)
        
        documents = self.mitre_stride_mapper.create_comprehensive_mappings()
        
        self.mitre_stride_mapper.save_processed_documents(documents)
        
        logger.info(f"Created {len(documents)} MITRE-STRIDE mapping documents")
        return documents
    
    def collect_cicevse_data(self, dataset_path: Optional[str] = None) -> List[VulnerabilityDocument]:
        """
        Process CICEVSE2024 dataset
        
        Args:
            dataset_path: Path to CICEVSE2024 CSV file
            
        Returns:
            List of VulnerabilityDocument instances
        """
        logger.info("=" * 80)
        logger.info("PHASE 5: Processing CICEVSE2024 Dataset")
        logger.info("=" * 80)
        
        if dataset_path is None:
            logger.warning("No CICEVSE2024 dataset path provided. Skipping this phase.")
            logger.info("To include CICEVSE2024 data, provide the dataset path.")
            return []
        
        df = self.cicevse_collector.load_dataset(dataset_path)
        
        if df.empty:
            logger.warning("Failed to load CICEVSE2024 dataset. Skipping this phase.")
            return []
        
        patterns = self.cicevse_collector.analyze_attack_patterns(df)
        documents = self.cicevse_collector.create_documents_from_patterns(patterns)
        
        self.cicevse_collector.save_processed_documents(documents)
        
        logger.info(f"Created {len(documents)} CICEVSE2024 documents")
        return documents
    
    def collect_ics_cert_advisories(self) -> List[VulnerabilityDocument]:
        """
        Create ICS-CERT advisory documents
        
        Returns:
            List of VulnerabilityDocument instances
        """
        documents = self.ics_cert_collector.create_evse_ics_advisories()
        self.ics_cert_collector.save_processed_documents(documents)
        logger.info(f"Created {len(documents)} ICS-CERT advisory documents")
        return documents
    
    def collect_protocol_vulnerabilities(self) -> List[VulnerabilityDocument]:
        """
        Create protocol vulnerability documents
        
        Returns:
            List of VulnerabilityDocument instances
        """
        documents = self.protocol_collector.create_protocol_vulnerabilities()
        self.protocol_collector.save_processed_documents(documents)
        logger.info(f"Created {len(documents)} protocol vulnerability documents")
        return documents
    
    def collect_pdf_documents(self) -> List[VulnerabilityDocument]:
        """
        Process PDF research documents and create chunked documents
        
        Returns:
            List of VulnerabilityDocument instances
        """
        documents = self.pdf_collector.collect()
        
        # Save to processed directory
        if documents:
            import json
            output_path = config.PROCESSED_DATA_DIR / 'pdf_documents.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([doc.model_dump() for doc in documents], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(documents)} PDF document chunks to {output_path}")
        
        return documents
    
    def embed_documents(self, documents: List[VulnerabilityDocument]) -> List[List[float]]:
        """
        Generate embeddings for documents
        
        Args:
            documents: List of VulnerabilityDocument instances
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(documents)} documents")
        
        texts = [doc.embedding_text for doc in documents]
        embeddings = self.embedder.embed_batch(texts, batch_size=32, show_progress=True)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def populate_vector_db(self, documents: List[VulnerabilityDocument], embeddings: List[List[float]]):
        """
        Populate vector database with documents and embeddings
        
        Args:
            documents: List of VulnerabilityDocument instances
            embeddings: List of embedding vectors
        """
        logger.info(f"Populating vector database with {len(documents)} documents")
        
        self.db_manager.add_documents(documents, embeddings)
        
        logger.info("Vector database populated successfully")
    
    def run_full_pipeline(
        self,
        nvd_start_date: str = "2022-01-01",
        nvd_max_results: int = 100,
        cicevse_dataset_path: Optional[str] = None,
        skip_nvd: bool = False,
        skip_mitre: bool = False,
        skip_stride: bool = False,
        skip_mitre_stride: bool = False,
        skip_cicevse: bool = False
    ):
        """
        Run the full pipeline to create the vulnerability vector database
        
        """
        logger.info("=" * 80)
        logger.info("STARTING VULNERABILITY VECTOR DATABASE CREATION PIPELINE")
        logger.info("=" * 80)
        
        all_documents = []
        
        if not skip_nvd:
            nvd_docs = self.collect_nvd_data(nvd_start_date, nvd_max_results)
            all_documents.extend(nvd_docs)
        
        if not skip_mitre:
            mitre_docs = self.collect_mitre_data()
            all_documents.extend(mitre_docs)
        
        if not skip_stride:
            stride_docs = self.collect_stride_patterns()
            all_documents.extend(stride_docs)
        
        if not skip_mitre_stride:
            mitre_stride_docs = self.collect_mitre_stride_mappings()
            all_documents.extend(mitre_stride_docs)
        
        if not skip_cicevse and cicevse_dataset_path:
            cicevse_docs = self.collect_cicevse_data(cicevse_dataset_path)
            all_documents.extend(cicevse_docs)
        
        # Phase 6: ICS-CERT Advisories
        logger.info("=" * 80)
        logger.info("PHASE 6: Creating ICS-CERT Advisory Documents")
        logger.info("=" * 80)
        ics_cert_docs = self.collect_ics_cert_advisories()
        all_documents.extend(ics_cert_docs)
        
        # Phase 7: Protocol Vulnerabilities
        logger.info("=" * 80)
        logger.info("PHASE 7: Creating Protocol Vulnerability Documents")
        logger.info("=" * 80)
        protocol_docs = self.collect_protocol_vulnerabilities()
        all_documents.extend(protocol_docs)
        
        # Phase 8: PDF Documents
        logger.info("=" * 80)
        logger.info("PHASE 8: Processing PDF Research Documents")
        logger.info("=" * 80)
        pdf_docs = self.collect_pdf_documents()
        all_documents.extend(pdf_docs)
        
        logger.info("=" * 80)
        logger.info("PHASE 9: Generating Embeddings and Populating Vector DB")
        logger.info("=" * 80)
        
        if all_documents:
            embeddings = self.embed_documents(all_documents)
            self.populate_vector_db(all_documents, embeddings)
            
            self.all_documents = all_documents
            
            stats = self.db_manager.get_collection_stats()
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Total documents in database: {stats['total_documents']}")
            logger.info(f"Document types: {stats.get('document_types', {})}")
            logger.info(f"Severity distribution: {stats.get('severity_distribution', {})}")
        else:
            logger.warning("No documents collected. Pipeline completed with empty database.")
    
    def get_database_stats(self) -> dict:
        """Get statistics about the vector database"""
        return self.db_manager.get_collection_stats()
    
    def export_database(self, filepath: str):
        """Export the vector database to JSON"""
        self.db_manager.export_collection(filepath)
