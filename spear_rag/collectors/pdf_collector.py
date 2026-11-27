

import os
import re
from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime

try:
    import PyPDF2
except ImportError:
    logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    logger.warning("pdfplumber not installed. Install with: pip install pdfplumber")
    pdfplumber = None

from schemas import VulnerabilityDocument

class PDFCollector:


    def __init__(self, pdf_directory: str = "pdf", chunk_size: int = 1000, chunk_overlap: int = 200):

        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if not PyPDF2 and not pdfplumber:
            raise ImportError("Either PyPDF2 or pdfplumber must be installed. Install with: pip install PyPDF2 pdfplumber")

    def extract_text_from_pdf(self, pdf_path: str) -> str:

        text = ""


        if pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                logger.info(f"Extracted {len(text)} characters using pdfplumber from {os.path.basename(pdf_path)}")
                return text
            except Exception as e:
                logger.warning(f"pdfplumber failed for {pdf_path}: {e}, trying PyPDF2...")


        if PyPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                logger.info(f"Extracted {len(text)} characters using PyPDF2 from {os.path.basename(pdf_path)}")
                return text
            except Exception as e:
                logger.error(f"PyPDF2 failed for {pdf_path}: {e}")
                return ""

        return text

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:

        chunks = []


        text = re.sub(r'\s+', ' ', text).strip()


        start = 0
        chunk_num = 1

        while start < len(text):
            end = start + self.chunk_size


            if end < len(text):

                sentence_end = text.rfind('. ', end - 100, end)
                if sentence_end != -1:
                    end = sentence_end + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_number'] = chunk_num
                chunk_metadata['total_chunks'] = 'TBD'
                chunk_metadata['char_start'] = start
                chunk_metadata['char_end'] = end

                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })

                chunk_num += 1


            start = end - self.chunk_overlap


        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)

        return chunks

    def extract_metadata_from_filename(self, filename: str) -> Dict:

        metadata = {
            'source_file': filename,
            'type': 'pdf_document'
        }


        filename_lower = filename.lower()

        if 'threat' in filename_lower and 'model' in filename_lower:
            metadata['document_category'] = 'threat_modeling'
        elif 'secur' in filename_lower:
            metadata['document_category'] = 'security_research'
        elif 'pnnl' in filename_lower or 'report' in filename_lower:
            metadata['document_category'] = 'technical_report'
        elif 'charging' in filename_lower or 'vehicle' in filename_lower or 'evse' in filename_lower:
            metadata['document_category'] = 'evse_security'
        else:
            metadata['document_category'] = 'general'

        return metadata

    def extract_keywords_from_text(self, text: str) -> List[str]:

        keywords = set()


        security_keywords = [
            'vulnerability', 'attack', 'threat', 'exploit', 'malicious',
            'authentication', 'authorization', 'encryption', 'security',
            'mitigation', 'defense', 'protection', 'risk', 'compromise'
        ]


        evse_keywords = [
            'evse', 'charging', 'vehicle', 'ocpp', 'dnp3', 'modbus',
            'grid', 'power', 'energy', 'station', 'charger', 'ev'
        ]


        stride_keywords = [
            'spoofing', 'tampering', 'repudiation', 'disclosure',
            'denial of service', 'elevation of privilege'
        ]


        mitre_pattern = r'T\d{4}'

        text_lower = text.lower()


        for keyword in security_keywords + evse_keywords + stride_keywords:
            if keyword in text_lower:
                keywords.add(keyword)


        mitre_matches = re.findall(mitre_pattern, text)
        keywords.update(mitre_matches)


        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        cve_matches = re.findall(cve_pattern, text.upper())
        keywords.update([cve.lower() for cve in cve_matches])

        return sorted(list(keywords))

    def create_documents_from_pdf(self, pdf_path: str) -> List[VulnerabilityDocument]:

        filename = os.path.basename(pdf_path)
        logger.info(f"Processing PDF: {filename}")


        text = self.extract_text_from_pdf(pdf_path)

        if not text or len(text) < 100:
            logger.warning(f"Insufficient text extracted from {filename} ({len(text)} chars)")
            return []


        base_metadata = self.extract_metadata_from_filename(filename)
        base_metadata['date_published'] = datetime.now().strftime('%Y-%m-%d')
        base_metadata['source'] = f"PDF Document: {filename}"


        keywords = self.extract_keywords_from_text(text)


        chunks = self.chunk_text(text, base_metadata)

        logger.info(f"Created {len(chunks)} chunks from {filename}")


        documents = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"PDF-{filename.replace('.pdf', '').replace(' ', '_').upper()}-CHUNK-{i+1:03d}"


            chunk_keywords = self.extract_keywords_from_text(chunk['text'])
            all_keywords = sorted(list(set(keywords + chunk_keywords)))


            title = f"{filename.replace('.pdf', '')} - Part {i+1}/{len(chunks)}"


            description = chunk['text'][:200].strip() + "..." if len(chunk['text']) > 200 else chunk['text']


            severity = self._infer_severity(chunk['text'])


            doc = VulnerabilityDocument(
                doc_id=chunk_id,
                type='pdf_document',
                title=title,
                description=description,
                source=base_metadata['source'],
                date_published=base_metadata['date_published'],
                severity=severity,
                cvss_score=0.0,
                affected_systems=self._extract_affected_systems(chunk['text']),
                attack_vector="N/A",
                exploitability="N/A",
                mitre_techniques=self._extract_mitre_techniques(chunk['text']),
                mitre_tactics=[],
                stride_categories=self._extract_stride_categories(chunk['text']),
                keywords=all_keywords,
                relevance_tags=[base_metadata['document_category'], 'pdf_document', 'research'],
                embedding_text=chunk['text']
            )

            documents.append(doc)

        return documents

    def _infer_severity(self, text: str) -> str:

        text_lower = text.lower()

        if any(word in text_lower for word in ['critical', 'severe', 'catastrophic', 'high risk']):
            return "High"
        elif any(word in text_lower for word in ['moderate', 'medium', 'significant']):
            return "Medium"
        else:
            return "Low"

    def _extract_affected_systems(self, text: str) -> List[str]:

        systems = []
        system_keywords = {
            'EVSE': ['evse', 'charging station', 'charger'],
            'CMS': ['central management', 'cms', 'management system'],
            'CCMS': ['ccms', 'charging management'],
            'DMS': ['dms', 'distribution management'],
            'SCADA': ['scada', 'supervisory control'],
            'Grid': ['grid', 'power grid', 'electrical grid'],
            'AGC': ['agc', 'automatic generation control']
        }

        text_lower = text.lower()
        for system, keywords in system_keywords.items():
            if any(kw in text_lower for kw in keywords):
                systems.append(system)

        return systems if systems else ['General']

    def _extract_mitre_techniques(self, text: str) -> List[str]:

        pattern = r'T\d{4}'
        return list(set(re.findall(pattern, text)))

    def _extract_stride_categories(self, text: str) -> List[str]:

        categories = []
        stride_map = {
            'Spoofing': ['spoof', 'impersonat', 'fake identity'],
            'Tampering': ['tamper', 'modif', 'alter data'],
            'Repudiation': ['repudiat', 'deny', 'non-repudiation'],
            'Information Disclosure': ['disclosure', 'leak', 'expos'],
            'Denial of Service': ['denial of service', 'dos', 'ddos'],
            'Elevation of Privilege': ['privilege escalation', 'elevation', 'unauthorized access']
        }

        text_lower = text.lower()
        for category, keywords in stride_map.items():
            if any(kw in text_lower for kw in keywords):
                categories.append(category)

        return categories

    def collect(self) -> List[VulnerabilityDocument]:

        if not os.path.exists(self.pdf_directory):
            logger.error(f"PDF directory not found: {self.pdf_directory}")
            return []

        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        all_documents = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            try:
                documents = self.create_documents_from_pdf(pdf_path)
                all_documents.extend(documents)
                logger.info(f"Successfully processed {pdf_file}: {len(documents)} chunks created")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                continue

        logger.info(f"Total PDF documents created: {len(all_documents)}")

        return all_documents

if __name__ == "__main__":

    collector = PDFCollector(pdf_directory="pdf", chunk_size=1000, chunk_overlap=200)
    documents = collector.collect()

    print(f"\nProcessed {len(documents)} document chunks from PDFs")

    if documents:
        print("\nSample document:")
        sample = documents[0]
        print(f"ID: {sample.id}")
        print(f"Title: {sample.title}")
        print(f"Description: {sample.description[:200]}...")
        print(f"Keywords: {', '.join(sample.keywords[:10])}")
        print(f"MITRE Techniques: {', '.join(sample.mitre_techniques)}")
        print(f"STRIDE Categories: {', '.join(sample.stride_categories)}")
