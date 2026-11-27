import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
from loguru import logger
import json

from config import config
from schemas import VulnerabilityDocument

class ChromaDBManager:
    """
    Manager for ChromaDB vector database
    Handles document storage, retrieval, and querying
    """
    
    def __init__(self, persist_directory: str = None, collection_name: str = None):
        self.persist_directory = persist_directory or str(config.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or config.COLLECTION_NAME
        
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing collection"""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "EVSE and Power Systems Vulnerability Database",
                    "embedding_model": config.EMBEDDING_MODEL,
                    "version": "1.0"
                }
            )
            logger.info(f"Collection '{self.collection_name}' initialized with {self.collection.count()} documents")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def add_documents(self, documents: List[VulnerabilityDocument], embeddings: List[List[float]]):
        """
        Add documents to the vector database
        
        Args:
            documents: List of VulnerabilityDocument instances
            embeddings: List of embedding vectors
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        logger.info(f"Adding {len(documents)} documents to collection")
        
        ids = [doc.doc_id for doc in documents]
        metadatas = [self._document_to_metadata(doc) for doc in documents]
        documents_text = [doc.embedding_text for doc in documents]
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            logger.info(f"Successfully added {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def _document_to_metadata(self, doc: VulnerabilityDocument) -> Dict[str, Any]:
        """Convert VulnerabilityDocument to metadata dictionary"""
        metadata = {
            "type": doc.type,
            "title": doc.title,
            "description": doc.description[:500],  # Truncate for metadata
            "source": doc.source,
            "date_published": doc.date_published,
            "severity": doc.severity,
            "cvss_score": doc.cvss_score,
            "attack_vector": doc.attack_vector,
            "exploitability": doc.exploitability,
            "stride_categories": json.dumps(doc.stride_categories),
            "mitre_tactics": json.dumps(doc.mitre_tactics),
            "mitre_techniques": json.dumps(doc.mitre_techniques),
            "affected_systems": json.dumps(doc.affected_systems),
            "keywords": json.dumps(doc.keywords[:10]),
            "relevance_tags": json.dumps(doc.relevance_tags)
        }
        
        return metadata
    
    def query(
        self,
        query_text: str = None,
        query_embedding: List[float] = None,
        n_results: int = 10,
        where: Dict = None,
        where_document: Dict = None
    ) -> Dict:
        """
        Query the vector database
        
        Args:
            query_text: Query text (will use collection's embedding function)
            query_embedding: Pre-computed query embedding
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            Query results dictionary
        """
        if query_text is None and query_embedding is None:
            raise ValueError("Either query_text or query_embedding must be provided")
        
        try:
            if query_embedding is not None:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where,
                    where_document=where_document
                )
            else:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where,
                    where_document=where_document
                )
            
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"ids": [], "distances": [], "metadatas": [], "documents": []}
    
    def query_by_filters(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        severity: Optional[str] = None,
        doc_type: Optional[str] = None,
        min_cvss: Optional[float] = None,
        stride_category: Optional[str] = None,
        mitre_technique: Optional[str] = None,
        affected_system: Optional[str] = None
    ) -> Dict:
        """
        Query with specific filters
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results
            severity: Filter by severity
            doc_type: Filter by document type
            min_cvss: Minimum CVSS score
            stride_category: Filter by STRIDE category
            mitre_technique: Filter by MITRE technique
            affected_system: Filter by affected system
            
        Returns:
            Query results
        """
        where_filter = {}
        
        if severity:
            where_filter["severity"] = severity
        
        if doc_type:
            where_filter["type"] = doc_type
        
        if min_cvss is not None:
            where_filter["cvss_score"] = {"$gte": min_cvss}
        
        where_document_filter = None
        
        if stride_category:
            where_document_filter = {"$contains": stride_category}
        
        return self.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where_filter if where_filter else None,
            where_document=where_document_filter
        )
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Get a specific document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None
        """
        try:
            result = self.collection.get(ids=[doc_id])
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'metadata': result['metadatas'][0],
                    'document': result['documents'][0]
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def get_all_documents(self, limit: int = None) -> Dict:
        """
        Get all documents from collection
        
        Args:
            limit: Maximum number of documents to retrieve
            
        Returns:
            All documents
        """
        try:
            if limit:
                return self.collection.get(limit=limit)
            else:
                return self.collection.get()
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return {"ids": [], "metadatas": [], "documents": []}
    
    def delete_documents(self, doc_ids: List[str]):
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
    
    def reset_collection(self):
        """Reset the collection (delete all documents)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self._initialize_collection()
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            
            all_docs = self.collection.get()
            
            stats = {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
            if all_docs['metadatas']:
                doc_types = {}
                severities = {}
                
                for metadata in all_docs['metadatas']:
                    doc_type = metadata.get('type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    severity = metadata.get('severity', 'unknown')
                    severities[severity] = severities.get(severity, 0) + 1
                
                stats['document_types'] = doc_types
                stats['severity_distribution'] = severities
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def export_collection(self, filepath: str):
        """Export collection to JSON file"""
        try:
            all_docs = self.get_all_documents()
            
            export_data = {
                "collection_name": self.collection_name,
                "total_documents": len(all_docs['ids']),
                "documents": []
            }
            
            for i in range(len(all_docs['ids'])):
                doc_data = {
                    "id": all_docs['ids'][i],
                    "metadata": all_docs['metadatas'][i],
                    "document": all_docs['documents'][i]
                }
                export_data['documents'].append(doc_data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(all_docs['ids'])} documents to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export collection: {e}")
