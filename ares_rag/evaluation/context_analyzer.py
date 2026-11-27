"""
Context usage analyzer to track how well RAG leverages retrieved documents
"""

from typing import List, Dict
from loguru import logger
import json

class ContextAnalyzer:
    """
    Analyzes how well RAG responses utilize retrieved context documents
    """
    
    @staticmethod
    def analyze_context_utilization(
        response: str,
        context_docs: List[Dict],
        query: str
    ) -> Dict[str, any]:
        """
        Comprehensive analysis of context utilization
        
        Args:
            response: Generated response
            context_docs: Retrieved context documents
            query: Original query
            
        Returns:
            Detailed context utilization analysis
        """
        if not context_docs:
            return {
                'error': 'No context documents provided',
                'utilization_score': 0.0
            }
        
        response_lower = response.lower()
        
        # Track document-level usage
        doc_usage = []
        
        for i, doc in enumerate(context_docs):
            doc_id = doc.get('id', f'doc_{i}')
            metadata = doc.get('metadata', {})
            content = doc.get('content', '')
            distance = doc.get('distance', 1.0)
            
            usage_info = {
                'doc_id': doc_id,
                'doc_type': metadata.get('type', 'unknown'),
                'title': metadata.get('title', 'N/A'),
                'distance': distance,
                'relevance_rank': i + 1,
                'used': False,
                'usage_indicators': []
            }
            
            # Check if document ID is referenced
            if doc_id.lower() in response_lower:
                usage_info['used'] = True
                usage_info['usage_indicators'].append('doc_id_mentioned')
            
            # Check MITRE techniques
            mitre_techniques = eval(metadata.get('mitre_techniques', '[]'))
            for technique in mitre_techniques:
                if technique.lower() in response_lower:
                    usage_info['used'] = True
                    usage_info['usage_indicators'].append(f'mitre_{technique}')
            
            # Check STRIDE categories
            stride_cats = eval(metadata.get('stride_categories', '[]'))
            for cat in stride_cats:
                if cat.lower() in response_lower:
                    usage_info['used'] = True
                    usage_info['usage_indicators'].append(f'stride_{cat}')
            
            # Check CVE IDs
            if metadata.get('type') == 'vulnerability':
                cve_id = metadata.get('title', '')
                if cve_id and cve_id.lower() in response_lower:
                    usage_info['used'] = True
                    usage_info['usage_indicators'].append(f'cve_{cve_id}')
            
            # Check for content overlap (key phrases)
            content_phrases = ContextAnalyzer._extract_key_phrases(content)
            matched_phrases = []
            for phrase in content_phrases:
                if phrase.lower() in response_lower:
                    matched_phrases.append(phrase)
            
            if matched_phrases:
                usage_info['used'] = True
                usage_info['usage_indicators'].append(f'content_phrases_{len(matched_phrases)}')
                usage_info['matched_phrases'] = matched_phrases[:5]  # Top 5
            
            doc_usage.append(usage_info)
        
        # Calculate utilization metrics
        docs_used = sum(1 for doc in doc_usage if doc['used'])
        utilization_rate = docs_used / len(context_docs) if context_docs else 0.0
        
        # Analyze usage by relevance rank
        top_3_used = sum(1 for doc in doc_usage[:3] if doc['used'])
        top_5_used = sum(1 for doc in doc_usage[:5] if doc['used'])
        
        # Calculate weighted utilization (more weight to top-ranked docs)
        weighted_score = 0.0
        for i, doc in enumerate(doc_usage):
            if doc['used']:
                weight = 1.0 / (i + 1)  # Higher weight for top-ranked docs
                weighted_score += weight
        
        max_weighted_score = sum(1.0 / (i + 1) for i in range(len(context_docs)))
        weighted_utilization = weighted_score / max_weighted_score if max_weighted_score > 0 else 0.0
        
        # Identify unused relevant documents
        unused_relevant = [
            doc for doc in doc_usage 
            if not doc['used'] and doc['distance'] < 0.5  # High relevance but unused
        ]
        
        return {
            'total_documents': len(context_docs),
            'documents_used': docs_used,
            'utilization_rate': utilization_rate,
            'weighted_utilization': weighted_utilization,
            'top_3_used': top_3_used,
            'top_5_used': top_5_used,
            'document_usage_details': doc_usage,
            'unused_relevant_docs': len(unused_relevant),
            'unused_relevant_details': unused_relevant[:3],  # Top 3 missed opportunities
            'utilization_score': weighted_utilization * 100
        }
    
    @staticmethod
    def _extract_key_phrases(text: str, min_length: int = 15) -> List[str]:
        """
        Extract key phrases from text
        
        Args:
            text: Text to extract from
            min_length: Minimum phrase length
            
        Returns:
            List of key phrases
        """
        # Simple extraction: sentences or clauses
        phrases = []
        
        # Split by common delimiters
        for delimiter in ['. ', '! ', '? ', '; ', ', ']:
            parts = text.split(delimiter)
            for part in parts:
                part = part.strip()
                if len(part) >= min_length and len(part) <= 100:
                    phrases.append(part)
        
        return phrases[:20]  # Limit to 20 phrases
    
    @staticmethod
    def generate_context_usage_report(analysis: Dict) -> str:
        """
        Generate human-readable context usage report
        
        Args:
            analysis: Context utilization analysis
            
        Returns:
            Formatted report
        """
        if 'error' in analysis:
            return f"Error: {analysis['error']}"
        
        report = []
        report.append("=" * 80)
        report.append("CONTEXT USAGE ANALYSIS")
        report.append("=" * 80)
        
        # Summary
        report.append(f"\n📊 SUMMARY:")
        report.append(f"   Total Documents Retrieved: {analysis['total_documents']}")
        report.append(f"   Documents Actually Used: {analysis['documents_used']}")
        report.append(f"   Utilization Rate: {analysis['utilization_rate']*100:.1f}%")
        report.append(f"   Weighted Utilization Score: {analysis['utilization_score']:.1f}/100")
        
        # Top documents usage
        report.append(f"\n🎯 TOP DOCUMENTS USAGE:")
        report.append(f"   Top 3 Documents Used: {analysis['top_3_used']}/3")
        report.append(f"   Top 5 Documents Used: {analysis['top_5_used']}/5")
        
        # Document-by-document breakdown
        report.append(f"\n📄 DOCUMENT USAGE DETAILS:")
        for doc in analysis['document_usage_details'][:10]:  # Show top 10
            status = "✓ USED" if doc['used'] else "✗ NOT USED"
            report.append(f"\n   [{doc['relevance_rank']}] {doc['doc_id']} - {status}")
            report.append(f"       Type: {doc['doc_type']}")
            report.append(f"       Title: {doc['title']}")
            report.append(f"       Relevance: {1 - doc['distance']:.2f}")
            if doc['used']:
                report.append(f"       Usage Indicators: {', '.join(doc['usage_indicators'][:3])}")
                if 'matched_phrases' in doc:
                    report.append(f"       Matched Phrases: {len(doc['matched_phrases'])}")
        
        # Missed opportunities
        if analysis['unused_relevant_docs'] > 0:
            report.append(f"\n⚠️  MISSED OPPORTUNITIES:")
            report.append(f"   {analysis['unused_relevant_docs']} highly relevant documents were NOT used")
            for doc in analysis['unused_relevant_details']:
                report.append(f"   - {doc['doc_id']}: {doc['title']}")
        
        report.append("\n" + "=" * 80)
        
        # Recommendations
        if analysis['utilization_rate'] < 0.3:
            report.append("\n💡 RECOMMENDATION: Low context usage detected.")
            report.append("   Consider improving prompts to explicitly instruct using retrieved context.")
        elif analysis['utilization_rate'] < 0.6:
            report.append("\n💡 RECOMMENDATION: Moderate context usage.")
            report.append("   RAG is using some context but could be more comprehensive.")
        else:
            report.append("\n✓ GOOD: High context utilization rate.")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    @staticmethod
    def compare_context_usage(
        rag_analysis: Dict,
        query: str
    ) -> Dict[str, any]:
        """
        Compare context usage across multiple queries or runs
        
        Args:
            rag_analysis: Context analysis results
            query: Query string
            
        Returns:
            Comparison metrics
        """
        return {
            'query': query,
            'utilization_score': rag_analysis['utilization_score'],
            'documents_used': rag_analysis['documents_used'],
            'total_documents': rag_analysis['total_documents'],
            'top_3_usage_rate': rag_analysis['top_3_used'] / 3 * 100,
            'missed_opportunities': rag_analysis['unused_relevant_docs'],
            'quality_indicator': 'Good' if rag_analysis['utilization_score'] > 60 else 'Needs Improvement'
        }
    
    @staticmethod
    def save_analysis(analysis: Dict, filepath: str):
        """
        Save context analysis to file
        
        Args:
            analysis: Analysis results
            filepath: Output file path
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Context analysis saved to {filepath}")
