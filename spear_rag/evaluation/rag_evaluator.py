

import google.generativeai as genai
from typing import List, Dict, Optional
from loguru import logger
import json
import time

from config import config
from vector_db import ChromaDBManager, DocumentEmbedder
from evaluation.metrics import EvaluationMetrics

class RAGEvaluator:


    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GOOGLE_API_KEY

        if not self.api_key:
            logger.warning("No Gemini API key provided. Set GOOGLE_API_KEY in .env file")
        else:
            genai.configure(api_key=self.api_key)
            logger.info("Gemini API configured for evaluation")

        self.db_manager = ChromaDBManager()
        self.embedder = DocumentEmbedder()

        model_name = config.GEMINI_MODEL
        self.model = genai.GenerativeModel(model_name)

        logger.info(f"Initialized RAG Evaluator with {model_name}")

    def get_rag_response(self, query: str, n_context_docs: int = 10) -> Dict[str, any]:

        logger.info(f"Getting RAG response for: {query[:100]}...")


        query_embedding = self.embedder.embed_text(query)
        results = self.db_manager.query(
            query_embedding=query_embedding,
            n_results=n_context_docs
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

        context_text = self._format_context(context_docs)


        prompt = f"""You are a cybersecurity expert specializing in EVSE and power grid systems.

User Query:
{query}

Relevant Context from Knowledge Base:
{context_text}

Based on the provided context, answer the query with specific, actionable information.
Include relevant MITRE ATT&CK techniques, STRIDE categories, CVE references, and mitigation strategies from the context.
"""


        try:
            start_time = time.time()
            response = self.model.generate_content(prompt)
            response_time = time.time() - start_time

            return {
                'response': response.text,
                'context_docs': context_docs,
                'n_context_docs': len(context_docs),
                'response_time': response_time,
                'prompt_length': len(prompt)
            }
        except Exception as e:
            logger.error(f"RAG response failed: {e}")
            return {
                'response': f"Error: {e}",
                'context_docs': [],
                'n_context_docs': 0,
                'response_time': 0,
                'prompt_length': 0
            }

    def get_non_rag_response(self, query: str) -> Dict[str, any]:

        logger.info(f"Getting non-RAG response for: {query[:100]}...")


        prompt = f"""You are a cybersecurity expert specializing in EVSE and power grid systems.

User Query:
{query}

Answer the query based on your general knowledge. Provide specific information about vulnerabilities,
attack techniques, and security recommendations for EVSE and power systems.
"""


        try:
            start_time = time.time()
            response = self.model.generate_content(prompt)
            response_time = time.time() - start_time

            return {
                'response': response.text,
                'response_time': response_time,
                'prompt_length': len(prompt)
            }
        except Exception as e:
            logger.error(f"Non-RAG response failed: {e}")
            return {
                'response': f"Error: {e}",
                'response_time': 0,
                'prompt_length': 0
            }

    def evaluate_query(self, query: str, n_context_docs: int = 10,
                      ground_truth: Optional[Dict] = None) -> Dict[str, any]:

        logger.info(f"Evaluating query: {query}")


        rag_result = self.get_rag_response(query, n_context_docs)
        non_rag_result = self.get_non_rag_response(query)


        comparison = EvaluationMetrics.compare_responses(
            rag_result['response'],
            non_rag_result['response'],
            ground_truth
        )


        report = EvaluationMetrics.generate_evaluation_report(comparison)

        evaluation = {
            'query': query,
            'rag_response': rag_result,
            'non_rag_response': non_rag_result,
            'comparison': comparison,
            'report': report,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        return evaluation

    def evaluate_test_set(self, test_queries: List[Dict],
                         output_file: Optional[str] = None) -> Dict[str, any]:

        logger.info(f"Evaluating {len(test_queries)} test queries")

        results = []

        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"Evaluating query {i}/{len(test_queries)}")

            query = test_case['query']
            ground_truth = test_case.get('ground_truth', None)
            n_context = test_case.get('n_context_docs', 10)

            evaluation = self.evaluate_query(query, n_context, ground_truth)
            results.append(evaluation)


            print(f"\n{'='*80}")
            print(f"Query {i}/{len(test_queries)}")
            print(f"{'='*80}")
            print(evaluation['report'])


            time.sleep(2)


        aggregated = self._aggregate_results(results)


        if output_file:
            self._save_results(results, aggregated, output_file)
            logger.info(f"Results saved to {output_file}")

        return aggregated

    def _aggregate_results(self, results: List[Dict]) -> Dict[str, any]:

        total_queries = len(results)


        mitre_advantages = sum(1 for r in results if r['comparison']['mitre_techniques']['rag_advantage'] > 0)
        stride_advantages = sum(1 for r in results if r['comparison']['stride_categories']['rag_advantage'] > 0)
        cve_advantages = sum(1 for r in results if r['comparison']['cve_references']['rag_advantage'] > 0)
        specificity_advantages = sum(1 for r in results if r['comparison']['specificity']['rag_advantage'] > 0)
        actionability_advantages = sum(1 for r in results if r['comparison']['actionability']['rag_advantage'] > 0)


        avg_rag_mitre = sum(r['comparison']['mitre_techniques']['rag_count'] for r in results) / total_queries
        avg_non_rag_mitre = sum(r['comparison']['mitre_techniques']['non_rag_count'] for r in results) / total_queries

        avg_rag_stride = sum(r['comparison']['stride_categories']['rag_count'] for r in results) / total_queries
        avg_non_rag_stride = sum(r['comparison']['stride_categories']['non_rag_count'] for r in results) / total_queries

        avg_rag_specificity = sum(r['comparison']['specificity']['rag']['total_specificity_score'] for r in results) / total_queries
        avg_non_rag_specificity = sum(r['comparison']['specificity']['non_rag']['total_specificity_score'] for r in results) / total_queries


        gt_results = [r for r in results if 'ground_truth_validation' in r['comparison']]

        avg_mitre_recall_rag = 0
        avg_mitre_recall_non_rag = 0
        avg_stride_recall_rag = 0
        avg_stride_recall_non_rag = 0

        if gt_results:
            mitre_gt = [r for r in gt_results if 'mitre_accuracy' in r['comparison']['ground_truth_validation']]
            if mitre_gt:
                avg_mitre_recall_rag = sum(r['comparison']['ground_truth_validation']['mitre_accuracy']['rag_recall'] for r in mitre_gt) / len(mitre_gt)
                avg_mitre_recall_non_rag = sum(r['comparison']['ground_truth_validation']['mitre_accuracy']['non_rag_recall'] for r in mitre_gt) / len(mitre_gt)

            stride_gt = [r for r in gt_results if 'stride_accuracy' in r['comparison']['ground_truth_validation']]
            if stride_gt:
                avg_stride_recall_rag = sum(r['comparison']['ground_truth_validation']['stride_accuracy']['rag_recall'] for r in stride_gt) / len(stride_gt)
                avg_stride_recall_non_rag = sum(r['comparison']['ground_truth_validation']['stride_accuracy']['non_rag_recall'] for r in stride_gt) / len(stride_gt)

        aggregated = {
            'total_queries': total_queries,
            'rag_advantages': {
                'mitre_techniques': f"{mitre_advantages}/{total_queries} ({mitre_advantages/total_queries*100:.1f}%)",
                'stride_categories': f"{stride_advantages}/{total_queries} ({stride_advantages/total_queries*100:.1f}%)",
                'cve_references': f"{cve_advantages}/{total_queries} ({cve_advantages/total_queries*100:.1f}%)",
                'specificity': f"{specificity_advantages}/{total_queries} ({specificity_advantages/total_queries*100:.1f}%)",
                'actionability': f"{actionability_advantages}/{total_queries} ({actionability_advantages/total_queries*100:.1f}%)"
            },
            'average_scores': {
                'mitre_techniques': {
                    'rag': avg_rag_mitre,
                    'non_rag': avg_non_rag_mitre,
                    'improvement': avg_rag_mitre - avg_non_rag_mitre
                },
                'stride_categories': {
                    'rag': avg_rag_stride,
                    'non_rag': avg_non_rag_stride,
                    'improvement': avg_rag_stride - avg_non_rag_stride
                },
                'specificity_score': {
                    'rag': avg_rag_specificity,
                    'non_rag': avg_non_rag_specificity,
                    'improvement': avg_rag_specificity - avg_non_rag_specificity
                }
            }
        }

        if gt_results:
            aggregated['ground_truth_accuracy'] = {
                'mitre_recall': {
                    'rag': avg_mitre_recall_rag,
                    'non_rag': avg_mitre_recall_non_rag,
                    'improvement': avg_mitre_recall_rag - avg_mitre_recall_non_rag
                },
                'stride_recall': {
                    'rag': avg_stride_recall_rag,
                    'non_rag': avg_stride_recall_non_rag,
                    'improvement': avg_stride_recall_rag - avg_stride_recall_non_rag
                }
            }

        return aggregated

    def _format_context(self, context_docs: List[Dict]) -> str:

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
- STRIDE: {metadata.get('stride_categories', '[]')}
- MITRE: {metadata.get('mitre_techniques', '[]')}
- Content: {content[:400]}...
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _save_results(self, results: List[Dict], aggregated: Dict, output_file: str):

        output = {
            'aggregated_results': aggregated,
            'individual_results': results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
