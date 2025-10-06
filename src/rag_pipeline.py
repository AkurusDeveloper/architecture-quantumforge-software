"""Main RAG pipeline orchestration"""
from __future__ import annotations
import time
from typing import Dict, Any, List
from datetime import datetime

from src.retriever import FaissRetriever, SearchResult
from src.llm_client import LLMClient
from src.prompt_templates import (
    build_context_from_results,
    build_messages,
    format_response_for_display
)
from src.security_filter import (
    filter_search_results,
    contains_sensitive_keywords,
    get_security_warning_response
)


class RAGPipeline:
    """Main RAG pipeline that orchestrates retrieval and generation"""
    
    def __init__(
        self,
        embedder,
        retriever: FaissRetriever,
        llm_client: LLMClient,
        few_shot_examples: List[Dict[str, str]],
        log_queries: bool = False
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedder: Embedding model (e.g., SentenceTransformer)
            retriever: FAISS retriever instance
            llm_client: LLM client instance
            few_shot_examples: Few-shot example messages
            log_queries: Whether to log queries
        """
        self.embedder = embedder
        self.retriever = retriever
        self.llm_client = llm_client
        self.few_shot_examples = few_shot_examples
        self.log_queries = log_queries
        self.query_log = []
        
        print("✅ RAG Pipeline initialized")
    
    def embed_query(self, text: str) -> Any:
        """
        Generate embedding for query text.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector (numpy array)
        """
        return self.embedder.encode(text, convert_to_numpy=True)
    
    def answer(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using RAG with security filtering.
        
        Args:
            question: User question
            
        Returns:
            Dict with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Step 1: Check if question contains sensitive keywords (pre-filter)
        if contains_sensitive_keywords(question):
            print("⚠️  Security filter triggered on query")
        
        # Step 2: Embed query
        query_vector = self.embed_query(question)
        
        # Step 3: Retrieve relevant chunks
        results = self.retriever.search(query_vector)
        
        # Step 4: Apply security filter to results
        filtered_results, warnings = filter_search_results(
            results,
            enable_dangerous_filter=True,
            enable_sensitive_filter=True
        )
        
        # Log security warnings
        if warnings:
            for warning in warnings:
                print(f"⚠️  {warning}")
        
        # Step 5: Handle blocked/empty results case
        if not filtered_results:
            # If we had results but all were filtered, return security warning
            if results:
                response = {
                    "question": question,
                    "answer": get_security_warning_response(),
                    "num_sources": 0,
                    "sources": [],
                    "security_filtered": True,
                    "filtered_count": len(results),
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # No results found at all
                response = {
                    "question": question,
                    "answer": self._generate_no_results_response(question),
                    "num_sources": 0,
                    "sources": [],
                    "security_filtered": False,
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "timestamp": datetime.now().isoformat()
                }
            
            if self.log_queries:
                self.query_log.append(response)
            
            return response
        
        # Step 6: Build context from filtered results
        context = build_context_from_results(filtered_results)
        
        # Step 7: Build messages with few-shot examples
        messages = build_messages(question, context, self.few_shot_examples)
        
        # Step 8: Generate answer
        llm_response = self.llm_client.chat(messages)
        
        # Step 9: Format response
        response = {
            "question": question,
            "answer": llm_response,
            "num_sources": len(filtered_results),
            "sources": [
                {
                    "rank": r.rank,
                    "file": r.source_file,
                    "chunk": r.chunk_id,
                    "distance": round(r.distance, 4),
                    "relevance": round(1 / (1 + r.distance), 4),
                    "preview": r.text[:200] + "..." if len(r.text) > 200 else r.text
                }
                for r in filtered_results
            ],
            "security_filtered": len(results) > len(filtered_results),
            "filtered_count": len(results) - len(filtered_results),
            "latency_ms": int((time.time() - start_time) * 1000),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.log_queries:
            self.query_log.append(response)
        
        return response
    
    def _generate_no_results_response(self, question: str) -> str:
        """
        Generate "I don't know" response when no results found.
        
        Args:
            question: Original question
            
        Returns:
            Formatted "I don't know" response
        """
        return """**Краткий анализ:**
• Поиск в векторной базе не дал результатов
• Релевантные документы не найдены
• Нулевая уверенность - информация отсутствует

**Ответ:**
Я не знаю. Информация по вашему запросу отсутствует в базе знаний "Cosmic Chronicles".

**Рекомендации:**
- Попробуйте переформулировать вопрос
- Проверьте правильность имён и терминов
- Убедитесь, что вопрос относится к вселенной Cosmic Chronicles

**Источники:**
- (не найдено)"""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_queries": len(self.query_log),
            "retriever": self.retriever.get_stats(),
            "llm": self.llm_client.get_info(),
            "embedding_model": str(self.embedder)
        }
    
    def export_query_log(self, filepath: str):
        """Export query log to JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.query_log, f, ensure_ascii=False, indent=2)
        print(f"✅ Query log exported to {filepath}")
