#!/usr/bin/env python3
"""FastAPI interface for RAG Bot"""
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_config import load_config
from src.retriever import FaissRetriever
from src.llm_client import LLMClient
from src.rag_pipeline import RAGPipeline
from src.prompt_templates import load_few_shot_examples


# Load environment
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="RAG Bot API",
    description="Cosmic Chronicles Knowledge Base Assistant API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of sources to retrieve")


class Source(BaseModel):
    rank: int
    file: str
    chunk: str
    distance: float
    relevance: float
    preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    num_sources: int
    sources: list[Source]
    latency_ms: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_model: str
    llm_model: str
    index_size: int


# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None
config = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    global rag_pipeline, config
    
    print("🚀 Initializing RAG system...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize embedding model
        print(f"📥 Loading embedding model: {config.embedding.model}")
        embedder = SentenceTransformer(config.embedding.model)
        
        # Initialize retriever
        print(f"📥 Loading FAISS index...")
        retriever = FaissRetriever(
            index_path=config.index.index_path,
            metadata_path=config.index.metadata_path,
            top_k=config.index.top_k,
            score_threshold=config.index.score_threshold
        )
        
        # Initialize LLM client
        print(f"📥 Initializing LLM client: {config.llm.model}")
        llm_client = LLMClient(
            provider=config.llm.provider,
            model=config.llm.model,
            max_tokens=config.llm.max_tokens,
            temperature=config.llm.temperature
        )
        
        # Load few-shot examples
        few_shot = load_few_shot_examples(config.app.few_shot_examples)
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            embedder=embedder,
            retriever=retriever,
            llm_client=llm_client,
            few_shot_examples=few_shot,
            log_queries=False  # Don't log in API mode
        )
        
        print("✅ RAG system initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Bot API - Cosmic Chronicles",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ask": "/ask (POST)",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    stats = rag_pipeline.get_stats()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        embedding_model=config.embedding.model,
        llm_model=config.llm.model,
        index_size=stats["retriever"]["total_vectors"]
    )


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    Ask a question to the RAG bot.
    
    Args:
        request: Query request with question and optional top_k
        
    Returns:
        Query response with answer and sources
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Override top_k if provided
        if request.top_k is not None:
            original_k = rag_pipeline.retriever.top_k
            rag_pipeline.retriever.top_k = request.top_k
        
        # Get answer
        response = rag_pipeline.answer(request.question)
        
        # Restore original top_k
        if request.top_k is not None:
            rag_pipeline.retriever.top_k = original_k
        
        # Convert to response model
        return QueryResponse(
            question=response["question"],
            answer=response["answer"],
            num_sources=response["num_sources"],
            sources=[Source(**s) for s in response["sources"]],
            latency_ms=response["latency_ms"],
            timestamp=response["timestamp"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats")
async def stats():
    """Get system statistics."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return rag_pipeline.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
