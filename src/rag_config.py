"""Configuration management for RAG Bot"""
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    provider: str = "local"
    model: str = "all-MiniLM-L6-v2"
    dim: int = 384


class IndexConfig(BaseModel):
    type: str = "faiss"
    index_path: str = "faiss.index"
    metadata_path: str = "chunks_metadata.json"
    top_k: int = 5
    score_threshold: float = 1.2


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 800
    temperature: float = 0.2


class AppConfig(BaseModel):
    few_shot_examples: str = "data/few_shot.json"
    log_queries: bool = True
    log_path: str = "logs/queries.log"


class RAGConfig(BaseModel):
    embedding: EmbeddingConfig
    index: IndexConfig
    llm: LLMConfig
    app: AppConfig


def load_config(config_path: str = "config.yaml") -> RAGConfig:
    """Load configuration from YAML file."""
    path = Path(config_path)
    
    if not path.exists():
        # Return default config
        return RAGConfig(
            embedding=EmbeddingConfig(),
            index=IndexConfig(),
            llm=LLMConfig(),
            app=AppConfig()
        )
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return RAGConfig(**data)
