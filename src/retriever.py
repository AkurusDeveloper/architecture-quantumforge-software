"""Retriever module for FAISS-based semantic search"""
from __future__ import annotations
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Single search result"""
    rank: int
    distance: float
    text: str
    source_file: str
    chunk_id: str
    chunk_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "distance": self.distance,
            "text": self.text,
            "source_file": self.source_file,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index
        }


class FaissRetriever:
    """FAISS-based retriever for semantic search"""
    
    def __init__(
        self, 
        index_path: str, 
        metadata_path: str,
        top_k: int = 5,
        score_threshold: float = 1.2
    ):
        """
        Initialize FAISS retriever.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to chunks metadata JSON
            top_k: Number of top results to return
            score_threshold: Maximum L2 distance threshold
        """
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        # Load FAISS index
        index_file = Path(index_path)
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.index = faiss.read_index(str(index_file))
        print(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")
        
        # Load metadata
        metadata_file = Path(metadata_path)
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"✅ Loaded metadata: {len(self.metadata)} chunks")
        
        if len(self.metadata) != self.index.ntotal:
            print(f"⚠️  Warning: Metadata count ({len(self.metadata)}) "
                  f"!= index count ({self.index.ntotal})")
    
    def search(self, query_vector: np.ndarray) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query_vector: Query embedding vector
            
        Returns:
            List of SearchResult objects
        """
        # Ensure correct shape and type
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_vector, self.top_k)
        
        # Build results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            # Skip invalid indices
            if idx == -1 or idx >= len(self.metadata):
                continue
            
            # Skip results above threshold
            if distance > self.score_threshold:
                continue
            
            meta = self.metadata[idx]
            
            result = SearchResult(
                rank=rank,
                distance=float(distance),
                text=meta.get('text', ''),
                source_file=meta.get('source_file', 'unknown'),
                chunk_id=meta.get('id', f'chunk_{idx}'),
                chunk_index=meta.get('chunk_index', idx)
            )
            
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "total_vectors": self.index.ntotal,
            "total_metadata": len(self.metadata),
            "embedding_dim": self.index.d,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold
        }
