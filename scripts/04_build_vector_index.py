#!/usr/bin/env python3
"""
Script to build a vector index from the knowledge base.
Creates embeddings using sentence-transformers and stores them in FAISS.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

# FAISS for vector indexing
import faiss

# Text splitter for chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, efficient, good quality
# Alternatives:
# - "all-mpnet-base-v2" (better quality, slower)
# - "paraphrase-multilingual-MiniLM-L12-v2" (multilingual support)
# - "sentence-transformers/all-MiniLM-L12-v2"

CHUNK_SIZE = 1000  # Characters
CHUNK_OVERLAP = 200  # Overlap to maintain context


def load_model():
    """Load the embedding model."""
    print(f"📥 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"   Embedding dimension: {embedding_dim}")
    return model, embedding_dim


def load_documents(kb_dir):
    """Load all markdown documents from knowledge base."""
    kb_path = Path(kb_dir)
    
    if not kb_path.exists():
        print(f"❌ Error: Knowledge base directory not found: {kb_dir}")
        sys.exit(1)
    
    md_files = list(kb_path.glob("*.md"))
    
    if not md_files:
        print(f"❌ Error: No markdown files found in {kb_dir}")
        sys.exit(1)
    
    print(f"📚 Loading {len(md_files)} documents from {kb_dir}")
    
    documents = []
    for path in tqdm(md_files, desc="Loading documents"):
        try:
            text = path.read_text(encoding='utf-8')
            documents.append({
                'path': str(path),
                'filename': path.stem,
                'text': text,
                'size': len(text)
            })
        except Exception as e:
            print(f"⚠️  Warning: Could not load {path.name}: {e}")
    
    total_chars = sum(doc['size'] for doc in documents)
    print(f"   Total documents: {len(documents)}")
    print(f"   Total characters: {total_chars:,}")
    print()
    
    return documents


def create_chunks(documents):
    """Split documents into chunks for embedding."""
    print("✂️  Splitting documents into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )
    
    all_chunks = []
    
    for doc in tqdm(documents, desc="Chunking documents"):
        # Split text
        text_chunks = splitter.split_text(doc['text'])
        
        # Create chunk metadata
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                'id': f"{doc['filename']}_{i}",
                'chunk_index': i,
                'text': chunk_text,
                'source_file': doc['filename'],
                'source_path': doc['path'],
                'char_count': len(chunk_text)
            }
            all_chunks.append(chunk)
    
    print(f"   Total chunks created: {len(all_chunks)}")
    avg_chunk_size = np.mean([c['char_count'] for c in all_chunks])
    print(f"   Average chunk size: {avg_chunk_size:.0f} characters")
    print()
    
    return all_chunks


def generate_embeddings(chunks, model):
    """Generate embeddings for all chunks."""
    print("🔢 Generating embeddings...")
    
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings with progress bar
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    print(f"   Embeddings shape: {embeddings.shape}")
    print()
    
    return embeddings


def build_faiss_index(embeddings, embedding_dim):
    """Build FAISS index from embeddings."""
    print("🏗️  Building FAISS index...")
    
    # Convert to float32 (FAISS requirement)
    embeddings_np = np.array(embeddings).astype('float32')
    
    # Create index (L2 distance)
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Add vectors
    index.add(embeddings_np)
    
    print(f"   Index type: Flat (exact search)")
    print(f"   Total vectors: {index.ntotal}")
    print()
    
    return index


def save_index(index, chunks, output_dir):
    """Save FAISS index and metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving index to {output_dir}/")
    
    # Save FAISS index
    index_file = output_path / "faiss.index"
    faiss.write_index(index, str(index_file))
    print(f"   ✅ FAISS index: {index_file.name}")
    
    # Save chunk metadata
    metadata_file = output_path / "chunks_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"   ✅ Metadata: {metadata_file.name}")
    
    # Save configuration
    config = {
        'model_name': MODEL_NAME,
        'embedding_dim': index.d,
        'num_vectors': index.ntotal,
        'num_chunks': len(chunks),
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'created_at': datetime.now().isoformat(),
        'index_type': 'FlatL2'
    }
    
    config_file = output_path / "index_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Config: {config_file.name}")
    
    print()
    return config


def test_search(index, chunks, model, num_queries=3):
    """Test the index with sample queries."""
    print("🔍 Testing index with sample queries...")
    print()
    
    # Sample queries related to our transformed universe
    test_queries = [
        "Who is Xarn Velgor?",
        "What is Synth Flux?",
        "Tell me about Aridian Prime",
        "What happened in the Replicant Conflict?",
        "Who are the Wardens?"
    ]
    
    for query in test_queries[:num_queries]:
        print(f"Query: '{query}'")
        print("-" * 80)
        
        # Encode query
        query_vector = model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search
        k = 3  # Top 3 results
        distances, indices = index.search(query_vector, k)
        
        # Display results
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            chunk = chunks[idx]
            preview = chunk['text'][:200].replace('\n', ' ')
            print(f"  {rank}. [{chunk['source_file']}] (distance: {dist:.4f})")
            print(f"     {preview}...")
            print()
        
        print()


def main():
    start_time = datetime.now()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    kb_dir = project_root / "knowledge_base"
    output_dir = project_root / "vector_index"
    
    print("=" * 80)
    print("VECTOR INDEX BUILDER")
    print("=" * 80)
    print()
    
    # Step 1: Load model
    model, embedding_dim = load_model()
    print()
    
    # Step 2: Load documents
    documents = load_documents(kb_dir)
    
    # Step 3: Create chunks
    chunks = create_chunks(documents)
    
    # Step 4: Generate embeddings
    embeddings = generate_embeddings(chunks, model)
    
    # Step 5: Build FAISS index
    index = build_faiss_index(embeddings, embedding_dim)
    
    # Step 6: Save everything
    config = save_index(index, chunks, output_dir)
    
    # Step 7: Test search
    test_search(index, chunks, model)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("=" * 80)
    print("✅ INDEX BUILDING COMPLETE")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Documents processed: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Index size: {index.ntotal} vectors")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print()
    print(f"📁 Index location: {output_dir}/")
    print("📋 Next step: Use scripts/05_search_index.py to query the index")
    print()


if __name__ == "__main__":
    main()
