#!/usr/bin/env python3
"""
Script to build vector index from the knowledge base.
Creates embeddings and FAISS index for semantic search.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import faiss


# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, efficient, good quality
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks
FAISS_INDEX_PATH = "faiss.index"
CHUNKS_METADATA_PATH = "chunks_metadata.json"


def load_model():
    """Load the embedding model."""
    print(f"📥 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"   ✅ Model loaded successfully")
    print(f"   📊 Embedding dimension: {embedding_dim}")
    return model, embedding_dim


def load_documents(kb_dir: Path) -> List[Dict[str, str]]:
    """Load all markdown documents from knowledge base."""
    documents = []
    md_files = list(kb_dir.glob("*.md"))
    
    if not md_files:
        print(f"❌ No markdown files found in {kb_dir}")
        return []
    
    print(f"📚 Loading {len(md_files)} documents...")
    
    for path in md_files:
        try:
            text = path.read_text(encoding='utf-8')
            documents.append({
                'filename': path.name,
                'filepath': str(path),
                'text': text
            })
        except Exception as e:
            print(f"⚠️  Warning: Could not read {path.name}: {e}")
    
    print(f"   ✅ Loaded {len(documents)} documents")
    return documents


def create_chunks(documents: List[Dict[str, str]]) -> List[Dict[str, any]]:
    """Split documents into chunks with metadata."""
    print(f"\n🔪 Splitting documents into chunks...")
    print(f"   Chunk size: {CHUNK_SIZE} characters")
    print(f"   Overlap: {CHUNK_OVERLAP} characters")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
    )
    
    all_chunks = []
    
    for doc in tqdm(documents, desc="Chunking documents"):
        chunks = splitter.split_text(doc['text'])
        
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                'id': f"{doc['filename'].replace('.md', '')}_{i}",
                'text': chunk_text,
                'source_file': doc['filename'],
                'source_path': doc['filepath'],
                'chunk_index': i,
                'chunk_length': len(chunk_text)
            })
    
    print(f"   ✅ Created {len(all_chunks)} chunks")
    
    # Statistics
    avg_length = np.mean([c['chunk_length'] for c in all_chunks])
    print(f"   📊 Average chunk length: {avg_length:.0f} characters")
    
    return all_chunks


def generate_embeddings(chunks: List[Dict], model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for all chunks."""
    print(f"\n🧮 Generating embeddings for {len(chunks)} chunks...")
    
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings with progress bar
    batch_size = 32
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings).astype('float32')
    
    print(f"   ✅ Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray, embedding_dim: int) -> faiss.Index:
    """Build FAISS index from embeddings."""
    print(f"\n🏗️  Building FAISS index...")
    
    # Use IndexFlatL2 for exact search (good for small datasets)
    # For larger datasets, consider IndexIVFFlat or IndexHNSW
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Add vectors to index
    index.add(embeddings)
    
    print(f"   ✅ Index built successfully")
    print(f"   📊 Total vectors in index: {index.ntotal}")
    
    return index


def save_index_and_metadata(index: faiss.Index, chunks: List[Dict], 
                            output_dir: Path):
    """Save FAISS index and chunk metadata."""
    print(f"\n💾 Saving index and metadata...")
    
    # Save FAISS index
    index_path = output_dir / FAISS_INDEX_PATH
    faiss.write_index(index, str(index_path))
    print(f"   ✅ FAISS index saved: {index_path}")
    
    # Save metadata (without text to save space)
    metadata_path = output_dir / CHUNKS_METADATA_PATH
    metadata = []
    for chunk in chunks:
        metadata.append({
            'id': chunk['id'],
            'source_file': chunk['source_file'],
            'source_path': chunk['source_path'],
            'chunk_index': chunk['chunk_index'],
            'chunk_length': chunk['chunk_length'],
            'text': chunk['text']  # Include text for retrieval
        })
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Metadata saved: {metadata_path}")
    
    # Save index info
    info = {
        'model_name': MODEL_NAME,
        'embedding_dimension': index.d,
        'total_chunks': len(chunks),
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'index_type': 'IndexFlatL2',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    info_path = output_dir / "index_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    print(f"   ✅ Index info saved: {info_path}")


def test_search(index: faiss.Index, chunks: List[Dict], 
                model: SentenceTransformer, k: int = 3):
    """Test the search functionality with sample queries."""
    print(f"\n🔍 Testing search functionality...\n")
    
    test_queries = [
        "Who is Xarn Velgor?",
        "What is Synth Flux?",
        "Tell me about Aridian Prime",
        "What happened in the Replicant Conflict?"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 80)
        
        # Encode query
        query_vector = model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search
        distances, indices = index.search(query_vector, k)
        
        # Display results
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
            chunk = chunks[idx]
            preview = chunk['text'][:200].replace('\n', ' ')
            print(f"{rank}. [{chunk['source_file']}] (distance: {distance:.4f})")
            print(f"   {preview}...")
            print()
        
        print()


def main():
    start_time = time.time()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    kb_dir = project_root / "knowledge_base"
    output_dir = project_root
    
    print("=" * 80)
    print("🚀 BUILDING VECTOR INDEX FOR KNOWLEDGE BASE")
    print("=" * 80)
    print()
    
    # Check if knowledge base exists
    if not kb_dir.exists():
        print(f"❌ Error: Knowledge base directory not found: {kb_dir}")
        print("   Please run scripts 01-03 first to create the knowledge base")
        sys.exit(1)
    
    # Step 1: Load model
    model, embedding_dim = load_model()
    
    # Step 2: Load documents
    documents = load_documents(kb_dir)
    if not documents:
        sys.exit(1)
    
    # Step 3: Create chunks
    chunks = create_chunks(documents)
    
    # Step 4: Generate embeddings
    embeddings = generate_embeddings(chunks, model)
    
    # Step 5: Build FAISS index
    index = build_faiss_index(embeddings, embedding_dim)
    
    # Step 6: Save everything
    save_index_and_metadata(index, chunks, output_dir)
    
    # Step 7: Test search
    test_search(index, chunks, model)
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 80)
    print("✅ INDEX BUILD COMPLETE!")
    print("=" * 80)
    print(f"📊 Statistics:")
    print(f"   - Documents processed: {len(documents)}")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Embedding dimension: {embedding_dim}")
    print(f"   - Time elapsed: {elapsed:.2f}s")
    print()
    print(f"📁 Output files:")
    print(f"   - {output_dir / FAISS_INDEX_PATH}")
    print(f"   - {output_dir / CHUNKS_METADATA_PATH}")
    print(f"   - {output_dir / 'index_info.json'}")
    print()
    print("🎯 Next steps:")
    print("   1. Use scripts/05_test_search.py to test queries")
    print("   2. Integrate with your RAG application")


if __name__ == "__main__":
    main()
