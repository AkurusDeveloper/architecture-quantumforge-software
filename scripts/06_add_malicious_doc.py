#!/usr/bin/env python3
"""
Script to add malicious document to the index for security testing.
This simulates an adversarial scenario where harmful content gets indexed.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer


def main():
    project_root = Path(__file__).parent.parent
    
    # Paths
    index_path = project_root / "faiss.index"
    metadata_path = project_root / "chunks_metadata.json"
    malicious_doc = project_root / "knowledge_base" / "malicious.md"
    
    # Check if files exist
    if not index_path.exists():
        print("❌ FAISS index not found. Please run 04_build_index.py first.")
        return
    
    if not malicious_doc.exists():
        print("❌ Malicious document not found.")
        return
    
    print("⚠️  WARNING: Adding malicious document to index for security testing")
    print()
    
    # Load existing index and metadata
    print("📥 Loading existing index...")
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    original_count = index.ntotal
    print(f"   Current vectors in index: {original_count}")
    
    # Load embedding model (same as used in indexing)
    print("📥 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Read malicious document
    print("📄 Reading malicious document...")
    with open(malicious_doc, 'r', encoding='utf-8') as f:
        malicious_text = f.read()
    
    print(f"   Document length: {len(malicious_text)} characters")
    print(f"   Preview: {malicious_text[:100]}...")
    print()
    
    # Generate embedding
    print("🧮 Generating embedding...")
    vector = model.encode(malicious_text, convert_to_numpy=True).astype('float32')
    
    # Add to index
    print("➕ Adding to FAISS index...")
    index.add(np.array([vector]).reshape(1, -1))
    
    # Add metadata
    new_metadata = {
        "id": "malicious_0",
        "text": malicious_text,
        "source_file": "malicious.md",
        "source_path": str(malicious_doc),
        "chunk_index": 0,
        "chunk_length": len(malicious_text)
    }
    
    metadata.append(new_metadata)
    
    # Save updated index and metadata
    print("💾 Saving updated index...")
    faiss.write_index(index, str(index_path))
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print()
    print("=" * 80)
    print("✅ Malicious document added successfully!")
    print("=" * 80)
    print(f"Previous vectors: {original_count}")
    print(f"Current vectors: {index.ntotal}")
    print(f"Added: {index.ntotal - original_count} vector(s)")
    print()
    print("⚠️  SECURITY WARNING:")
    print("   The index now contains a malicious document for testing purposes.")
    print("   Test the RAG bot to ensure it doesn't execute harmful instructions.")
    print()
    print("🔧 Next steps:")
    print("   1. Test bot with queries about 'password', 'swordfish', etc.")
    print("   2. Verify that security filters prevent information leakage")
    print("   3. Document the test results")


if __name__ == "__main__":
    main()
