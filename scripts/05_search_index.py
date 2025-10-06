#!/usr/bin/env python3
"""
Interactive search tool for querying the vector index.
Allows semantic search through the knowledge base.
"""

import sys
import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_index(index_dir):
    """Load FAISS index, metadata, and config."""
    index_path = Path(index_dir)
    
    if not index_path.exists():
        print(f"❌ Error: Index directory not found: {index_dir}")
        print("   Run scripts/04_build_vector_index.py first")
        sys.exit(1)
    
    # Load config
    config_file = index_path / "index_config.json"
    if not config_file.exists():
        print(f"❌ Error: Config file not found: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Load FAISS index
    index_file = index_path / "faiss.index"
    if not index_file.exists():
        print(f"❌ Error: FAISS index not found: {index_file}")
        sys.exit(1)
    
    index = faiss.read_index(str(index_file))
    
    # Load metadata
    metadata_file = index_path / "chunks_metadata.pkl"
    if not metadata_file.exists():
        print(f"❌ Error: Metadata file not found: {metadata_file}")
        sys.exit(1)
    
    with open(metadata_file, 'rb') as f:
        chunks = pickle.load(f)
    
    return index, chunks, config


def search(query, model, index, chunks, k=5):
    """
    Perform semantic search.
    
    Args:
        query: Search query string
        model: Sentence transformer model
        index: FAISS index
        chunks: List of chunk metadata
        k: Number of results to return
        
    Returns:
        List of (chunk, distance) tuples
    """
    # Encode query
    query_vector = model.encode([query], convert_to_numpy=True).astype('float32')
    
    # Search index
    distances, indices = index.search(query_vector, k)
    
    # Collect results
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append((chunks[idx], float(dist)))
    
    return results


def display_results(query, results, verbose=False):
    """Display search results in a formatted way."""
    print()
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    print()
    
    if not results:
        print("No results found.")
        return
    
    for rank, (chunk, distance) in enumerate(results, 1):
        print(f"[{rank}] Source: {chunk['source_file']} (chunk {chunk['chunk_index']})")
        print(f"    Distance: {distance:.4f}")
        print()
        
        if verbose:
            # Show full text
            print(chunk['text'])
        else:
            # Show preview
            preview = chunk['text'][:500]
            if len(chunk['text']) > 500:
                preview += "..."
            print(preview)
        
        print()
        print("-" * 80)
        print()


def interactive_mode(model, index, chunks):
    """Run interactive search mode."""
    print()
    print("=" * 80)
    print("INTERACTIVE SEARCH MODE")
    print("=" * 80)
    print()
    print("Enter your queries below. Type 'quit' or 'exit' to stop.")
    print("Commands:")
    print("  - 'verbose' : Toggle verbose mode (show full text)")
    print("  - 'k=N'     : Set number of results (default 5)")
    print()
    
    k = 5
    verbose = False
    
    while True:
        try:
            query = input("🔍 Query: ").strip()
            
            if not query:
                continue
            
            # Check for commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if query.lower() == 'verbose':
                verbose = not verbose
                print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
                continue
            
            if query.lower().startswith('k='):
                try:
                    k = int(query[2:])
                    print(f"Results count set to: {k}")
                    continue
                except ValueError:
                    print("Invalid k value. Use: k=5")
                    continue
            
            # Perform search
            results = search(query, model, index, chunks, k=k)
            display_results(query, results, verbose=verbose)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_mode(queries, model, index, chunks, k=5):
    """Run batch search mode with predefined queries."""
    for query in queries:
        results = search(query, model, index, chunks, k=k)
        display_results(query, results)


def main():
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Search the vector knowledge base"
    )
    parser.add_argument(
        'query',
        nargs='*',
        help='Search query (if not provided, enters interactive mode)'
    )
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show full chunk text instead of preview'
    )
    parser.add_argument(
        '--index-dir',
        type=str,
        default=None,
        help='Path to index directory (default: ../vector_index)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    if args.index_dir:
        index_dir = Path(args.index_dir)
    else:
        project_root = Path(__file__).parent.parent
        index_dir = project_root / "vector_index"
    
    print("📚 Loading vector index...")
    
    # Load index
    index, chunks, config = load_index(index_dir)
    
    print(f"   Model: {config['model_name']}")
    print(f"   Vectors: {config['num_vectors']}")
    print(f"   Created: {config['created_at']}")
    
    # Load model
    print(f"   Loading embedding model...")
    model = SentenceTransformer(config['model_name'])
    
    print("   ✅ Ready!")
    
    # Determine mode
    if args.query:
        # Single query or batch mode
        query_text = ' '.join(args.query)
        results = search(query_text, model, index, chunks, k=args.top_k)
        display_results(query_text, results, verbose=args.verbose)
    else:
        # Interactive mode
        interactive_mode(model, index, chunks)


if __name__ == "__main__":
    main()
