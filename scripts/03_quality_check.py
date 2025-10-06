#!/usr/bin/env python3
"""
Quality check script to verify that no original terms leaked into the knowledge base.
Scans all markdown files and checks for occurrences of original terms.
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import defaultdict


def load_original_terms(terms_file):
    """Load original terms from the mapping file."""
    with open(terms_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get original terms (keys in mapping)
    mapping = {k: v for k, v in data.get('mapping', {}).items() 
               if not k.startswith('_')}
    
    original_terms = list(mapping.keys())
    return original_terms, data


def scan_file_for_terms(filepath, terms):
    """
    Scan a file for any occurrences of original terms.
    
    Returns:
        dict: {term: [line_numbers]} for each found term
    """
    findings = defaultdict(list)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, start=1):
            for term in terms:
                # Case-insensitive search with word boundaries
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, line, re.IGNORECASE):
                    findings[term].append(line_num)
        
    except Exception as e:
        print(f"⚠️  Warning: Could not scan {filepath.name}: {e}")
    
    return findings


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    terms_file = project_root / "terms_map.json"
    kb_dir = project_root / "knowledge_base"
    
    # Check if terms map exists
    if not terms_file.exists():
        print(f"❌ Error: {terms_file} not found!")
        sys.exit(1)
    
    # Check if knowledge base exists
    if not kb_dir.exists() or not list(kb_dir.glob('*.md')):
        print(f"❌ Error: No markdown files found in {kb_dir}")
        print("Please run 02_apply_replacements.py first")
        sys.exit(1)
    
    # Load original terms
    print("🔍 Loading original terms to check...")
    original_terms, metadata = load_original_terms(terms_file)
    print(f"   Universe: {metadata.get('universe', 'Unknown')}")
    print(f"   Checking for {len(original_terms)} original terms")
    print()
    
    if len(original_terms) == 0:
        print("⚠️  Warning: No terms to check (all keys start with '_')")
        sys.exit(0)
    
    # Get all markdown files
    md_files = list(kb_dir.glob('*.md'))
    print(f"📄 Scanning {len(md_files)} markdown files...")
    print()
    
    # Scan all files
    all_findings = defaultdict(lambda: defaultdict(list))
    total_leaks = 0
    
    for md_file in md_files:
        findings = scan_file_for_terms(md_file, original_terms)
        
        if findings:
            for term, line_numbers in findings.items():
                all_findings[term][md_file.name] = line_numbers
                total_leaks += len(line_numbers)
    
    # Report results
    print("=" * 80)
    
    if not all_findings:
        print("✅ SUCCESS: No original terms detected!")
        print()
        print("Your knowledge base is clean and ready for RAG testing.")
        print(f"📁 Location: {kb_dir}")
        print(f"📊 Files: {len(md_files)} markdown documents")
        sys.exit(0)
    
    else:
        print("❌ LEAK DETECTED: Original terms found in knowledge base!")
        print()
        print(f"Found {len(all_findings)} original term(s) in {len(md_files)} file(s)")
        print(f"Total occurrences: {total_leaks}")
        print()
        print("Details:")
        print("-" * 80)
        
        for term, files in sorted(all_findings.items()):
            print(f"\n🔴 Term: '{term}'")
            for filename, line_numbers in sorted(files.items()):
                lines_str = ", ".join(map(str, line_numbers[:10]))
                if len(line_numbers) > 10:
                    lines_str += f" (and {len(line_numbers) - 10} more)"
                print(f"   📄 {filename}: lines {lines_str}")
        
        print()
        print("-" * 80)
        print("⚠️  Action required:")
        print("   1. Check if these terms need to be added to terms_map.json")
        print("   2. Run 02_apply_replacements.py again")
        print("   3. Re-run this quality check")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
