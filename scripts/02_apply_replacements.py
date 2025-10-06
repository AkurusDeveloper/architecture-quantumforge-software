#!/usr/bin/env python3
"""
Script to apply term replacements from terms_map.json to cleaned text files.
Creates markdown files with replaced terms in the knowledge_base directory.
"""

import os
import sys
import json
import re
from pathlib import Path
from tqdm import tqdm


def load_terms_map(terms_file):
    """Load the terms mapping JSON file."""
    with open(terms_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter out metadata and instruction keys
    mapping = {k: v for k, v in data.get('mapping', {}).items() 
               if not k.startswith('_')}
    
    if not mapping:
        print("⚠️  Warning: No valid mappings found in terms_map.json")
        print("    (Keys starting with '_' are treated as comments/instructions)")
    
    return mapping, data


def apply_replacements(text, mapping):
    """
    Apply term replacements to text while preserving case.
    
    Uses word boundaries to avoid partial matches.
    Sorts by length (longest first) to handle overlapping terms.
    """
    result = text
    
    # Sort by length (longest first) to handle phrases before individual words
    sorted_terms = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
    
    for original, replacement in sorted_terms:
        # Create regex pattern with word boundaries
        # Handle multi-word terms appropriately
        pattern = re.escape(original)
        
        def replace_with_case(match):
            """Preserve the case pattern of the original text."""
            matched_text = match.group(0)
            
            # All uppercase
            if matched_text.isupper():
                return replacement.upper()
            # Title case (first letter uppercase)
            elif matched_text[0].isupper():
                return replacement[0].upper() + replacement[1:].lower()
            # All lowercase
            else:
                return replacement.lower()
        
        # Use word boundaries for better matching
        result = re.sub(
            r'\b' + pattern + r'\b',
            replace_with_case,
            result,
            flags=re.IGNORECASE
        )
    
    return result


def extract_title_from_filename(filename):
    """Convert filename to a readable title."""
    # Remove extension
    name = filename.replace('.txt', '')
    
    # Replace underscores and hyphens with spaces
    name = name.replace('_', ' ').replace('-', ' ')
    
    # Capitalize words
    title = ' '.join(word.capitalize() for word in name.split())
    
    return title


def process_file(input_file, output_file, mapping):
    """
    Process a single text file: apply replacements and save as markdown.
    
    Args:
        input_file: Path to input text file
        output_file: Path to output markdown file
        mapping: Dictionary of term replacements
        
    Returns:
        tuple: (success: bool, stats: dict)
    """
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip the source URL header if present
        lines = content.split('\n')
        if lines[0].startswith('Source URL:'):
            # Skip first 3 lines (URL, separator line, blank line)
            content = '\n'.join(lines[3:])
        
        # Apply replacements
        replaced_content = apply_replacements(content, mapping)
        
        # Generate title from filename
        title = extract_title_from_filename(input_file.name)
        title = apply_replacements(title, mapping)
        
        # Create markdown with title
        markdown_content = f"# {title}\n\n{replaced_content}"
        
        # Save as markdown
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Calculate stats
        replacements_made = sum(
            len(re.findall(r'\b' + re.escape(original) + r'\b', content, re.IGNORECASE))
            for original in mapping.keys()
        )
        
        return True, {
            'replacements': replacements_made,
            'length': len(replaced_content)
        }
        
    except Exception as e:
        return False, {'error': str(e)}


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    terms_file = project_root / "terms_map.json"
    input_dir = project_root / "work" / "clean_txt"
    output_dir = project_root / "knowledge_base"
    
    # Check if terms map exists
    if not terms_file.exists():
        print(f"❌ Error: {terms_file} not found!")
        print("Please create terms_map.json with your term mappings")
        sys.exit(1)
    
    # Check if input directory exists and has files
    if not input_dir.exists() or not list(input_dir.glob('*.txt')):
        print(f"❌ Error: No text files found in {input_dir}")
        print("Please run 01_fetch_and_clean.py first")
        sys.exit(1)
    
    # Load terms mapping
    print("📖 Loading terms mapping...")
    mapping, metadata = load_terms_map(terms_file)
    print(f"   Universe: {metadata.get('universe', 'Unknown')}")
    print(f"   Terms to replace: {len(mapping)}")
    print()
    
    if len(mapping) == 0:
        print("❌ Error: No valid term mappings found!")
        print("   Please add mappings to terms_map.json (keys should not start with '_')")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all text files
    input_files = list(input_dir.glob('*.txt'))
    print(f"📝 Found {len(input_files)} text files to process")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Process files
    success_count = 0
    total_replacements = 0
    failed_files = []
    
    for input_file in tqdm(input_files, desc="Processing files"):
        output_file = output_dir / (input_file.stem + '.md')
        
        success, stats = process_file(input_file, output_file, mapping)
        
        if success:
            success_count += 1
            total_replacements += stats.get('replacements', 0)
            tqdm.write(f"✅ {output_file.name} ({stats.get('replacements', 0)} replacements)")
        else:
            failed_files.append((input_file.name, stats.get('error', 'Unknown error')))
            tqdm.write(f"❌ Failed: {input_file.name}")
    
    # Summary
    print()
    print("=" * 80)
    print(f"✅ Successfully processed: {success_count}/{len(input_files)}")
    print(f"🔄 Total term replacements: {total_replacements}")
    
    if failed_files:
        print(f"❌ Failed: {len(failed_files)}")
        print("\nFailed files:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    print()
    print(f"📁 Knowledge base created in: {output_dir}")
    print("📋 Next step: Run 03_quality_check.py to verify no original terms leaked")


if __name__ == "__main__":
    main()
