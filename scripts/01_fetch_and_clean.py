#!/usr/bin/env python3
"""
Script to fetch web pages and extract clean text content.
Uses trafilatura for robust content extraction.
"""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse
import trafilatura
from tqdm import tqdm
import time


def sanitize_filename(url):
    """Convert URL to a safe filename."""
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    
    # Use the last part of the URL path, or domain if no path
    if path_parts and path_parts[-1]:
        filename = path_parts[-1]
    else:
        filename = parsed.netloc.replace('.', '_')
    
    # Remove query parameters and clean up
    filename = filename.split('?')[0].split('#')[0]
    
    # Ensure it's a valid filename
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
    
    return filename[:200]  # Limit length


def fetch_and_clean(url, output_dir):
    """
    Fetch a URL and extract clean text using trafilatura.
    
    Args:
        url: URL to fetch
        output_dir: Directory to save cleaned text
        
    Returns:
        tuple: (success: bool, filename: str or None, error: str or None)
    """
    try:
        # Download the page
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            return False, None, "Failed to download page"
        
        # Extract main text content
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False
        )
        
        if not text or len(text.strip()) < 100:
            return False, None, "Extracted text too short or empty"
        
        # Generate filename
        filename = sanitize_filename(url) + ".txt"
        filepath = output_dir / filename
        
        # Save cleaned text
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Source URL: {url}\n")
            f.write("=" * 80 + "\n\n")
            f.write(text)
        
        return True, filename, None
        
    except Exception as e:
        return False, None, str(e)


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    sources_file = project_root / "sources" / "urls.txt"
    output_dir = project_root / "work" / "clean_txt"
    
    # Check if sources file exists
    if not sources_file.exists():
        print(f"❌ Error: {sources_file} not found!")
        print("Please create sources/urls.txt with URLs to scrape (one per line)")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read URLs
    with open(sources_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    if not urls:
        print("❌ No URLs found in sources/urls.txt")
        sys.exit(1)
    
    print(f"📥 Found {len(urls)} URLs to process")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Process URLs
    success_count = 0
    failed_urls = []
    
    for url in tqdm(urls, desc="Processing URLs"):
        success, filename, error = fetch_and_clean(url, output_dir)
        
        if success:
            success_count += 1
            tqdm.write(f"✅ {filename}")
        else:
            failed_urls.append((url, error))
            tqdm.write(f"❌ Failed: {url[:60]}... - {error}")
        
        # Be polite to servers
        time.sleep(0.5)
    
    # Summary
    print()
    print("=" * 80)
    print(f"✅ Successfully processed: {success_count}/{len(urls)}")
    
    if failed_urls:
        print(f"❌ Failed: {len(failed_urls)}")
        print("\nFailed URLs:")
        for url, error in failed_urls:
            print(f"  - {url}")
            print(f"    Error: {error}")
    
    print()
    print(f"📁 Cleaned text files saved to: {output_dir}")


if __name__ == "__main__":
    main()
