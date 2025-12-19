#!/usr/bin/env python3
"""
Test script to verify each component of the pipeline.
Run without API keys to test fetching and filtering.
"""

import asyncio
import sys

async def test_arxiv_source():
    """Test arXiv fetching."""
    print("\n🧪 Testing arXiv Source...")
    from sources import ArxivSource
    
    source = ArxivSource(categories=["cs.LG", "cs.CL"])
    papers = await source.fetch(days_back=1, max_results=20)
    
    print(f"   ✅ Fetched {len(papers)} papers from arXiv")
    if papers:
        print(f"   📄 Sample: {papers[0].title[:60]}...")
        print(f"      Authors: {', '.join([a.name for a in papers[0].authors[:2]])}")
    
    return papers


async def test_hf_source():
    """Test HuggingFace Daily Papers fetching."""
    print("\n🧪 Testing HuggingFace Source...")
    from sources import HuggingFaceSource
    
    source = HuggingFaceSource()
    papers = await source.fetch()
    
    print(f"   ✅ Fetched {len(papers)} papers from HuggingFace")
    if papers:
        print(f"   📄 Sample: {papers[0].title[:60]}...")
    
    return papers


def test_keyword_filter(papers):
    """Test keyword filtering."""
    print("\n🧪 Testing Keyword Filter...")
    from filters import KeywordFilter
    
    keywords = [
        "diffusion", "language model", "reasoning",
        "representation", "safety", "transformer"
    ]
    
    filter_ = KeywordFilter(keywords=keywords)
    filtered = filter_.filter(papers)
    
    print(f"   ✅ Filtered {len(papers)} → {len(filtered)} papers")
    if filtered:
        print(f"   📄 Top match: {filtered[0].title[:60]}...")
        print(f"      Keywords: {', '.join(filtered[0].matched_keywords)}")
    
    return filtered


async def test_full_pipeline_dry():
    """Test the full pipeline without API keys."""
    print("=" * 60)
    print("🚀 Paper Assistant - Component Test")
    print("=" * 60)
    
    # Test sources
    arxiv_papers = await test_arxiv_source()
    hf_papers = await test_hf_source()
    
    # Combine and deduplicate
    all_papers = arxiv_papers + hf_papers
    seen = set()
    unique_papers = []
    for p in all_papers:
        key = p.arxiv_id or p.url
        if key not in seen:
            seen.add(key)
            unique_papers.append(p)
    
    print(f"\n📊 Total unique papers: {len(unique_papers)}")
    
    # Test filtering
    filtered = test_keyword_filter(unique_papers)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    print(f"   arXiv papers:     {len(arxiv_papers)}")
    print(f"   HuggingFace:      {len(hf_papers)}")
    print(f"   After dedup:      {len(unique_papers)}")
    print(f"   After filtering:  {len(filtered)}")
    print("\n✅ All components working!")
    
    if filtered:
        print("\n📝 Sample filtered papers:")
        for i, p in enumerate(filtered[:5], 1):
            print(f"\n{i}. {p.title}")
            print(f"   URL: {p.url}")
            print(f"   Keywords: {', '.join(p.matched_keywords)}")
    
    return filtered


if __name__ == "__main__":
    asyncio.run(test_full_pipeline_dry())
