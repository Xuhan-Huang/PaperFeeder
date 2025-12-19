"""
Paper filtering modules.
Supports keyword matching, embedding similarity, and LLM-based filtering.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional, List
import asyncio
import json

from models import Paper


class BaseFilter(ABC):
    """Abstract base class for paper filters."""
    
    @abstractmethod
    def filter(self, papers: List[Paper], **kwargs) -> List[Paper]:
        pass


class KeywordFilter(BaseFilter):
    """Filter papers based on keyword matching in title and abstract."""
    
    def __init__(
        self,
        keywords: List[str],
        exclude_keywords: List[str] = None,
        case_sensitive: bool = False,
        match_all: bool = False,  # If True, all keywords must match
    ):
        self.keywords = keywords
        self.exclude_keywords = exclude_keywords or []
        self.case_sensitive = case_sensitive
        self.match_all = match_all
        
        # Compile regex patterns for efficiency
        flags = 0 if case_sensitive else re.IGNORECASE
        self.patterns = [re.compile(rf"\b{re.escape(kw)}\b", flags) for kw in keywords]
        self.exclude_patterns = [re.compile(rf"\b{re.escape(kw)}\b", flags) for kw in self.exclude_keywords]
    
    def filter(self, papers: List[Paper], **kwargs) -> List[Paper]:
        """Filter papers that match keywords."""
        filtered = []
        
        for paper in papers:
            # Combine title and abstract for matching
            text = f"{paper.title} {paper.abstract}"
            
            # Check exclusions first
            excluded = False
            for pattern in self.exclude_patterns:
                if pattern.search(text):
                    excluded = True
                    break
            
            if excluded:
                continue
            
            # Check keyword matches
            matched_keywords = []
            for kw, pattern in zip(self.keywords, self.patterns):
                if pattern.search(text):
                    matched_keywords.append(kw)
            
            # Determine if paper passes filter
            if self.match_all:
                passes = len(matched_keywords) == len(self.keywords)
            else:
                passes = len(matched_keywords) > 0
            
            if passes:
                paper.matched_keywords = matched_keywords
                paper.relevance_score = len(matched_keywords) / len(self.keywords)
                filtered.append(paper)
        
        # Sort by relevance score (number of matched keywords)
        filtered.sort(key=lambda p: p.relevance_score, reverse=True)
        
        return filtered


class AuthorFilter(BaseFilter):
    """Filter papers by author names or affiliations."""
    
    def __init__(
        self,
        authors: list[str] = None,
        affiliations: list[str] = None,
    ):
        self.authors = [a.lower() for a in (authors or [])]
        self.affiliations = [a.lower() for a in (affiliations or [])]
    
    def filter(self, papers: list[Paper], **kwargs) -> list[Paper]:
        """Filter papers by author/affiliation."""
        if not self.authors and not self.affiliations:
            return papers
        
        filtered = []
        
        for paper in papers:
            match = False
            
            for author in paper.authors:
                # Check author name
                if self.authors:
                    author_name_lower = author.name.lower()
                    for target in self.authors:
                        if target in author_name_lower:
                            match = True
                            break
                
                # Check affiliation
                if self.affiliations and author.affiliation:
                    affiliation_lower = author.affiliation.lower()
                    for target in self.affiliations:
                        if target in affiliation_lower:
                            match = True
                            break
                
                if match:
                    break
            
            if match:
                filtered.append(paper)
        
        return filtered


class LLMFilter(BaseFilter):
    """Use LLM to filter papers based on research interests."""
    
    def __init__(
        self,
        api_key: str,
        research_interests: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",  # Use cheaper model for filtering
    ):
        self.api_key = api_key
        self.research_interests = research_interests
        self.base_url = base_url
        self.model = model
    
    async def filter(self, papers: list[Paper], max_papers: int = 20, **kwargs) -> list[Paper]:
        """Use LLM to score and filter papers."""
        from llm_client import LLMClient
        
        client = LLMClient(api_key=self.api_key, base_url=self.base_url, model=self.model)
        
        # Prepare papers summary for LLM
        papers_text = ""
        for i, paper in enumerate(papers):
            papers_text += f"""
Paper {i+1}:
Title: {paper.title}
Abstract: {paper.abstract[:500]}...
Categories: {', '.join(paper.categories)}
---
"""
        
        prompt = f"""You are a research paper filtering assistant. Given my research interests and a list of papers, score each paper's relevance from 0-10.

MY RESEARCH INTERESTS:
{self.research_interests}

PAPERS TO EVALUATE:
{papers_text}

Return a JSON array with the paper numbers and scores, sorted by relevance. Format:
[{{"paper_num": 1, "score": 8, "reason": "brief reason"}}, ...]

Only include papers with score >= 5. Be selective - I only want highly relevant papers.
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            result_text = await client.achat(messages, max_tokens=2000)
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if not json_match:
                print("LLM filter: Could not parse response")
                return papers[:max_papers]
            
            scores = json.loads(json_match.group())
            
            # Map scores back to papers
            scored_papers = []
            for item in scores:
                paper_idx = item["paper_num"] - 1
                if 0 <= paper_idx < len(papers):
                    paper = papers[paper_idx]
                    paper.relevance_score = item["score"] / 10.0
                    scored_papers.append(paper)
            
            # Sort by score and limit
            scored_papers.sort(key=lambda p: p.relevance_score, reverse=True)
            return scored_papers[:max_papers]
            
        except Exception as e:
            print(f"LLM filter error: {e}")
            return papers[:max_papers]
    
    def filter(self, papers: list[Paper], **kwargs) -> list[Paper]:
        """Sync wrapper for async filter."""
        return asyncio.run(self.filter(papers, **kwargs))


class CompositeFilter(BaseFilter):
    """Combine multiple filters."""
    
    def __init__(self, filters: list[BaseFilter], mode: str = "sequential"):
        """
        mode:
        - "sequential": Apply filters one after another
        - "union": Include paper if it passes any filter
        - "intersection": Include paper only if it passes all filters
        """
        self.filters = filters
        self.mode = mode
    
    def filter(self, papers: list[Paper], **kwargs) -> list[Paper]:
        if self.mode == "sequential":
            result = papers
            for f in self.filters:
                result = f.filter(result, **kwargs)
            return result
        
        elif self.mode == "union":
            seen = set()
            result = []
            for f in self.filters:
                for paper in f.filter(papers, **kwargs):
                    if paper not in seen:
                        seen.add(paper)
                        result.append(paper)
            return result
        
        elif self.mode == "intersection":
            if not self.filters:
                return papers
            
            # Start with first filter's results
            result_set = set(self.filters[0].filter(papers, **kwargs))
            
            # Intersect with subsequent filters
            for f in self.filters[1:]:
                result_set &= set(f.filter(papers, **kwargs))
            
            return list(result_set)
        
        return papers
