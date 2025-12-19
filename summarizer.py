"""
Paper summarization using any LLM.
Generates daily digest with summaries and insights.
"""

from __future__ import annotations

import asyncio
import base64
import aiohttp
from datetime import datetime
from typing import Optional, List

from models import Paper
from llm_client import LLMClient


class PaperSummarizer:
    """Generate paper summaries and insights using any LLM."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        research_interests: str = "",
        use_pdf_direct: bool = True,  # 直接传 PDF 给模型（如果支持）
    ):
        self.client = LLMClient(api_key=api_key, base_url=base_url, model=model)
        self.research_interests = research_interests
        self.use_pdf_direct = use_pdf_direct and self.client.supports_pdf_native()
    
    async def generate_report(
        self, 
        papers: list[Paper],
        include_full_paper: bool = False,  # 是否包含全文
        top_n_full: int = 3,  # 只对 top N 篇传全文
    ) -> str:
        """Generate a full HTML report for the papers."""
        
        # Build papers context
        papers_context = await self._build_papers_context(
            papers, 
            include_full_paper=include_full_paper,
            top_n_full=top_n_full
        )
        
        prompt = f"""You are an AI research assistant helping a researcher stay up-to-date with the latest papers.

RESEARCHER'S INTERESTS:
{self.research_interests}

TODAY'S PAPERS ({len(papers)} total):
{papers_context}

Generate a daily digest report in HTML format. Include:

1. **Executive Summary** (2-3 sentences): The most important trends or breakthroughs from today's papers.

2. **Top Highlights** (3-5 papers): The most relevant papers with:
   - Why it matters to the researcher's interests
   - Key contribution in one sentence
   - Potential applications or implications

3. **Paper Summaries**: For each paper, provide:
   - A 2-3 sentence summary of the main contribution
   - Key methods or findings
   - Relevance score indicator (🔥 highly relevant, ⭐ relevant, 📄 general interest)

4. **Research Insights**: 
   - Emerging trends across papers
   - Connections between papers
   - Questions or ideas for future exploration

Format the output as clean, readable HTML with:
- Use semantic HTML (h2, h3, p, ul, li)
- Include links to papers (use the provided URLs)
- Use emojis sparingly for visual hierarchy
- Keep it scannable - researchers are busy!
- Style should work well in email clients

Do not include <html>, <head>, or <body> tags - just the content that goes inside body.
"""
        
        messages = [{"role": "user", "content": prompt}]
        html_content = await self.client.achat(messages, max_tokens=4000)
        
        # Wrap in email-friendly HTML template
        return self._wrap_in_template(html_content, papers)
    
    async def _build_papers_context(
        self, 
        papers: list[Paper],
        include_full_paper: bool = False,
        top_n_full: int = 3,
    ) -> str:
        """Build a formatted context string for all papers."""
        context = ""
        
        for i, paper in enumerate(papers, 1):
            authors_str = ", ".join([a.name for a in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors_str += f" et al. ({len(paper.authors)} authors)"
            
            categories_str = ", ".join(paper.categories[:3]) if paper.categories else "N/A"
            
            context += f"""
---
Paper {i}:
Title: {paper.title}
Authors: {authors_str}
Categories: {categories_str}
URL: {paper.url}
Source: {paper.source.value}
Matched Keywords: {', '.join(paper.matched_keywords) if paper.matched_keywords else 'N/A'}

Abstract:
{paper.abstract}
"""
            # 对 top N 篇论文，尝试获取全文
            if include_full_paper and i <= top_n_full and paper.pdf_url:
                full_text = await self._get_paper_content(paper)
                if full_text:
                    # 截断到合理长度
                    truncated = full_text[:10000]
                    if len(full_text) > 10000:
                        truncated += "\n[... truncated ...]"
                    context += f"\nFull Paper Content:\n{truncated}\n"
            
            if paper.notes:
                context += f"Notes: {paper.notes}\n"
        
        return context
    
    async def _get_paper_content(self, paper: Paper) -> Optional[str]:
        """Get full paper content - either from PDF or extracted text."""
        # 如果已经有提取的全文
        if hasattr(paper, 'full_text') and paper.full_text:
            return paper.full_text
        
        # 尝试下载 PDF 并提取文本
        if paper.pdf_url:
            try:
                pdf_base64 = await self._download_pdf_base64(paper.pdf_url)
                if pdf_base64:
                    return self.client._extract_pdf_text_from_base64(pdf_base64)
            except Exception as e:
                print(f"      Failed to get paper content: {e}")
        
        return None
    
    async def _download_pdf_base64(self, url: str) -> Optional[str]:
        """Download PDF and return as base64."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        content = await response.read()
                        return base64.standard_b64encode(content).decode("utf-8")
        except Exception as e:
            print(f"      PDF download failed: {e}")
        return None
    
    async def summarize_single_paper_with_pdf(self, paper: Paper) -> str:
        """
        Summarize a single paper by directly passing its PDF to the LLM.
        Only works with models that support native PDF input (Claude, Gemini).
        """
        if not paper.pdf_url:
            return await self.summarize_single_paper(paper)
        
        if not self.client.supports_pdf_native():
            # Fallback to text extraction
            return await self.summarize_single_paper(paper)
        
        prompt = f"""Summarize this research paper, focusing on:
1. The main problem it addresses
2. The key contribution or method
3. Main results or findings
4. How it relates to: {self.research_interests}

Be concise and technical. Provide a 3-5 sentence summary."""

        try:
            return self.client.chat_with_pdf(prompt, pdf_url=paper.pdf_url)
        except Exception as e:
            print(f"PDF summarization failed: {e}")
            return await self.summarize_single_paper(paper)
    
    async def summarize_single_paper(self, paper: Paper) -> str:
        """Generate a summary for a single paper using abstract only."""
        prompt = f"""Summarize this research paper in 2-3 sentences, focusing on:
1. The main problem it addresses
2. The key contribution or method
3. Main results or findings

Title: {paper.title}

Abstract:
{paper.abstract}

Be concise and technical. Use specific terminology."""

        messages = [{"role": "user", "content": prompt}]
        return await self.client.achat(messages, max_tokens=300)
    
    def _wrap_in_template(self, content: str, papers: list[Paper]) -> str:
        """Wrap content in an email-friendly HTML template."""
        today = datetime.now().strftime("%B %d, %Y")
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Paper Digest - {today}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #555;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .paper-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #3498db;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 0.85em;
            color: #666;
            text-align: center;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Daily Paper Digest</h1>
        <p style="color: #666;">{today} • {len(papers)} papers curated for you</p>
        
        {content}
        
        <div class="footer">
            <p>Generated by Daily Paper Assistant</p>
            <p>Keywords matched: {self._get_unique_keywords(papers)}</p>
        </div>
    </div>
</body>
</html>"""
    
    def _get_unique_keywords(self, papers: list[Paper]) -> str:
        """Get unique matched keywords across all papers."""
        keywords = set()
        for paper in papers:
            keywords.update(paper.matched_keywords)
        return ", ".join(sorted(keywords)) if keywords else "N/A"


# Backward compatibility alias
ClaudeSummarizer = PaperSummarizer
