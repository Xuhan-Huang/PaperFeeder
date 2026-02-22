#!/usr/bin/env python3
"""
Daily Paper Assistant - AI Agent Workflow
Pipeline: Fetch → Keyword Filter (Recall) → LLM Coarse Filter → Research (Enrichment) → LLM Fine Filter (Ranking) → Synthesize

NEW: Blog posts from priority sources (OpenAI, Anthropic, etc.) skip filtering!
"""

from __future__ import annotations

import asyncio
import argparse
import base64
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlsplit, urlunsplit

from sources import ArxivSource, HuggingFaceSource, ManualSource, SemanticScholarSource, BlogSource
from sources.blog_sources import fetch_blog_posts
from filters import KeywordFilter, LLMFilter
from researcher import PaperResearcher, MockPaperResearcher
from summarizer import PaperSummarizer
from emailer import ResendEmailer, FileEmailer
from config import Config
from models import Paper, PaperSource
from semantic_memory import SemanticMemoryStore
from semantic_feedback import (
    build_feedback_run_view_url,
    export_run_feedback_manifest,
    get_run_id_from_manifest,
    inject_feedback_actions_into_report,
    inject_feedback_entry_link,
    publish_feedback_run_to_d1,
)

# Check if blog sources are available (feedparser required)
try:
    import feedparser
    BLOG_SOURCE_AVAILABLE = True
except ImportError:
    BLOG_SOURCE_AVAILABLE = False



async def fetch_papers(config: Config, days_back: int = 1) -> List[Paper]:
    """Stage 1: Fetch papers from all sources (Recall)."""
    papers = []
    
    # arXiv
    print("📚 Fetching from arXiv...")
    arxiv_source = ArxivSource(config.arxiv_categories)
    arxiv_papers = await arxiv_source.fetch(days_back=days_back, max_results=300)
    papers.extend(arxiv_papers)
    print(f"   Found {len(arxiv_papers)} papers")
    
    # Hugging Face Daily Papers
    print("🤗 Fetching from HuggingFace Daily Papers...")
    hf_source = HuggingFaceSource()
    hf_papers = await hf_source.fetch()
    papers.extend(hf_papers)
    print(f"   Found {len(hf_papers)} papers")
    
    # Manual additions (from D1 or local)
    if config.manual_source_enabled:
        print("📝 Fetching manual additions...")
        manual_source = ManualSource(config.manual_source_path)
        manual_papers = await manual_source.fetch()
        papers.extend(manual_papers)
        print(f"   Found {len(manual_papers)} papers")

    # Semantic Scholar recommendations (seed-based)
    setattr(config, "_semantic_memory_store", None)
    if getattr(config, "semantic_scholar_enabled", False):
        print("🧠 Fetching from Semantic Scholar recommendations...")
        memory_store = None
        if getattr(config, "semantic_memory_enabled", True):
            memory_store = SemanticMemoryStore(
                path=getattr(config, "semantic_memory_path", "semantic_scholar_memory.json"),
                max_ids=getattr(config, "semantic_memory_max_ids", 5000),
            )
            memory_store.load()
            pruned = memory_store.prune_expired(getattr(config, "semantic_seen_ttl_days", 30))
            if pruned:
                print(f"      🧹 Semantic memory pruned expired seen IDs: {pruned}")
        s2_source = SemanticScholarSource(
            api_key=getattr(config, "semantic_scholar_api_key", ""),
            seeds_path=getattr(config, "semantic_scholar_seeds_path", "semantic_scholar_seeds.json"),
            max_results=getattr(config, "semantic_scholar_max_results", 50),
            memory_store=memory_store,
            seen_ttl_days=getattr(config, "semantic_seen_ttl_days", 30),
        )
        s2_papers = await s2_source.fetch()
        papers.extend(s2_papers)
        print(f"   Found {len(s2_papers)} papers")
        stats = getattr(s2_source, "last_stats", None)
        if stats:
            print(
                "   📊 Semantic Scholar stats: "
                f"total={stats.get('total', 0)}, "
                f"suppressed={stats.get('suppressed', 0)}, "
                f"forwarded={stats.get('forwarded', 0)}"
            )
        setattr(config, "_semantic_memory_store", memory_store)
    
    # Deduplicate by arxiv_id or url
    seen = set()
    unique_papers = []
    for p in papers:
        key = p.arxiv_id or p.url
        if key not in seen:
            seen.add(key)
            unique_papers.append(p)
    
    print(f"✅ Total unique papers: {len(unique_papers)}")
    return unique_papers


def _normalize_url_for_match(url: str) -> str:
    """Normalize URL for robust report-to-paper matching."""
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
        scheme = parts.scheme.lower() if parts.scheme else "https"
        netloc = parts.netloc.lower()
        path = parts.path.rstrip("/")
        return urlunsplit((scheme, netloc, path, "", ""))
    except Exception:
        return url.strip().lower().rstrip("/")


def _extract_report_urls(report_html: str) -> set[str]:
    """Extract all href URLs from rendered report HTML."""
    if not report_html:
        return set()
    urls = set()
    for raw in re.findall(r'href=["\']([^"\']+)["\']', report_html, flags=re.IGNORECASE):
        norm = _normalize_url_for_match(raw)
        if norm:
            urls.add(norm)
    return urls


def update_semantic_memory_from_report(final_papers: List[Paper], report_html: str, config: Config) -> None:
    """Mark only report-visible Semantic Scholar papers as seen and persist memory."""
    if not getattr(config, "semantic_memory_enabled", True):
        return

    memory_store = getattr(config, "_semantic_memory_store", None)
    if memory_store is None:
        return

    report_urls = _extract_report_urls(report_html)
    s2_final = [
        p
        for p in final_papers
        if getattr(p, "source", None) == PaperSource.SEMANTIC_SCHOLAR
        and getattr(p, "semantic_paper_id", None)
    ]
    if not s2_final:
        print("   ⭕ Semantic memory: no final Semantic Scholar papers")
        return

    visible_ids = sorted(
        {
            p.semantic_paper_id
            for p in s2_final
            if _normalize_url_for_match(getattr(p, "url", "")) in report_urls
        }
    )
    if not visible_ids:
        print(
            "   ⭕ Semantic memory: no report-visible Semantic Scholar papers to update "
            f"(final_selected={len(s2_final)})"
        )
        return

    try:
        memory_store.mark_seen(visible_ids)
        removed = memory_store.prune_expired(getattr(config, "semantic_seen_ttl_days", 30))
        memory_store.save()
        print(
            "   ✅ Semantic memory updated: "
            f"final_selected={len(s2_final)}, report_visible={len(visible_ids)}, "
            f"seen_added={len(visible_ids)}, expired_removed={removed}"
        )
    except Exception as e:
        print(f"   ⚠️ Semantic memory update failed (non-blocking): {e}")


async def fetch_blogs(config: Config, days_back: int = 7) -> tuple[List[Paper], List[Paper]]:
    """
    Fetch blog posts from RSS feeds.
    
    Returns:
        (priority_posts, normal_posts)
        - priority_posts: From top labs (OpenAI, Anthropic, etc.), skip filtering
        - normal_posts: From other sources, go through normal pipeline
    """
    if not BLOG_SOURCE_AVAILABLE:
        return [], []
    
    # Check if blogs are enabled
    if not getattr(config, 'blogs_enabled', True):
        return [], []
    
    print("📝 Fetching from blogs...")
    
    # Get config options
    enabled_blogs = getattr(config, 'enabled_blogs', None)
    custom_blogs = getattr(config, 'custom_blogs', None)
    blog_days_back = getattr(config, 'blog_days_back', days_back)
    
    source = BlogSource(
        enabled_blogs=enabled_blogs,
        custom_blogs=custom_blogs,
        include_non_priority=True,
    )
    
    all_posts = await source.fetch(
        days_back=blog_days_back,
        max_posts_per_blog=5,
    )
    
    # Separate priority and normal
    priority_posts = [p for p in all_posts if getattr(p, 'skip_filter', False)]
    normal_posts = [p for p in all_posts if not getattr(p, 'skip_filter', False)]
    
    print(f"   🔥 Priority blogs (skip filter): {len(priority_posts)}")
    print(f"   📄 Normal blogs (go through filter): {len(normal_posts)}")
    
    return priority_posts, normal_posts


async def filter_papers_coarse(papers: List[Paper], config: Config) -> List[Paper]:
    """
    Stage 2 & 3: Apply filters to select relevant papers.
    
    Stage 2: Keyword filter (Recall) - 保留较多数量，避免漏网之鱼
    Stage 3: LLM Coarse filter - 基于title+abstract粗筛，得到Top 20
    """
    print(f"\n🔍 Filtering {len(papers)} papers...")
    
    # Stage 2: Keyword filter (Recall)
    print("\n--- Stage 2: Keyword Filter (Recall) ---")
    keyword_filter = KeywordFilter(
        keywords=config.keywords,
        exclude_keywords=config.exclude_keywords
    )
    filtered = keyword_filter.filter(papers)
    print(f"   ✅ Keyword filter: {len(filtered)} papers matched (保留较多，避免漏网)")
    
    # Stage 3: LLM Coarse filter (Title + Abstract only)
    print("\n--- Stage 3: LLM Coarse Filter (Title + Abstract) ---")
    if config.llm_filter_enabled and len(filtered) > config.llm_filter_threshold:
        # Use filter-specific API key if provided, otherwise use main LLM API key
        filter_api_key = config.llm_filter_api_key
        filter_base_url = config.llm_filter_base_url
        filter_model = config.llm_filter_model
        
        print(f"   🤖 Applying LLM Coarse Filter ({filter_model})...")
        llm_filter = LLMFilter(
            api_key=filter_api_key,
            research_interests=config.research_interests,
            base_url=filter_base_url,
            model=filter_model
        )
        
        # Coarse filtering: 不包含community signals
        filtered = await llm_filter.filter(
            filtered, 
            max_papers=20,  # 粗筛得到Top 20进入下一阶段
            include_community_signals=False
        )
        print(f"   ✅ LLM Coarse Filter: {len(filtered)} papers selected for enrichment")
        
        # Show top papers
        if filtered:
            print(f"   📌 Top candidates for research:")
            for i, paper in enumerate(filtered[:5], 1):
                score = getattr(paper, 'relevance_score', 0) * 10
                print(f"      {i}. [{score:.1f}/10] {paper.title[:60]}...")
    elif config.llm_filter_enabled:
        print(f"   ⭕ Skipping LLM Coarse Filter (only {len(filtered)} papers, threshold: {config.llm_filter_threshold})")
    
    return filtered


async def enrich_papers(papers: List[Paper], config: Config) -> List[Paper]:
    """
    Stage 4: Research (Enrichment) - 联网调研，收集社区信号
    """
    print("\n--- Stage 4: Research & Enrichment ---")
    
    # Check if Tavily API key is available
    tavily_api_key = config.tavily_api_key
    
    if not tavily_api_key:
        print("   ⚠️  TAVILY_API_KEY not found, using mock researcher")
        researcher = MockPaperResearcher()
    else:
        print(f"   🔍 Using Tavily API for research")
        researcher = PaperResearcher(
            api_key=tavily_api_key,
            max_concurrent=5,
            search_depth="basic"
        )
    
    # Enrich papers with community signals
    enriched_papers = await researcher.research(papers)
    
    # Show some research results
    print(f"\n   📊 Sample research notes:")
    for i, paper in enumerate(enriched_papers[:3], 1):
        notes = getattr(paper, 'research_notes', 'N/A')
        print(f"      {i}. {paper.title[:50]}...")
        print(f"         🔍 {notes[:100]}...")
    
    return enriched_papers


async def filter_papers_fine(papers: List[Paper], config: Config) -> List[Paper]:
    """
    Stage 5: LLM Fine Filter (Ranking) - 基于content + community signals精筛
    从20篇中选出真正值得深度阅读的Top 3
    """
    print("\n--- Stage 5: LLM Fine Filter (Ranking with Community Signals) ---")
    
    if not config.llm_filter_enabled:
        print("   LLM filter disabled, returning all papers")
        return papers[:config.max_papers]
    
    # Fine ranking uses filter model settings (same provider path as filtering stage).
    filter_api_key = config.llm_filter_api_key
    filter_base_url = config.llm_filter_base_url
    filter_model = config.llm_filter_model
    
    print(f"   🤖 Applying LLM Fine Filter with Community Signals ({filter_model})...")
    llm_filter = LLMFilter(
        api_key=filter_api_key,
        research_interests=config.research_interests,
        base_url=filter_base_url,
        model=filter_model
    )
    
    # Fine filtering: 包含community signals
    final_papers = await llm_filter.filter(
        papers,
        max_papers=config.max_papers,  # 精筛得到最终的Top 3
        include_community_signals=True  # 关键: 使用community signals
    )
    print(f"   ✅ LLM Fine Filter: Selected {len(final_papers)} papers for final report")
    
    # Show final selections with reasons
    if final_papers and hasattr(final_papers[0], 'filter_reason'):
        print(f"\n   🏆 Final selections:")
        for i, paper in enumerate(final_papers, 1):
            reason = getattr(paper, 'filter_reason', '')
            score = getattr(paper, 'relevance_score', 0) * 10
            print(f"      {i}. [{score:.1f}/10] {paper.title[:50]}...")
            if reason:
                print(f"         → {reason[:80]}...")
    
    return final_papers


async def summarize_papers(papers: list[Paper], config: Config, priority_blogs: list[Paper] = None) -> str:
    """
    Stage 6: Synthesize - 生成最终报告

    Args:
        papers: Filtered papers from the pipeline
        config: Configuration
        priority_blogs: All blog posts (skip filtering/research, go directly to summarize)
    """
    print(f"\n--- Stage 6: Synthesis (Report Generation) ---")
    
    # Merge priority blogs with filtered papers
    all_content = []
    
    if priority_blogs:
        print(f"   🔥 Including {len(priority_blogs)} priority blog posts")
        all_content.extend(priority_blogs)
    
    all_content.extend(papers)
    
    print(f"   📝 Generating report for {len(all_content)} items...")
    print(f"      - Blog posts: {len(priority_blogs) if priority_blogs else 0}")
    print(f"      - Papers: {len(papers)}")
    print(f"   Using: {config.llm_model} @ {config.llm_base_url}")
    
    # 从环境变量或配置读取调试选项
    debug_save_pdfs = getattr(config, 'debug_save_pdfs', False)
    debug_pdf_dir = getattr(config, 'debug_pdf_dir', 'debug_pdfs')
    pdf_max_pages = getattr(config, 'pdf_max_pages', 10)
    
    summarizer = PaperSummarizer(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        research_interests=config.research_interests,
        debug_save_pdfs=debug_save_pdfs,
        debug_pdf_dir=debug_pdf_dir,
        pdf_max_pages=pdf_max_pages
    )
    
    # 使用PDF多模态输入（如果模型支持）
    report = await summarizer.generate_report(
        all_content,
        use_pdf_multimodal=config.extract_fulltext,
    )
    print("   ✅ Report generated!")
    return report


async def send_email(report: str, config: Config, attachments: Optional[List[dict]] = None) -> bool:
    """Send the report via email."""
    print(f"\n📧 Sending email to {config.email_to}...")
    
    emailer = ResendEmailer(
        api_key=config.resend_api_key,
        from_email=config.email_from
    )
    
    today = datetime.now().strftime("%Y-%m-%d")
    subject = f"📚 Daily Paper Digest - {today}"
    
    success = await emailer.send(
        to=config.email_to,
        subject=subject,
        html_content=report,
        attachments=attachments or [],
    )
    
    if success:
        print("   ✅ Email sent successfully!")
    else:
        print("   ❌ Failed to send email")
    
    return success


def _build_email_attachments(paths: List[str]) -> List[dict]:
    """Build base64 email attachments from local files."""
    attachments: List[dict] = []
    for p in paths:
        path = Path(p)
        if not path.exists() or not path.is_file():
            continue
        content = base64.b64encode(path.read_bytes()).decode("ascii")
        attachments.append(
            {
                "filename": path.name,
                "content": content,
                "content_type": "application/json",
            }
        )
    return attachments


async def run_pipeline(config_path: str = "config.yaml", days_back: int = 1, dry_run: bool = False, no_papers: bool = False, no_blogs: bool = False):
    """
    Run the full AI Agent pipeline.

    Workflow:
    1. Fetch: 获取论文 (arXiv, HuggingFace, Manual) - 可选
    1b. Fetch Blogs: 获取博客 (完全跳过过滤/研究，直接进summarize) - 可选
    2. Keyword Filter (Recall): 关键词匹配，保留较多数量 (仅论文)
    3. LLM Coarse Filter: 基于title+abstract粗筛，得到Top 20 (仅论文)
    4. Research (Enrichment): 联网调研Top 20，获取社区信号 (仅论文)
    5. LLM Fine Filter (Ranking): 基于content+signals精筛，得到Top 3 (仅论文)
    6. Synthesize: 生成最终报告 (论文 + 所有博客)

    Use --no-papers or --no-blogs to selectively disable sources.
    Blogs skip all filtering/research stages and go directly to report generation.
    """
    print("=" * 80)
    print(f"🚀 PaperFeeder AI Agent - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print("\n📋 Workflow: Fetch → Recall → Coarse Filter → Enrich → Fine Filter → Synthesize\n")

    # Set environment variables for source control before loading config
    if no_papers:
        os.environ['PAPERS_ENABLED'] = 'false'
    if no_blogs:
        os.environ['BLOGS_ENABLED'] = 'false'

    # Load config
    config = Config.from_yaml(config_path)
    
    # Stage 1: Fetch (Recall)
    print("=" * 80)
    print("STAGE 1: FETCH (Recall)")
    print("=" * 80)

    # Fetch papers if enabled
    papers = []
    if getattr(config, 'papers_enabled', True):
        papers = await fetch_papers(config, days_back=days_back)
    else:
        print("   📄 Paper fetching disabled (--no-papers flag)")

    # Stage 1b: Fetch Blogs (directly to summarize, skip filtering/research)
    priority_blogs, normal_blogs = await fetch_blogs(config, days_back=7)

    # Combine all blogs for direct summarization
    all_blogs = priority_blogs + normal_blogs
    if all_blogs:
        print(f"   📝 {len(all_blogs)} blogs fetched (skip filtering, go directly to summarize)")

    if not papers and not all_blogs:
        print("\n⚠️ No papers or blogs found. Exiting.")
        return

    # Stage 2-3: Keyword Filter + LLM Coarse Filter (papers only)
    coarse_filtered = []
    if papers:
        print("\n" + "=" * 80)
        print("STAGE 2-3: FILTERING (Recall → Coarse)")
        print("=" * 80)
        coarse_filtered = await filter_papers_coarse(papers, config)

    if not coarse_filtered and not all_blogs:
        print("\n⚠️ No papers passed coarse filter and no blogs. Exiting.")
        return

    # Stage 4: Research & Enrichment (papers only)
    enriched_papers = []
    if coarse_filtered:
        print("\n" + "=" * 80)
        print("STAGE 4: ENRICHMENT (Research)")
        print("=" * 80)
        enriched_papers = await enrich_papers(coarse_filtered, config)

    # Stage 5: LLM Fine Filter (Ranking) (papers only)
    final_papers = []
    if enriched_papers:
        print("\n" + "=" * 80)
        print("STAGE 5: RANKING (Fine Filter with Signals)")
        print("=" * 80)
        final_papers = await filter_papers_fine(enriched_papers, config)
    
    if not final_papers and not all_blogs:
        print("\n⚠️ No papers passed fine filter and no blogs. Exiting.")
        return

    # Stage 6: Synthesize (includes all blogs!)
    print("\n" + "=" * 80)
    print("STAGE 6: SYNTHESIS (Report Generation)")
    print("=" * 80)
    report = await summarize_papers(final_papers, config, priority_blogs=all_blogs)

    email_report = report
    # Export feedback manifest for human-in-the-loop seed updates (non-blocking).
    try:
        feedback_artifacts = export_run_feedback_manifest(
            final_papers,
            report,
            output_dir="artifacts",
            feedback_endpoint_base_url=getattr(config, "feedback_endpoint_base_url", ""),
            feedback_link_signing_secret=getattr(config, "feedback_link_signing_secret", ""),
            reviewer=getattr(config, "feedback_reviewer", "") or getattr(config, "email_to", ""),
            token_ttl_days=getattr(config, "feedback_token_ttl_days", 7),
        )
        if feedback_artifacts:
            manifest_path, questionnaire_path = feedback_artifacts
            print(f"   ✅ Feedback manifest exported: {manifest_path}")
            print(f"   ✅ Feedback questionnaire template exported: {questionnaire_path}")
            try:
                web_report = inject_feedback_actions_into_report(report, str(manifest_path))
                run_id = get_run_id_from_manifest(str(manifest_path))
                run_view_url = build_feedback_run_view_url(
                    getattr(config, "feedback_endpoint_base_url", ""),
                    run_id,
                )
                if run_view_url:
                    email_report = inject_feedback_entry_link(report, run_view_url)
                    print("   ✅ Web feedback entry link injected into email report")
                else:
                    print("   ⚠️ Feedback endpoint base URL missing; skipped web feedback entry link")

                try:
                    publish_feedback_run_to_d1(
                        manifest_path=str(manifest_path),
                        report_html=web_report,
                        account_id=getattr(config, "cloudflare_account_id", ""),
                        api_token=getattr(config, "cloudflare_api_token", ""),
                        database_id=getattr(config, "d1_database_id", ""),
                    )
                    print("   ✅ Published web viewer report to D1")
                except Exception as e:
                    print(f"   ⚠️ D1 run publish failed (non-blocking): {e}")
            except Exception as e:
                print(f"   ⚠️ Feedback web-view build failed (non-blocking): {e}")
        else:
            print("   ⭕ Feedback manifest: no report-visible papers to export")
    except Exception as e:
        print(f"   ⚠️ Feedback manifest export failed (non-blocking): {e}")

    # Persist seen-memory only for report-visible Semantic Scholar papers (non-blocking).
    update_semantic_memory_from_report(final_papers, report, config)
    
    # Output/Send
    print("\n" + "=" * 80)
    print("DELIVERY")
    print("=" * 80)
    
    if dry_run:
        print("\n📝 DRY RUN - Saving report to file...")
        file_emailer = FileEmailer("report_preview.html")
        await file_emailer.send(
            to=config.email_to,
            subject=f"Paper Digest - {datetime.now().strftime('%Y-%m-%d')}",
            html_content=email_report
        )
        print("✅ Report saved to report_preview.html")
    else:
        await send_email(email_report, config)
    
    print("\n" + "=" * 80)
    print("✨ Pipeline Complete!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   - Papers fetched: {len(papers)}")
    print(f"   - Blogs fetched (skip filter/research): {len(all_blogs)}")
    print(f"   - After keyword filter: {len(coarse_filtered) if coarse_filtered else 0}")
    print(f"   - After enrichment: {len(enriched_papers) if enriched_papers else 0}")
    print(f"   - Final papers: {len(final_papers) if final_papers else 0}")
    print(f"   - Total in report: {len(final_papers) + len(all_blogs)}")


def main():
    parser = argparse.ArgumentParser(
        description="PaperFeeder AI Agent - Hunt for 'The Next Big Thing'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Fetch papers from arXiv, HuggingFace, and manual sources (optional)
  1b. Fetch blogs from RSS feeds (optional, priority blogs skip filtering!)
  2. Keyword filter (Recall) - Cast a wide net
  3. LLM Coarse filter - Quick scoring based on title/abstract → Top 20
  4. Research & Enrichment - Gather community signals via Tavily API
  5. LLM Fine filter - Deep ranking with community signals → Top 3
  6. Synthesis - Generate "Editor's Choice" style report

NEW: Flexible Source Selection
  - Use --no-papers to disable paper fetching (only blogs)
  - Use --no-blogs to disable blog fetching (only papers)
  - Priority blogs (OpenAI, Anthropic, DeepMind, etc.) skip filtering
  - Normal blogs go through the full pipeline
  - Configure in config.yaml or via environment variables

Environment Variables:
  LLM_API_KEY         - Main LLM API key (for summarization)
  LLM_FILTER_API_KEY  - Filter LLM API key (optional, uses cheaper model)
  TAVILY_API_KEY      - Tavily search API key (for research stage)
  RESEND_API_KEY      - Email delivery API key
  EMAIL_TO            - Recipient email address
        """
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--days", type=int, default=1, help="Days to look back for papers")
    parser.add_argument("--blog-days", type=int, default=7, help="Days to look back for blogs")
    parser.add_argument("--dry-run", action="store_true", help="Don't send email, save to file")
    parser.add_argument("--no-blogs", action="store_true", help="Disable blog fetching")
    parser.add_argument("--no-papers", action="store_true", help="Disable paper fetching")
    
    args = parser.parse_args()
    
    asyncio.run(run_pipeline(
        config_path=args.config,
        days_back=args.days,
        dry_run=args.dry_run,
        no_papers=args.no_papers,
        no_blogs=args.no_blogs
    ))


if __name__ == "__main__":
    main()
