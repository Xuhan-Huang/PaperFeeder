"""
Configuration management for the paper assistant.
Updated: Added blog source configuration support.
"""

from __future__ import annotations

import os
import yaml
from dotenv import load_dotenv

# 自动加载 .env 文件
load_dotenv()
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Config:
    # LLM settings (通用配置，支持任意 OpenAI 兼容 API)
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    
    # 常用预设:
    # OpenAI:    base_url="https://api.openai.com/v1", model="gpt-4o-mini"
    # Claude:    base_url="https://api.anthropic.com/v1", model="claude-sonnet-4-20250514"
    # DeepSeek:  base_url="https://api.deepseek.com/v1", model="deepseek-chat"
    # Gemini:    base_url="https://generativelanguage.googleapis.com/v1beta/openai", model="gemini-2.0-flash"
    # Qwen:      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-turbo"
    # Local:     base_url="http://localhost:11434/v1", model="llama3"
    
    # Email settings
    resend_api_key: str = ""
    email_to: str = ""
    email_from: str = "paperfeeder@resend.dev"
    
    # arXiv settings (fewer categories = faster queries)
    arxiv_categories: list[str] = field(default_factory=lambda: [
        "cs.LG",   # Machine Learning
        "cs.CL",   # Computation and Language  
        # "cs.CV",   # Computer Vision - 可选，取消注释来启用
        # "cs.AI",   # Artificial Intelligence - 可选
        # "stat.ML", # Statistics - Machine Learning - 可选
    ])
    
    # Keywords for filtering (title + abstract)
    keywords: list[str] = field(default_factory=lambda: [
        # Generative models
        "diffusion model", "diffusion language", "flow matching",
        "generative model", "autoregressive",
        # LLM reasoning
        "chain of thought", "reasoning", "llm", "large language model",
        "in-context learning", "prompt",
        # Representation learning
        "representation learning", "contrastive learning", 
        "self-supervised", "foundation model",
        # AI Safety
        "ai safety", "alignment", "rlhf", "red teaming",
        "jailbreak", "safety benchmark", "harmful",
        # Specific interests
        "tokenizer", "tokenization", "continuous token",
        "latent space", "latent reasoning",
    ])
    
    exclude_keywords: list[str] = field(default_factory=lambda: [
        # Exclude if you want
    ])
    
    # Research interests description (for LLM filtering/summarization)
    research_interests: str = """
    I'm a Master's student researching:
    1. Generative models, especially diffusion models for language
    2. LLM reasoning, including chain-of-thought and latent reasoning
    3. Representation learning and continuous tokenization
    4. AI safety, including benchmarks and alignment
    
    I'm particularly interested in papers that:
    - Bridge generation and representation learning
    - Propose new reasoning paradigms for LLMs
    - Introduce novel safety evaluation methods
    - Have strong mathematical foundations
    """
    
    # Filtering settings
    llm_filter_enabled: bool = True   # Enable LLM-based filtering (recommended)
    llm_filter_threshold: int = 5     # Only use LLM filter if > N papers after keyword filter
    max_papers: int = 20              # Max papers in final report
    
    # LLM Filter settings (use cheaper model for filtering)
    llm_filter_api_key: str = ""      # API key for filter LLM
    llm_filter_base_url: str = "https://api.openai.com/v1"  # Base URL for filter LLM
    llm_filter_model: str = "gpt-4o-mini"  # Cheaper model for filtering (e.g., gpt-4o-mini, gpt-3.5-turbo)

    tavily_api_key: str = ""
    
    # Runner-local paper extraction and holistic synthesis
    extract_fulltext: bool = True     # Download PDFs and extract bounded text on the runner
    fulltext_top_n: int = 5           # Deprecated, kept for compatibility
    pdf_max_pages: int = 10
    synthesis_mode: str = "structured"  # structured or legacy
    paper_extraction_mode: str = "markdown"  # markdown, text, or auto
    pdf_download_timeout_sec: int = 60
    pdf_download_retries: int = 2
    pdf_max_bytes: int = 25000000
    paper_evidence_chars: int = 18000
    synthesis_aggregate_chars: int = 180000
    extraction_quality_threshold: int = 70
    extraction_quality_min_chars_per_page: int = 200
    extraction_quality_max_empty_page_ratio: float = 0.5
    extraction_quality_min_coverage_ratio: float = 0.5
    extraction_quality_max_unreadable_ratio: float = 0.02
    extraction_quality_max_duplicate_ratio: float = 0.25
    tex_source_enabled: bool = False
    tex_source_max_papers: int = 3
    tex_download_timeout_sec: int = 60
    tex_archive_max_bytes: int = 20000000
    tex_expanded_max_bytes: int = 30000000
    tex_max_files: int = 250
    tex_file_max_bytes: int = 3000000
    tex_include_max_depth: int = 6
    synthesis_timeout_sec: int = 240
    synthesis_retries: int = 2
    synthesis_retry_base_delay_sec: float = 2.0
    synthesis_streaming: bool = True
    synthesis_failure_notification: bool = True
    adaptive_compaction_concurrency: int = 3
    adaptive_compaction_max_tokens: int = 4096
    synthesis_max_output_tokens: int = 16384
    blog_excerpt_chars: int = 1200
    
    # Source enablement settings
    papers_enabled: bool = True            # Enable fetching from paper sources (arXiv, HF, Manual)
    manual_source_enabled: bool = True
    manual_source_path: str = "manual_papers.json"  # Or D1 connection string
    semantic_scholar_enabled: bool = False
    semantic_scholar_api_key: str = ""
    semantic_scholar_max_results: int = 30
    semantic_scholar_seeds_path: str = "semantic_scholar_seeds.json"
    semantic_memory_enabled: bool = True
    semantic_memory_path: str = "semantic_scholar_memory.json"
    semantic_seen_ttl_days: int = 30
    semantic_memory_max_ids: int = 5000
    
    # =============================================================================
    # Blog Source Settings (NEW!)
    # =============================================================================
    blogs_enabled: bool = True        # Enable blog fetching from RSS feeds
    blog_days_back: int = 1           # How many days back to look for blog posts
    
    # Which blogs to enable (if None, uses all priority blogs)
    # Available keys: openai, anthropic, deepmind, google_ai, meta_ai,
    #                 bair, stanford_ai, karpathy, lilianweng, colah,
    #                 jay_alammar, distill, fastai, the_gradient,
    #                 nvidia_ai, microsoft_research, aws_ml,
    #                 alignment_forum, lesswrong_ai
    enabled_blogs: Optional[List[str]] = None
    
    # Custom blogs (add your own RSS feeds)
    # Format: {"key": {"name": "...", "feed_url": "...", "priority": True/False}}
    custom_blogs: Optional[Dict[str, Dict[str, Any]]] = None
    
    # D1 settings (for future chatbot integration)
    cloudflare_account_id: str = ""
    cloudflare_api_token: str = ""
    d1_database_id: str = ""
    
    # V3 one-click feedback settings
    feedback_endpoint_base_url: str = ""
    feedback_link_signing_secret: str = ""
    feedback_token_ttl_days: int = 7
    feedback_reviewer: str = ""
    feedback_resolution_enabled: bool = True
    feedback_resolution_timeout_sec: int = 8
    feedback_resolution_max_lookups: int = 25
    feedback_resolution_no_key_max_lookups: int = 10
    feedback_resolution_time_budget_sec: int = 20
    feedback_resolution_run_cache_enabled: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file, with env var overrides."""
        config_data = {}
        
        if os.path.exists(path):
            with open(path, "r") as f:
                config_data = yaml.safe_load(f) or {}
        
        # Environment variable overrides (for GitHub Actions secrets)
        env_overrides = {
            "llm_api_key": os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"),
            "llm_base_url": os.getenv("LLM_BASE_URL"),
            "llm_model": os.getenv("LLM_MODEL"),
            "llm_filter_api_key": os.getenv("LLM_FILTER_API_KEY"),
            "llm_filter_base_url": os.getenv("LLM_FILTER_BASE_URL"),
            "llm_filter_model": os.getenv("LLM_FILTER_MODEL"),
            "resend_api_key": os.getenv("RESEND_API_KEY"),
            "email_to": os.getenv("EMAIL_TO"),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "cloudflare_account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            "cloudflare_api_token": os.getenv("CLOUDFLARE_API_TOKEN"),
            "d1_database_id": os.getenv("D1_DATABASE_ID"),
            "feedback_endpoint_base_url": os.getenv("FEEDBACK_ENDPOINT_BASE_URL"),
            "feedback_link_signing_secret": os.getenv("FEEDBACK_LINK_SIGNING_SECRET"),
            "feedback_token_ttl_days": os.getenv("FEEDBACK_TOKEN_TTL_DAYS"),
            "feedback_reviewer": os.getenv("FEEDBACK_REVIEWER"),
            "feedback_resolution_enabled": os.getenv("FEEDBACK_RESOLUTION_ENABLED"),
            "feedback_resolution_timeout_sec": os.getenv("FEEDBACK_RESOLUTION_TIMEOUT_SEC"),
            "feedback_resolution_max_lookups": os.getenv("FEEDBACK_RESOLUTION_MAX_LOOKUPS"),
            "feedback_resolution_no_key_max_lookups": os.getenv("FEEDBACK_RESOLUTION_NO_KEY_MAX_LOOKUPS"),
            "feedback_resolution_time_budget_sec": os.getenv("FEEDBACK_RESOLUTION_TIME_BUDGET_SEC"),
            "feedback_resolution_run_cache_enabled": os.getenv("FEEDBACK_RESOLUTION_RUN_CACHE_ENABLED"),
            # Structured extraction and holistic synthesis
            "synthesis_mode": os.getenv("SYNTHESIS_MODE"),
            "paper_extraction_mode": os.getenv("PAPER_EXTRACTION_MODE"),
            "pdf_download_timeout_sec": os.getenv("PDF_DOWNLOAD_TIMEOUT_SEC"),
            "pdf_download_retries": os.getenv("PDF_DOWNLOAD_RETRIES"),
            "pdf_max_bytes": os.getenv("PDF_MAX_BYTES"),
            "paper_evidence_chars": os.getenv("PAPER_EVIDENCE_CHARS"),
            "synthesis_aggregate_chars": os.getenv("SYNTHESIS_AGGREGATE_CHARS"),
            "extraction_quality_threshold": os.getenv("EXTRACTION_QUALITY_THRESHOLD"),
            "extraction_quality_min_chars_per_page": os.getenv("EXTRACTION_QUALITY_MIN_CHARS_PER_PAGE"),
            "extraction_quality_max_empty_page_ratio": os.getenv("EXTRACTION_QUALITY_MAX_EMPTY_PAGE_RATIO"),
            "extraction_quality_min_coverage_ratio": os.getenv("EXTRACTION_QUALITY_MIN_COVERAGE_RATIO"),
            "extraction_quality_max_unreadable_ratio": os.getenv("EXTRACTION_QUALITY_MAX_UNREADABLE_RATIO"),
            "extraction_quality_max_duplicate_ratio": os.getenv("EXTRACTION_QUALITY_MAX_DUPLICATE_RATIO"),
            "tex_source_enabled": os.getenv("TEX_SOURCE_ENABLED"),
            "tex_source_max_papers": os.getenv("TEX_SOURCE_MAX_PAPERS"),
            "tex_download_timeout_sec": os.getenv("TEX_DOWNLOAD_TIMEOUT_SEC"),
            "tex_archive_max_bytes": os.getenv("TEX_ARCHIVE_MAX_BYTES"),
            "tex_expanded_max_bytes": os.getenv("TEX_EXPANDED_MAX_BYTES"),
            "tex_max_files": os.getenv("TEX_MAX_FILES"),
            "tex_file_max_bytes": os.getenv("TEX_FILE_MAX_BYTES"),
            "tex_include_max_depth": os.getenv("TEX_INCLUDE_MAX_DEPTH"),
            "synthesis_timeout_sec": os.getenv("SYNTHESIS_TIMEOUT_SEC"),
            "synthesis_retries": os.getenv("SYNTHESIS_RETRIES"),
            "synthesis_retry_base_delay_sec": os.getenv("SYNTHESIS_RETRY_BASE_DELAY_SEC"),
            "synthesis_streaming": os.getenv("SYNTHESIS_STREAMING"),
            "synthesis_failure_notification": os.getenv("SYNTHESIS_FAILURE_NOTIFICATION"),
            "adaptive_compaction_concurrency": os.getenv("ADAPTIVE_COMPACTION_CONCURRENCY"),
            "adaptive_compaction_max_tokens": os.getenv("ADAPTIVE_COMPACTION_MAX_TOKENS"),
            "synthesis_max_output_tokens": os.getenv("SYNTHESIS_MAX_OUTPUT_TOKENS"),
            "blog_excerpt_chars": os.getenv("BLOG_EXCERPT_CHARS"),
            # Source enablement
            "papers_enabled": os.getenv("PAPERS_ENABLED"),
            "semantic_scholar_enabled": os.getenv("SEMANTIC_SCHOLAR_ENABLED"),
            "semantic_scholar_api_key": os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            "semantic_scholar_max_results": os.getenv("SEMANTIC_SCHOLAR_MAX_RESULTS"),
            "semantic_scholar_seeds_path": os.getenv("SEMANTIC_SCHOLAR_SEEDS_PATH"),
            "semantic_memory_enabled": os.getenv("SEMANTIC_MEMORY_ENABLED"),
            "semantic_memory_path": os.getenv("SEMANTIC_MEMORY_PATH"),
            "semantic_seen_ttl_days": os.getenv("SEMANTIC_SEEN_TTL_DAYS"),
            "semantic_memory_max_ids": os.getenv("SEMANTIC_MEMORY_MAX_IDS"),
            # Blog settings from environment
            "blogs_enabled": os.getenv("BLOGS_ENABLED"),
            "blog_days_back": os.getenv("BLOG_DAYS_BACK"),
        }
        
        # Apply environment variable overrides only when a non-empty value is provided.
        for key, value in env_overrides.items():
            if value not in (None, ""):
                # Handle boolean conversion for source enablement
                if key in (
                    "blogs_enabled",
                    "papers_enabled",
                    "semantic_scholar_enabled",
                    "semantic_memory_enabled",
                    "feedback_resolution_enabled",
                    "feedback_resolution_run_cache_enabled",
                    "tex_source_enabled",
                    "synthesis_streaming",
                    "synthesis_failure_notification",
                ):
                    config_data[key] = value.lower() not in ("false", "0", "no", "off")
                # Handle int conversion for blog_days_back
                elif key in (
                    "blog_days_back",
                    "semantic_scholar_max_results",
                    "semantic_seen_ttl_days",
                    "semantic_memory_max_ids",
                    "feedback_token_ttl_days",
                    "feedback_resolution_timeout_sec",
                    "feedback_resolution_max_lookups",
                    "feedback_resolution_no_key_max_lookups",
                    "feedback_resolution_time_budget_sec",
                    "pdf_download_timeout_sec",
                    "pdf_download_retries",
                    "pdf_max_bytes",
                    "paper_evidence_chars",
                    "synthesis_aggregate_chars",
                    "extraction_quality_threshold",
                    "extraction_quality_min_chars_per_page",
                    "tex_source_max_papers",
                    "tex_download_timeout_sec",
                    "tex_archive_max_bytes",
                    "tex_expanded_max_bytes",
                    "tex_max_files",
                    "tex_file_max_bytes",
                    "tex_include_max_depth",
                    "synthesis_timeout_sec",
                    "synthesis_retries",
                    "adaptive_compaction_concurrency",
                    "adaptive_compaction_max_tokens",
                    "synthesis_max_output_tokens",
                    "blog_excerpt_chars",
                ):
                    try:
                        config_data[key] = int(value)
                    except ValueError:
                        pass
                elif key in (
                    "extraction_quality_max_empty_page_ratio",
                    "extraction_quality_min_coverage_ratio",
                    "extraction_quality_max_unreadable_ratio",
                    "extraction_quality_max_duplicate_ratio",
                    "synthesis_retry_base_delay_sec",
                ):
                    try:
                        config_data[key] = float(value)
                    except ValueError:
                        pass
                else:
                    config_data[key] = value
        
        # Auto-detect base_url based on model if not explicitly set
        if config_data.get("llm_filter_model") and not config_data.get("llm_filter_base_url"):
            model = config_data["llm_filter_model"].lower()
            if "deepseek" in model:
                config_data["llm_filter_base_url"] = "https://api.deepseek.com/v1"
            elif "claude" in model:
                config_data["llm_filter_base_url"] = "https://api.anthropic.com/v1"
            elif "gemini" in model:
                config_data["llm_filter_base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif "qwen" in model:
                config_data["llm_filter_base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        return cls(**config_data)
    
    def to_yaml(self, path: str):
        """Save config to YAML file."""
        data = {
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "email_to": self.email_to,
            "email_from": self.email_from,
            "arxiv_categories": self.arxiv_categories,
            "keywords": self.keywords,
            "exclude_keywords": self.exclude_keywords,
            "research_interests": self.research_interests,
            "llm_filter_enabled": self.llm_filter_enabled,
            "llm_filter_threshold": self.llm_filter_threshold,
            "max_papers": self.max_papers,
            "llm_filter_api_key": self.llm_filter_api_key,
            "llm_filter_base_url": self.llm_filter_base_url,
            "llm_filter_model": self.llm_filter_model,
            "extract_fulltext": self.extract_fulltext,
            "fulltext_top_n": self.fulltext_top_n,
            "pdf_max_pages": getattr(self, 'pdf_max_pages', 10),
            "synthesis_mode": self.synthesis_mode,
            "paper_extraction_mode": self.paper_extraction_mode,
            "pdf_download_timeout_sec": self.pdf_download_timeout_sec,
            "pdf_download_retries": self.pdf_download_retries,
            "pdf_max_bytes": self.pdf_max_bytes,
            "paper_evidence_chars": self.paper_evidence_chars,
            "synthesis_aggregate_chars": self.synthesis_aggregate_chars,
            "extraction_quality_threshold": self.extraction_quality_threshold,
            "extraction_quality_min_chars_per_page": self.extraction_quality_min_chars_per_page,
            "extraction_quality_max_empty_page_ratio": self.extraction_quality_max_empty_page_ratio,
            "extraction_quality_min_coverage_ratio": self.extraction_quality_min_coverage_ratio,
            "extraction_quality_max_unreadable_ratio": self.extraction_quality_max_unreadable_ratio,
            "extraction_quality_max_duplicate_ratio": self.extraction_quality_max_duplicate_ratio,
            "tex_source_enabled": self.tex_source_enabled,
            "tex_source_max_papers": self.tex_source_max_papers,
            "tex_download_timeout_sec": self.tex_download_timeout_sec,
            "tex_archive_max_bytes": self.tex_archive_max_bytes,
            "tex_expanded_max_bytes": self.tex_expanded_max_bytes,
            "tex_max_files": self.tex_max_files,
            "tex_file_max_bytes": self.tex_file_max_bytes,
            "tex_include_max_depth": self.tex_include_max_depth,
            "synthesis_timeout_sec": self.synthesis_timeout_sec,
            "synthesis_retries": self.synthesis_retries,
            "synthesis_retry_base_delay_sec": self.synthesis_retry_base_delay_sec,
            "synthesis_streaming": self.synthesis_streaming,
            "synthesis_failure_notification": self.synthesis_failure_notification,
            "adaptive_compaction_concurrency": self.adaptive_compaction_concurrency,
            "adaptive_compaction_max_tokens": self.adaptive_compaction_max_tokens,
            "synthesis_max_output_tokens": self.synthesis_max_output_tokens,
            "blog_excerpt_chars": self.blog_excerpt_chars,
            "papers_enabled": self.papers_enabled,
            "manual_source_enabled": self.manual_source_enabled,
            "manual_source_path": self.manual_source_path,
            "semantic_scholar_enabled": self.semantic_scholar_enabled,
            "semantic_scholar_max_results": self.semantic_scholar_max_results,
            "semantic_scholar_seeds_path": self.semantic_scholar_seeds_path,
            "semantic_memory_enabled": self.semantic_memory_enabled,
            "semantic_memory_path": self.semantic_memory_path,
            "semantic_seen_ttl_days": self.semantic_seen_ttl_days,
            "semantic_memory_max_ids": self.semantic_memory_max_ids,
            # Blog settings
            "blogs_enabled": self.blogs_enabled,
            "blog_days_back": self.blog_days_back,
            "enabled_blogs": self.enabled_blogs,
            "custom_blogs": self.custom_blogs,
            "feedback_endpoint_base_url": self.feedback_endpoint_base_url,
            "feedback_token_ttl_days": self.feedback_token_ttl_days,
            "feedback_reviewer": self.feedback_reviewer,
            "feedback_resolution_enabled": self.feedback_resolution_enabled,
            "feedback_resolution_timeout_sec": self.feedback_resolution_timeout_sec,
            "feedback_resolution_max_lookups": self.feedback_resolution_max_lookups,
            "feedback_resolution_no_key_max_lookups": self.feedback_resolution_no_key_max_lookups,
            "feedback_resolution_time_budget_sec": self.feedback_resolution_time_budget_sec,
            "feedback_resolution_run_cache_enabled": self.feedback_resolution_run_cache_enabled,
        }
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def create_default_config(path: str = "config.yaml"):
    """Create a default config file."""
    config = Config()
    config.to_yaml(path)
    print(f"Created default config at {path}")
    print("Please edit it and add your API keys as environment variables.")
