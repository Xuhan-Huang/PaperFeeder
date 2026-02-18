# Updates Log

## 2026-02-18

### 1) Semantic Scholar Source Integrated

- Added seed-based Semantic Scholar recommendations as a first-class paper source.
- Added config controls:
  - `semantic_scholar_enabled`
  - `semantic_scholar_max_results`
  - `semantic_scholar_seeds_path`
  - optional `SEMANTIC_SCHOLAR_API_KEY`
- Added local seed file:
  - `semantic_scholar_seeds.json`
  - keys: `positive_paper_ids`, `negative_paper_ids`
- Added ID normalization so numeric corpus IDs are automatically converted to `CorpusId:<id>`.
- Semantic Scholar candidates are merged into the existing paper pool before dedup/filtering.

### 2) Local Timeout / China Network Stability Improvements

- Enabled proxy-aware network requests (`aiohttp` with `trust_env=True`) in paper/blog/semantic source fetchers.
- Added stronger blog fetch robustness:
  - browser-like headers
  - retry on timeout/5xx
- Result:
  - fewer random blog timeouts in China environment
  - remaining failures are mostly deterministic endpoint issues (for example, Anthropic 404).

### 3) Coarse vs Fine Filter Model Wiring Fixed

- Coarse filter still uses `llm_filter_*` (cheap model path).
- Fine filter now uses main model settings `llm_*` as intended.
- This aligns final ranking with your preferred higher-quality model.

### 4) Additional Runtime Hardening

- Fixed arXiv recent-paper date filtering to use UTC-consistent comparison.
- Added null-safe handling for papers without `pdf_url` to avoid summarize-stage crash.
- Hardened PDF download error logging for missing URL cases.

