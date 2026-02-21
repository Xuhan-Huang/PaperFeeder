# Updates Log

## 2026-02-21

### 1) Semantic Memory Persistence Moved to Dedicated Branch

- GitHub Actions now reads/writes `semantic_scholar_memory.json` from/to a dedicated `memory-state` branch.
- This isolates high-frequency memory state churn from `main` and reduces push/rebase friction for code changes.
- If `memory-state` or the memory file does not exist, workflow initializes an empty memory state safely.

### 2) Report-Visible Memory Marking Fix

- Semantic seen-memory updates now mark only Semantic Scholar papers that are actually visible in final rendered report links.
- This avoids false positives where a paper was selected internally but did not appear in final HTML sections.

### 3) Fine Filter Model Wiring + Parse Debugging

- Fine filtering path is explicitly wired to main `llm_*` model settings (while coarse filtering remains on `llm_filter_*`).
- Added detailed debug logging for `LLM filter: Could not parse response` cases:
  - full prompt
  - raw model response
  - model/base URL/stage metadata
  - saved under `llm_filter_debug/`

### 4) Documentation Sync (README Operational Notes)

- Added operational notes covering:
  - fetch/report dedup rules and current limitations
  - semantic memory TTL/cap behavior
  - `days_back` usage in GitHub Actions manual runs
  - daily git workflow with auto-updating memory branch
  - troubleshooting for parse failures and report count mismatches
- Clarified fine filter output target as Top `1-5`.
- Clarified OpenSpec artifacts are local workflow files (gitignored by default in this repo).

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
