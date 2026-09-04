"""Runner-local paper content extraction for bounded LLM evidence packets."""

from __future__ import annotations

import asyncio
import io
import json
import random
import re
import tarfile
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Optional
from urllib.parse import quote

import aiohttp

from models import Paper


EXPECTED_SECTIONS = (
    "abstract",
    "introduction",
    "background",
    "method",
    "approach",
    "experiment",
    "result",
    "discussion",
    "limitation",
    "conclusion",
)
LOW_PRIORITY_SECTIONS = ("references", "bibliography", "acknowledgments", "acknowledgements")
ALLOWED_TEX_SUFFIXES = {".tex", ".bbl"}


@dataclass
class ExtractionSettings:
    enabled: bool = True
    mode: str = "markdown"
    pdf_max_pages: int = 15
    pdf_download_timeout_sec: int = 60
    pdf_download_retries: int = 2
    pdf_max_bytes: int = 25_000_000
    per_paper_chars: int = 18_000
    aggregate_chars: int = 180_000
    quality_threshold: int = 70
    quality_min_chars_per_page: int = 200
    quality_max_empty_page_ratio: float = 0.5
    quality_min_coverage_ratio: float = 0.5
    quality_max_unreadable_ratio: float = 0.02
    quality_max_duplicate_ratio: float = 0.25
    tex_enabled: bool = False
    tex_max_papers: int = 3
    tex_download_timeout_sec: int = 60
    tex_archive_max_bytes: int = 20_000_000
    tex_expanded_max_bytes: int = 30_000_000
    tex_max_files: int = 250
    tex_file_max_bytes: int = 3_000_000
    tex_include_max_depth: int = 6


@dataclass
class ExtractionQuality:
    score: int
    metrics: dict[str, Any]
    hard_fail_reasons: list[str] = field(default_factory=list)
    decision: str = "accept_pdf_markdown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidencePacket:
    item_id: str
    title: str
    url: str
    arxiv_id: str
    semantic_paper_id: str
    source: str
    abstract: str
    research_notes: str
    content: str
    extraction_source: str
    authors: list[str] = field(default_factory=list)
    total_pages: int = 0
    processed_pages: int = 0
    original_chars: int = 0
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)
    quality: Optional[ExtractionQuality] = None
    elapsed_seconds: float = 0.0

    @property
    def content_chars(self) -> int:
        return len(self.content)

    def diagnostic_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "url": self.url,
            "arxiv_id": self.arxiv_id or None,
            "semantic_paper_id": self.semantic_paper_id or None,
            "source": self.source,
            "extraction_source": self.extraction_source,
            "author_count": len(self.authors),
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "original_chars": self.original_chars,
            "content_chars": self.content_chars,
            "truncated": self.truncated,
            "warnings": list(self.warnings),
            "quality": self.quality.to_dict() if self.quality else None,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass
class _ExtractionCandidate:
    paper: Paper
    item_id: str
    markdown: str = ""
    plain_text: str = ""
    total_pages: int = 0
    processed_pages: int = 0
    quality: Optional[ExtractionQuality] = None
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _source_name(paper: Paper) -> str:
    source = getattr(paper, "source", "")
    return str(getattr(source, "value", source) or "")


def _fallback_content(paper: Paper) -> str:
    sections = [f"# {paper.title}"]
    if paper.abstract:
        sections.extend(("## Abstract", paper.abstract.strip()))
    if getattr(paper, "research_notes", None):
        sections.extend(("## Community Signals", str(paper.research_notes).strip()))
    if len(sections) == 1:
        sections.append("No extractable full text or abstract was available.")
    return "\n\n".join(sections).strip()


def _clean_text(value: str) -> str:
    value = (value or "").replace("\x00", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def _normalized_lines(value: str) -> list[str]:
    lines = []
    for raw_line in value.splitlines():
        line = " ".join(raw_line.lower().split())
        if len(line) >= 20:
            lines.append(line)
    return lines


def _duplicate_line_ratio(value: str) -> float:
    lines = _normalized_lines(value)
    if not lines:
        return 0.0
    return max(0.0, (len(lines) - len(set(lines))) / len(lines))


def _unreadable_ratio(value: str) -> float:
    visible = [char for char in value if not char.isspace()]
    if not visible:
        return 1.0
    unreadable = 0
    for char in visible:
        category = unicodedata.category(char)
        if char == "\ufffd" or (category.startswith("C") and char not in "\n\r\t"):
            unreadable += 1
    unreadable += len(re.findall(r"\(cid:\d+\)", value, flags=re.IGNORECASE))
    return min(1.0, unreadable / len(visible))


def _tokenize_title(title: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (title or "").lower())
        if len(token) >= 3
    }


def _title_recall(title: str, content: str) -> float:
    tokens = _tokenize_title(title)
    if not tokens:
        return 1.0
    content_lower = content.lower()
    return sum(token in content_lower for token in tokens) / len(tokens)


def _sections_found(content: str) -> list[str]:
    lowered = content.lower()
    return sorted({section for section in EXPECTED_SECTIONS if re.search(rf"\b{section}\w*\b", lowered)})


def _multi_column_page_ratio(document: Any) -> float:
    if not document or len(document) == 0:
        return 0.0
    risky_pages = 0
    for page in document:
        midpoint = page.rect.width / 2
        blocks = [block for block in page.get_text("blocks") if len(block) >= 5 and str(block[4]).strip()]
        left = [block for block in blocks if block[0] < midpoint * 0.8 and block[2] <= midpoint * 1.15]
        right = [block for block in blocks if block[0] >= midpoint * 0.85]
        if len(left) >= 2 and len(right) >= 2:
            risky_pages += 1
    return risky_pages / len(document)


def evaluate_markdown_quality(
    *,
    title: str,
    markdown: str,
    plain_text: str,
    page_texts: list[str],
    total_pages: int,
    processed_pages: int,
    multi_column_ratio: float,
    settings: ExtractionSettings,
) -> ExtractionQuality:
    markdown = _clean_text(markdown)
    plain_text = _clean_text(plain_text)
    markdown_chars = len(markdown)
    plain_chars = len(plain_text)
    page_count = max(1, len(page_texts))
    usable_chars_per_page = plain_chars / page_count
    empty_page_ratio = sum(len(_clean_text(page)) < 50 for page in page_texts) / page_count
    coverage_ratio = markdown_chars / plain_chars if plain_chars else (1.0 if markdown_chars else 0.0)
    unreadable_ratio = _unreadable_ratio(markdown)
    duplicate_ratio = _duplicate_line_ratio(markdown)
    title_recall = _title_recall(title, markdown)
    sections = _sections_found(markdown)
    conclusion_found = any(section in sections for section in ("conclusion", "discussion"))
    truncated_without_conclusion = total_pages > processed_pages and not conclusion_found

    metrics = {
        "markdown_chars": markdown_chars,
        "plain_text_chars": plain_chars,
        "usable_chars_per_page": round(usable_chars_per_page, 2),
        "empty_page_ratio": round(empty_page_ratio, 4),
        "coverage_ratio": round(coverage_ratio, 4),
        "unreadable_char_ratio": round(unreadable_ratio, 4),
        "duplicate_line_ratio": round(duplicate_ratio, 4),
        "title_token_recall": round(title_recall, 4),
        "sections_found": sections,
        "multi_column_page_ratio": round(multi_column_ratio, 4),
        "truncated_without_conclusion": truncated_without_conclusion,
    }

    hard_fail_reasons = []
    if markdown_chars < 200 and plain_chars >= 500:
        hard_fail_reasons.append("empty_or_tiny_markdown")
    if unreadable_ratio > max(0.10, settings.quality_max_unreadable_ratio * 5):
        hard_fail_reasons.append("predominantly_unreadable")
    if duplicate_ratio > max(0.65, settings.quality_max_duplicate_ratio * 2):
        hard_fail_reasons.append("severe_duplication")
    if plain_chars >= 1_000 and coverage_ratio < min(0.15, settings.quality_min_coverage_ratio):
        hard_fail_reasons.append("implausibly_low_coverage")
    if empty_page_ratio > 0.8:
        hard_fail_reasons.append("mostly_empty_pages")

    score = 100.0
    if usable_chars_per_page < settings.quality_min_chars_per_page:
        score -= 25
    elif usable_chars_per_page < settings.quality_min_chars_per_page * 3:
        score -= 10
    if empty_page_ratio > settings.quality_max_empty_page_ratio:
        score -= min(25, (empty_page_ratio - settings.quality_max_empty_page_ratio) * 50)
    if coverage_ratio < settings.quality_min_coverage_ratio:
        score -= min(30, (settings.quality_min_coverage_ratio - coverage_ratio) * 60)
    elif coverage_ratio > 1.8:
        score -= min(15, (coverage_ratio - 1.8) * 10)
    if unreadable_ratio > settings.quality_max_unreadable_ratio:
        score -= min(20, (unreadable_ratio - settings.quality_max_unreadable_ratio) * 200)
    if duplicate_ratio > settings.quality_max_duplicate_ratio:
        score -= min(20, (duplicate_ratio - settings.quality_max_duplicate_ratio) * 60)
    if title_recall < 0.4:
        score -= 10
    if len(sections) < 2:
        score -= 8
    score -= min(5, multi_column_ratio * 5)
    if truncated_without_conclusion:
        score -= 5
    score_value = max(0, min(100, int(round(score))))

    if hard_fail_reasons:
        decision = "hard_fail"
    elif score_value < settings.quality_threshold:
        decision = "try_alternative"
    else:
        decision = "accept_pdf_markdown"
    return ExtractionQuality(
        score=score_value,
        metrics=metrics,
        hard_fail_reasons=hard_fail_reasons,
        decision=decision,
    )


def _remove_repeated_lines(content: str) -> str:
    seen: set[str] = set()
    output = []
    for raw_line in content.splitlines():
        normalized = " ".join(raw_line.lower().split())
        if len(normalized) >= 30 and normalized in seen:
            continue
        if len(normalized) >= 30:
            seen.add(normalized)
        output.append(raw_line)
    return "\n".join(output)


def _remove_low_priority_tail(content: str) -> str:
    pattern = re.compile(
        r"(?im)^\s{0,3}(?:#{1,6}\s*)?(?:references|bibliography|acknowledg(?:e)?ments)\s*$"
    )
    match = pattern.search(content)
    if match and match.start() > len(content) * 0.45:
        return content[: match.start()].rstrip()
    return content


def truncate_evidence(content: str, limit: int) -> tuple[str, bool]:
    cleaned = _remove_low_priority_tail(_remove_repeated_lines(_clean_text(content)))
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned, False
    tail_size = min(limit // 5, 4_000)
    head_size = max(0, limit - tail_size - len("\n\n[...truncated...]\n\n"))
    truncated = cleaned[:head_size].rstrip()
    if tail_size:
        truncated += "\n\n[...truncated...]\n\n" + cleaned[-tail_size:].lstrip()
    return truncated[:limit], True


def balanced_character_limits(lengths: Iterable[int], total_limit: int, per_item_limit: int) -> list[int]:
    values = [max(0, min(int(length), max(0, per_item_limit))) for length in lengths]
    if total_limit <= 0:
        return [0 for _ in values]
    if sum(values) <= total_limit:
        return values
    allocations = [0 for _ in values]
    remaining = set(range(len(values)))
    budget = total_limit
    while remaining and budget > 0:
        share = max(1, budget // len(remaining))
        progressed = False
        for index in list(remaining):
            needed = values[index] - allocations[index]
            grant = min(needed, share, budget)
            allocations[index] += grant
            budget -= grant
            progressed = progressed or grant > 0
            if allocations[index] >= values[index]:
                remaining.remove(index)
            if budget <= 0:
                break
        if not progressed:
            break
    return allocations


def rebalance_packets(packets: list[EvidencePacket], total_limit: int, per_item_limit: int) -> list[EvidencePacket]:
    limits = balanced_character_limits(
        [len(packet.content) for packet in packets],
        total_limit=total_limit,
        per_item_limit=per_item_limit,
    )
    for packet, limit in zip(packets, limits):
        content, truncated = truncate_evidence(packet.content, limit)
        packet.content = content
        packet.truncated = packet.truncated or truncated
        if truncated and "aggregate_budget_reduction" not in packet.warnings:
            packet.warnings.append("aggregate_budget_reduction")
    return packets


class PaperContentExtractor:
    def __init__(self, settings: ExtractionSettings):
        self.settings = settings

    async def extract(self, papers: list[Paper]) -> list[EvidencePacket]:
        candidates = []
        for index, paper in enumerate(papers, 1):
            item_id = f"p{index:02d}"
            if not self.settings.enabled or not getattr(paper, "pdf_url", None):
                warning = "fulltext_disabled" if not self.settings.enabled else "missing_pdf_url"
                candidates.append(
                    _ExtractionCandidate(paper=paper, item_id=item_id, warnings=[warning])
                )
                continue
            candidates.append(await self._extract_pdf_candidate(paper, item_id))

        tex_candidates = [
            candidate
            for candidate in candidates
            if self.settings.tex_enabled
            and getattr(candidate.paper, "arxiv_id", None)
            and candidate.quality is not None
            and candidate.quality.decision != "accept_pdf_markdown"
        ]
        tex_candidates.sort(
            key=lambda candidate: (
                0 if candidate.quality and candidate.quality.hard_fail_reasons else 1,
                candidate.quality.score if candidate.quality else 101,
                candidate.item_id,
            )
        )
        tex_content: dict[str, str] = {}
        for candidate in tex_candidates[: max(0, self.settings.tex_max_papers)]:
            try:
                extracted = await self._extract_tex_source(str(candidate.paper.arxiv_id))
            except Exception as error:
                candidate.warnings.append(f"tex_fallback:{type(error).__name__}")
                continue
            if extracted:
                tex_content[candidate.item_id] = extracted

        packets = []
        for candidate in candidates:
            source = "abstract"
            content = ""
            if candidate.item_id in tex_content:
                source = "tex"
                content = tex_content[candidate.item_id]
            elif candidate.quality and candidate.quality.decision == "accept_pdf_markdown" and candidate.markdown:
                source = "pdf_markdown"
                content = candidate.markdown
            elif candidate.plain_text:
                source = "pdf_plain_text"
                content = candidate.plain_text
                candidate.warnings.append("markdown_quality_fallback")
            elif candidate.markdown:
                source = "pdf_markdown_low_quality"
                content = candidate.markdown
                candidate.warnings.append("plain_text_unavailable")
            else:
                content = _fallback_content(candidate.paper)
                candidate.warnings.append("abstract_fallback")

            original_chars = len(content)
            content, truncated = truncate_evidence(content, self.settings.per_paper_chars)
            packet = EvidencePacket(
                item_id=candidate.item_id,
                title=candidate.paper.title,
                url=candidate.paper.url,
                arxiv_id=str(getattr(candidate.paper, "arxiv_id", "") or ""),
                semantic_paper_id=str(getattr(candidate.paper, "semantic_paper_id", "") or ""),
                source=_source_name(candidate.paper),
                abstract=str(getattr(candidate.paper, "abstract", "") or ""),
                research_notes=str(getattr(candidate.paper, "research_notes", "") or ""),
                content=content,
                extraction_source=source,
                authors=[author.name for author in getattr(candidate.paper, "authors", [])],
                total_pages=candidate.total_pages,
                processed_pages=candidate.processed_pages,
                original_chars=original_chars,
                truncated=truncated,
                warnings=list(dict.fromkeys(candidate.warnings)),
                quality=candidate.quality,
                elapsed_seconds=candidate.elapsed_seconds,
            )
            packets.append(packet)
            score = packet.quality.score if packet.quality else "n/a"
            print(
                f"      {packet.item_id}: source={packet.extraction_source} "
                f"quality={score} chars={packet.content_chars} truncated={packet.truncated}"
            )
        return packets

    async def _extract_pdf_candidate(self, paper: Paper, item_id: str) -> _ExtractionCandidate:
        started = time.monotonic()
        candidate = _ExtractionCandidate(paper=paper, item_id=item_id)
        pdf_bytes = await self._download_bytes(
            str(paper.pdf_url),
            timeout_sec=self.settings.pdf_download_timeout_sec,
            retries=self.settings.pdf_download_retries,
            max_bytes=self.settings.pdf_max_bytes,
            expected_pdf=True,
        )
        if not pdf_bytes:
            candidate.warnings.append("pdf_download_failed")
            candidate.elapsed_seconds = time.monotonic() - started
            return candidate

        try:
            import pymupdf

            source_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            candidate.total_pages = len(source_document)
            page_limit = candidate.total_pages
            if self.settings.pdf_max_pages > 0:
                page_limit = min(page_limit, self.settings.pdf_max_pages)
            limited_document = pymupdf.open()
            if page_limit:
                limited_document.insert_pdf(source_document, from_page=0, to_page=page_limit - 1)
            source_document.close()
            candidate.processed_pages = len(limited_document)
            page_texts = [page.get_text() for page in limited_document]
            candidate.plain_text = _clean_text("\n\n".join(page_texts))
            candidate.markdown = self._extract_markdown(limited_document)
            layout_risk = _multi_column_page_ratio(limited_document)
            candidate.quality = evaluate_markdown_quality(
                title=paper.title,
                markdown=candidate.markdown,
                plain_text=candidate.plain_text,
                page_texts=page_texts,
                total_pages=candidate.total_pages,
                processed_pages=candidate.processed_pages,
                multi_column_ratio=layout_risk,
                settings=self.settings,
            )
            limited_document.close()
        except Exception as error:
            candidate.warnings.append(f"pdf_extract_failed:{type(error).__name__}")
        candidate.elapsed_seconds = time.monotonic() - started
        return candidate

    def _extract_markdown(self, document: Any) -> str:
        if self.settings.mode.lower() not in {"markdown", "auto"}:
            return ""
        try:
            import pymupdf4llm

            try:
                markdown = pymupdf4llm.to_markdown(
                    document,
                    header=False,
                    footer=False,
                    use_ocr=False,
                    ignore_images=True,
                    write_images=False,
                    page_separators=True,
                )
            except TypeError:
                markdown = pymupdf4llm.to_markdown(document)
            return _clean_text(str(markdown or ""))
        except Exception:
            return ""

    async def _download_bytes(
        self,
        url: str,
        *,
        timeout_sec: int,
        retries: int,
        max_bytes: int,
        expected_pdf: bool = False,
    ) -> Optional[bytes]:
        timeout = aiohttp.ClientTimeout(total=max(1, timeout_sec), connect=min(20, max(1, timeout_sec)))
        for attempt in range(max(0, retries) + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                    async with session.get(url) as response:
                        if response.status != 200:
                            raise ValueError(f"HTTP {response.status}")
                        declared_length = int(response.headers.get("content-length", "0") or 0)
                        if declared_length and declared_length > max_bytes:
                            raise ValueError("download exceeds size limit")
                        chunks = []
                        payload_size = 0
                        async for chunk in response.content.iter_chunked(64 * 1024):
                            payload_size += len(chunk)
                            if payload_size > max_bytes:
                                raise ValueError("download exceeds size limit")
                            chunks.append(chunk)
                        payload = b"".join(chunks)
                        if expected_pdf and not payload.startswith(b"%PDF"):
                            raise ValueError("download is not a PDF")
                        return payload
            except Exception:
                if attempt >= max(0, retries):
                    return None
                delay = min(8.0, 1.5 * (2**attempt)) + random.uniform(0, 0.25)
                await asyncio.sleep(delay)
        return None

    async def _extract_tex_source(self, arxiv_id: str) -> str:
        normalized_id = re.sub(r"(?i)^arxiv:", "", arxiv_id).strip()
        payload = await self._download_bytes(
            f"https://export.arxiv.org/e-print/{quote(normalized_id, safe='/')}",
            timeout_sec=self.settings.tex_download_timeout_sec,
            retries=1,
            max_bytes=self.settings.tex_archive_max_bytes,
        )
        if not payload:
            return ""
        files = self._read_safe_tex_archive(payload)
        if not files:
            return ""
        main_name = self._select_main_tex(files)
        resolved = self._resolve_tex_includes(main_name, files, depth=0, visited=set())
        return _clean_text(self._tex_to_markdown(resolved))

    def _read_safe_tex_archive(self, payload: bytes) -> dict[str, str]:
        files: dict[str, str] = {}
        expanded_size = 0
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            members = archive.getmembers()
            if len(members) > self.settings.tex_max_files:
                raise ValueError("tex archive file count exceeded")
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("unsafe tex archive path")
                if member.issym() or member.islnk():
                    raise ValueError("tex archive links are not allowed")
                if not member.isfile() or path.suffix.lower() not in ALLOWED_TEX_SUFFIXES:
                    continue
                if member.size > self.settings.tex_file_max_bytes:
                    raise ValueError("tex source file size exceeded")
                expanded_size += member.size
                if expanded_size > self.settings.tex_expanded_max_bytes:
                    raise ValueError("tex archive expanded size exceeded")
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                raw = extracted.read(self.settings.tex_file_max_bytes + 1)
                if len(raw) > self.settings.tex_file_max_bytes:
                    raise ValueError("tex source file size exceeded")
                files[str(path)] = raw.decode("utf-8", errors="replace")
        return files

    @staticmethod
    def _select_main_tex(files: dict[str, str]) -> str:
        tex_files = {name: value for name, value in files.items() if name.lower().endswith(".tex")}
        if not tex_files:
            return max(files, key=lambda name: len(files[name]))
        document_files = [
            name for name, value in tex_files.items() if "\\begin{document}" in value
        ]
        candidates = document_files or list(tex_files)
        return max(candidates, key=lambda name: len(tex_files[name]))

    def _resolve_tex_includes(
        self,
        name: str,
        files: dict[str, str],
        *,
        depth: int,
        visited: set[str],
    ) -> str:
        if depth > self.settings.tex_include_max_depth:
            raise ValueError("tex include depth exceeded")
        normalized_name = str(PurePosixPath(name))
        if normalized_name in visited:
            return ""
        if normalized_name not in files:
            return ""
        visited.add(normalized_name)
        current_dir = PurePosixPath(normalized_name).parent
        content = self._strip_tex_comments(files[normalized_name])

        def replace_include(match: re.Match[str]) -> str:
            raw_target = match.group(1).strip()
            target = PurePosixPath(raw_target)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError("unsafe tex include path")
            combined = current_dir / target
            if combined.suffix.lower() not in ALLOWED_TEX_SUFFIXES:
                combined = combined.with_suffix(".tex")
            return self._resolve_tex_includes(
                str(combined),
                files,
                depth=depth + 1,
                visited=visited,
            )

        return re.sub(r"\\(?:input|include)\s*\{([^}]+)\}", replace_include, content)

    @staticmethod
    def _strip_tex_comments(content: str) -> str:
        output = []
        for line in content.splitlines():
            output.append(re.split(r"(?<!\\)%", line, maxsplit=1)[0])
        return "\n".join(output)

    @staticmethod
    def _tex_to_markdown(content: str) -> str:
        content = re.sub(r"\\(?:documentclass|usepackage)(?:\[[^]]*\])?\{[^}]*\}", "", content)
        content = re.sub(r"\\begin\{document\}|\\end\{document\}", "", content)
        content = re.sub(r"\\begin\{abstract\}", "\n## Abstract\n", content)
        content = re.sub(r"\\end\{abstract\}", "\n", content)
        for command, prefix in (("section", "#"), ("subsection", "##"), ("subsubsection", "###")):
            content = re.sub(rf"\\{command}\*?\{{([^}}]+)\}}", rf"\n{prefix} \1\n", content)
        content = re.sub(r"\\(?:label|bibliographystyle|bibliography)\{[^}]*\}", "", content)
        content = re.sub(r"\\(?:emph|textit|textbf|texttt)\{([^{}]*)\}", r"\1", content)
        return content


def write_extraction_report(
    packets: list[EvidencePacket],
    *,
    output_dir: str = "artifacts",
    run_id: str,
    synthesis_mode: str,
    aggregate_threshold: int,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"extraction_quality_{run_id}.json"
    payload = {
        "version": "v1",
        "run_id": run_id,
        "synthesis_mode": synthesis_mode,
        "aggregate_threshold": aggregate_threshold,
        "aggregate_content_chars": sum(packet.content_chars for packet in packets),
        "papers": [packet.diagnostic_dict() for packet in packets],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
