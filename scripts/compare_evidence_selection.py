"""Capture a fixed PDF cohort and compare selectors locally, without LLM calls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence_selection import SelectionSettings, main_body, segment, select_evidence
from models import Paper, PaperSource
from paper_extraction import ExtractionSettings, PaperContentExtractor, truncate_evidence


async def compare(manifest: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(manifest.read_text())["papers"]
    summaries = []
    for entry in cohort:
        item_id = entry["item_id"]
        cache_path = output_dir / f"{item_id}.source.json"
        if cache_path.exists():
            source = json.loads(cache_path.read_text())
        else:
            arxiv_id = entry.get("arxiv_id")
            if not arxiv_id:
                summaries.append({"item_id": item_id, "status": "no_arxiv_pdf", "url": entry["url"]})
                continue
            extractor = PaperContentExtractor(ExtractionSettings())
            download = extractor._download_bytes
            pdf_path = output_dir / f"{item_id}.pdf"

            async def cached_download(url, **kwargs):
                if pdf_path.exists():
                    return pdf_path.read_bytes()
                payload = await download(url, **kwargs)
                if payload:
                    pdf_path.write_bytes(payload)
                return payload

            extractor._download_bytes = cached_download
            candidate = await extractor._extract_pdf_candidate(Paper(
                title=entry["title"], abstract="", url=entry["url"],
                arxiv_id=arxiv_id, pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                source=PaperSource.ARXIV,
            ), item_id)
            source = {
                "item_id": item_id, "arxiv_id": arxiv_id, "url": entry["url"],
                "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest() if pdf_path.exists() else None,
                "total_pages": candidate.total_pages, "processed_pages": candidate.processed_pages,
                "markdown": candidate.markdown, "plain_text": candidate.plain_text,
                "quality": asdict(candidate.quality) if candidate.quality else None,
                "warnings": candidate.warnings,
            }
            cache_path.write_text(json.dumps(source, ensure_ascii=False, indent=2))
        text = source["markdown"] if source["quality"] and source["quality"]["decision"] == "accept_pdf_markdown" else source["plain_text"]
        if not text:
            summaries.append({"item_id": item_id, "status": "pdf_unavailable", "warnings": source["warnings"]})
            print(f"{item_id}: PDF unavailable", flush=True)
            continue
        old, _ = truncate_evidence(re.sub(r"(?m)^<!-- evidence-page:\d+ -->\s*", "", text), 18000)
        new, diagnostics = select_evidence(text, 18000, SelectionSettings(),
                                          total_pages=source["total_pages"], processed_pages=source["processed_pages"])
        (output_dir / f"{item_id}.old.md").write_text(old)
        (output_dir / f"{item_id}.new.md").write_text(new)
        body, _ = main_body(segment(text)[0])
        probes = []
        for section in body:
            blocks = [block.strip() for block in re.split(r"\n\s*\n", section.content) if len(block.strip()) >= 100]
            probes.append({
                "heading": section.title, "role": section.role, "probe_count": len(blocks),
                "old_block_prefix_hits": sum(block[:80] in old for block in blocks),
                "new_block_prefix_hits": sum(block[:80] in new for block in blocks),
            })
        summary = {"item_id": item_id, "status": "compared", "arxiv_id": source["arxiv_id"],
                   "pdf_sha256": source["pdf_sha256"], "old_chars": len(old), "new_chars": len(new),
                   "selection": diagnostics, "block_prefix_probes": probes}
        summaries.append(summary)
        print(f"{item_id}: old={len(old)} new={len(new)} sections={diagnostics['section_count']} boundary={diagnostics['boundary_confidence']}", flush=True)
    (output_dir / "comparison.json").write_text(json.dumps({
        "cohort_manifest": str(manifest), "measurement": "block-prefix coverage heuristic, not evidence completeness or token usage",
        "papers": summaries,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(compare(arguments.manifest, arguments.output_dir))
