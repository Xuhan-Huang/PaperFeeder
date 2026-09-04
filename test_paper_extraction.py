from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pymupdf

from models import Paper, PaperSource
from paper_extraction import (
    EvidencePacket,
    ExtractionQuality,
    ExtractionSettings,
    PaperContentExtractor,
    balanced_character_limits,
    evaluate_markdown_quality,
    truncate_evidence,
    write_extraction_report,
)


def build_pdf_bytes() -> bytes:
    document = pymupdf.open()
    sections = (
        ("Reliable Paper Synthesis", "Abstract", "This paper studies reliable synthesis systems."),
        ("Method", "Method", "We use deterministic extraction and bounded evidence packets."),
        ("Results", "Results", "The method reduces request size and timeout failures."),
        ("Conclusion", "Conclusion", "Structured text is reliable for holistic synthesis."),
    )
    for title, section, body in sections:
        page = document.new_page()
        lines = [title, section]
        lines.extend(f"{body} Observation {index} provides distinct evidence." for index in range(45))
        page.insert_text((40, 50), "\n".join(lines), fontsize=8, lineheight=1.1)
    payload = document.tobytes(deflate=True)
    document.close()
    return payload


def build_tar(entries: dict[str, bytes], *, symlink: str = "") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if symlink:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = "main.tex"
            archive.addfile(info)
    return buffer.getvalue()


class ExtractionQualityTests(unittest.TestCase):
    def test_default_ten_paper_budget_stays_in_direct_mode(self) -> None:
        settings = ExtractionSettings()
        self.assertEqual(settings.per_paper_chars, 18000)
        self.assertLessEqual(settings.per_paper_chars * 10, settings.aggregate_chars)

    def test_quality_accepts_clean_markdown(self) -> None:
        page_texts = [
            "Reliable Paper Synthesis\nAbstract\n" + "Distinct evidence text. " * 80,
            "Method\n" + "Method detail with measurable result. " * 80,
            "Conclusion\n" + "Concluding evidence. " * 80,
        ]
        plain_text = "\n".join(page_texts)
        markdown = "# Reliable Paper Synthesis\n\n## Abstract\n" + plain_text
        quality = evaluate_markdown_quality(
            title="Reliable Paper Synthesis",
            markdown=markdown,
            plain_text=plain_text,
            page_texts=page_texts,
            total_pages=3,
            processed_pages=3,
            multi_column_ratio=0.0,
            settings=ExtractionSettings(),
        )
        self.assertGreaterEqual(quality.score, 70)
        self.assertEqual(quality.decision, "accept_pdf_markdown")
        self.assertEqual(quality.hard_fail_reasons, [])

    def test_quality_hard_fails_tiny_markdown(self) -> None:
        quality = evaluate_markdown_quality(
            title="Paper",
            markdown="tiny",
            plain_text="useful text " * 300,
            page_texts=["useful text " * 300],
            total_pages=1,
            processed_pages=1,
            multi_column_ratio=0.0,
            settings=ExtractionSettings(),
        )
        self.assertEqual(quality.decision, "hard_fail")
        self.assertIn("empty_or_tiny_markdown", quality.hard_fail_reasons)

    def test_quality_threshold_is_configurable(self) -> None:
        page_texts = ["Paper Title\nAbstract\nMethod\nResult\n" + "unique content " * 100]
        plain_text = page_texts[0]
        quality = evaluate_markdown_quality(
            title="Paper Title",
            markdown=plain_text,
            plain_text=plain_text,
            page_texts=page_texts,
            total_pages=1,
            processed_pages=1,
            multi_column_ratio=0.0,
            settings=ExtractionSettings(quality_threshold=101),
        )
        self.assertEqual(quality.decision, "try_alternative")

    def test_balanced_limits_preserve_short_items_and_equalize_long_items(self) -> None:
        allocations = balanced_character_limits([100, 100, 10], total_limit=110, per_item_limit=100)
        self.assertEqual(sum(allocations), 110)
        self.assertEqual(allocations[0], allocations[1])
        self.assertEqual(allocations[2], 10)

    def test_truncate_removes_repeated_and_reference_tail(self) -> None:
        repeated = "A distinct repeated header with enough characters"
        content = f"# Main\n{repeated}\nBody evidence\n{repeated}\n# References\nCitation " * 20
        truncated, changed = truncate_evidence(content, 500)
        self.assertTrue(changed)
        self.assertLessEqual(len(truncated), 500)
        self.assertLessEqual(truncated.count(repeated), 1)


class PaperContentExtractorTests(unittest.IsolatedAsyncioTestCase):
    async def test_markdown_packet_uses_bounded_runner_local_content(self) -> None:
        paper = Paper(
            title="Reliable Paper Synthesis",
            abstract="abstract",
            url="https://arxiv.org/abs/2601.00001",
            pdf_url="https://arxiv.org/pdf/2601.00001.pdf",
            arxiv_id="2601.00001",
            source=PaperSource.ARXIV,
        )
        extractor = PaperContentExtractor(ExtractionSettings(quality_threshold=0, per_paper_chars=5000))
        with patch.object(extractor, "_download_bytes", new=AsyncMock(return_value=build_pdf_bytes())):
            packets = await extractor.extract([paper])
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0].item_id, "p01")
        self.assertEqual(packets[0].extraction_source, "pdf_markdown")
        self.assertLessEqual(packets[0].content_chars, 5000)
        self.assertNotIn("base64", packets[0].diagnostic_dict())

    async def test_markdown_failure_falls_back_to_plain_text(self) -> None:
        paper = Paper(
            title="Fallback Paper",
            abstract="abstract",
            url="https://arxiv.org/abs/2601.00002",
            pdf_url="https://arxiv.org/pdf/2601.00002.pdf",
            source=PaperSource.ARXIV,
        )
        extractor = PaperContentExtractor(ExtractionSettings())
        with (
            patch.object(extractor, "_download_bytes", new=AsyncMock(return_value=build_pdf_bytes())),
            patch.object(extractor, "_extract_markdown", return_value=""),
        ):
            packet = (await extractor.extract([paper]))[0]
        self.assertEqual(packet.extraction_source, "pdf_plain_text")
        self.assertIn("markdown_quality_fallback", packet.warnings)

    async def test_missing_pdf_uses_abstract_fallback(self) -> None:
        paper = Paper(
            title="Abstract Paper",
            abstract="Important abstract evidence.",
            url="https://example.com/paper",
            source=PaperSource.MANUAL,
        )
        packet = (await PaperContentExtractor(ExtractionSettings()).extract([paper]))[0]
        self.assertEqual(packet.extraction_source, "abstract")
        self.assertIn("Important abstract evidence", packet.content)
        self.assertIn("missing_pdf_url", packet.warnings)

    async def test_tex_replaces_low_quality_pdf_within_budget(self) -> None:
        paper = Paper(
            title="TeX Paper",
            abstract="abstract",
            url="https://arxiv.org/abs/2601.00003",
            pdf_url="https://arxiv.org/pdf/2601.00003.pdf",
            arxiv_id="2601.00003",
            source=PaperSource.ARXIV,
        )
        settings = ExtractionSettings(tex_enabled=True, tex_max_papers=1, per_paper_chars=500)
        extractor = PaperContentExtractor(settings)
        low_quality = ExtractionQuality(score=20, metrics={}, decision="try_alternative")
        from paper_extraction import _ExtractionCandidate

        candidate = _ExtractionCandidate(
            paper=paper,
            item_id="p01",
            markdown="bad markdown",
            plain_text="plain text",
            quality=low_quality,
        )
        with (
            patch.object(extractor, "_extract_pdf_candidate", new=AsyncMock(return_value=candidate)),
            patch.object(extractor, "_extract_tex_source", new=AsyncMock(return_value="TeX evidence " * 100)),
        ):
            packet = (await extractor.extract([paper]))[0]
        self.assertEqual(packet.extraction_source, "tex")
        self.assertLessEqual(packet.content_chars, 500)

    def test_diagnostic_report_excludes_full_content(self) -> None:
        packet = EvidencePacket(
            item_id="p01",
            title="Paper",
            url="https://example.com/paper",
            arxiv_id="",
            semantic_paper_id="",
            source="manual",
            abstract="secret abstract",
            research_notes="secret notes",
            content="FULL_PRIVATE_CONTENT signed-feedback-token",
            extraction_source="abstract",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_extraction_report(
                [packet],
                output_dir=directory,
                run_id="run-test",
                synthesis_mode="direct",
                aggregate_threshold=180000,
            )
            raw = path.read_text(encoding="utf-8")
        self.assertNotIn("FULL_PRIVATE_CONTENT", raw)
        self.assertNotIn("signed-feedback-token", raw)
        self.assertNotIn("secret abstract", raw)


class TeXArchiveSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = PaperContentExtractor(ExtractionSettings())

    def test_safe_tex_archive_and_include_resolution(self) -> None:
        payload = build_tar(
            {
                "main.tex": b"\\documentclass{article}\\begin{document}\\section{Intro}\\input{parts/method}\\end{document}",
                "parts/method.tex": b"\\section{Method}Safe method content.",
            }
        )
        files = self.extractor._read_safe_tex_archive(payload)
        resolved = self.extractor._resolve_tex_includes("main.tex", files, depth=0, visited=set())
        markdown = self.extractor._tex_to_markdown(resolved)
        self.assertIn("# Intro", markdown)
        self.assertIn("# Method", markdown)

    def test_parent_traversal_is_rejected(self) -> None:
        payload = build_tar({"../escape.tex": b"bad"})
        with self.assertRaisesRegex(ValueError, "unsafe tex archive path"):
            self.extractor._read_safe_tex_archive(payload)

    def test_archive_symlink_is_rejected(self) -> None:
        payload = build_tar({"main.tex": b"safe"}, symlink="link.tex")
        with self.assertRaisesRegex(ValueError, "links are not allowed"):
            self.extractor._read_safe_tex_archive(payload)

    def test_file_count_limit_is_enforced(self) -> None:
        extractor = PaperContentExtractor(ExtractionSettings(tex_max_files=1))
        payload = build_tar({"a.tex": b"a", "b.tex": b"b"})
        with self.assertRaisesRegex(ValueError, "file count exceeded"):
            extractor._read_safe_tex_archive(payload)

    def test_expanded_size_limit_is_enforced(self) -> None:
        extractor = PaperContentExtractor(ExtractionSettings(tex_expanded_max_bytes=5))
        payload = build_tar({"a.tex": b"123456"})
        with self.assertRaisesRegex(ValueError, "expanded size exceeded"):
            extractor._read_safe_tex_archive(payload)

    def test_file_size_limit_is_enforced(self) -> None:
        extractor = PaperContentExtractor(ExtractionSettings(tex_file_max_bytes=5))
        payload = build_tar({"a.tex": b"123456"})
        with self.assertRaisesRegex(ValueError, "file size exceeded"):
            extractor._read_safe_tex_archive(payload)

    def test_unsafe_include_is_rejected(self) -> None:
        files = {"main.tex": "\\input{../outside}"}
        with self.assertRaisesRegex(ValueError, "unsafe tex include path"):
            self.extractor._resolve_tex_includes("main.tex", files, depth=0, visited=set())

    def test_recursive_include_terminates(self) -> None:
        files = {"main.tex": "start \\input{main} end"}
        resolved = self.extractor._resolve_tex_includes("main.tex", files, depth=0, visited=set())
        self.assertEqual(resolved, "start  end")


if __name__ == "__main__":
    unittest.main()
