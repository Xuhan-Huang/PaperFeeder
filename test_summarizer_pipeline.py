from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llm_client import LLMClient
from main import update_semantic_memory_from_report
from models import Paper, PaperSource
from paper_extraction import EvidencePacket, ExtractionSettings
from semantic_feedback import export_run_feedback_manifest
from semantic_memory import SemanticMemoryStore
from summarizer import (
    EDITORIAL_SYSTEM_PROMPT,
    PaperSummarizer,
    SynthesisError,
    SynthesisValidationError,
)


def make_paper(index: int = 1, abstract: str = "Useful evidence.") -> Paper:
    return Paper(
        title=f"Paper {index}",
        abstract=abstract,
        url=f"https://arxiv.org/abs/2601.{index:05d}",
        source=PaperSource.ARXIV,
        arxiv_id=f"2601.{index:05d}",
        semantic_paper_id=f"CorpusId:{index}",
    )


def valid_payload() -> dict:
    return {
        "blog_highlights": [],
        "editors_choice": [
            {"item_id": "p01", "verdict": "Strong result.", "signal": "N/A", "badge": "high"}
        ],
        "deep_dives": [
            {
                "item_id": "p01",
                "kind": "paper",
                "aha": "A counter-intuitive result.",
                "methodology": "A concrete method.",
                "reality_check": "Evidence with caveats.",
                "my_take": "Reproduce it.",
            }
        ],
        "worth_skimming": [],
    }


class StructuredSynthesisTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_mode_preserves_prompt_and_canonical_url(self) -> None:
        paper = make_paper()
        with tempfile.TemporaryDirectory() as directory:
            summarizer = PaperSummarizer(
                api_key="test",
                base_url="https://example.com/v1",
                model="anthropic/claude-opus-5",
                research_interests="Reliability",
                extraction_settings=ExtractionSettings(enabled=False, aggregate_chars=180000),
                synthesis_streaming=True,
                diagnostic_output_dir=directory,
            )
            summarizer.client.achat_stream = AsyncMock(return_value=json.dumps(valid_payload()))
            report = await summarizer.generate_report([paper], use_pdf_multimodal=False)

            self.assertEqual(summarizer.last_synthesis_mode, "direct")
            self.assertIn(f'href="{paper.url}"', report)
            self.assertNotIn("paper:p01", report)
            messages = summarizer.client.achat_stream.await_args.args[0]
            self.assertEqual(messages[0]["content"], EDITORIAL_SYSTEM_PROMPT)
            self.assertIsInstance(messages[1]["content"], str)
            self.assertIn('<document id="p01"', messages[1]["content"])
            self.assertNotIn('"type": "document"', messages[1]["content"])
            self.assertTrue(summarizer.last_extraction_report_path.exists())

    async def test_adaptive_mode_uses_facts_then_one_holistic_prompt(self) -> None:
        paper = make_paper(1, abstract="Long evidence " * 100)
        fact = {
            "item_id": "p01",
            "canonical_url": paper.url,
            "abstract": paper.abstract[:100],
            "core_claim": "claim",
            "method": "method",
            "evidence": ["evidence"],
            "limitations": ["limit"],
            "surprising_points": ["surprise"],
            "selected_excerpts": ["excerpt"],
            "fallback": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            summarizer = PaperSummarizer(
                api_key="test",
                base_url="https://example.com/v1",
                model="anthropic/claude-opus-5",
                extraction_settings=ExtractionSettings(enabled=False, aggregate_chars=100),
                diagnostic_output_dir=directory,
            )
            summarizer._compact_packets = AsyncMock(return_value=[fact])
            summarizer.client.achat_stream = AsyncMock(return_value=json.dumps(valid_payload()))
            report = await summarizer.generate_report([paper], use_pdf_multimodal=False)

            self.assertEqual(summarizer.last_synthesis_mode, "adaptive")
            summarizer._compact_packets.assert_awaited_once()
            messages = summarizer.client.achat_stream.await_args.args[0]
            self.assertIn('compacted="true"', messages[1]["content"])
            self.assertEqual(messages[0]["content"], EDITORIAL_SYSTEM_PROMPT)
            self.assertIn(paper.url, report)

    async def test_adaptive_compaction_failure_uses_deterministic_fallback(self) -> None:
        paper = make_paper()
        packet = EvidencePacket(
            item_id="p01",
            title=paper.title,
            url=paper.url,
            arxiv_id=paper.arxiv_id or "",
            semantic_paper_id=paper.semantic_paper_id or "",
            source="arxiv",
            abstract=paper.abstract,
            research_notes="notes",
            content="bounded evidence",
            extraction_source="pdf_markdown",
        )
        summarizer = PaperSummarizer(
            api_key="test",
            base_url="https://example.com/v1",
            model="anthropic/claude-opus-5",
        )
        summarizer._call_with_retry = AsyncMock(side_effect=SynthesisError("timeout"))
        records = await summarizer._compact_packets([packet])
        self.assertTrue(records[0]["fallback"])
        self.assertEqual(records[0]["item_id"], "p01")
        self.assertEqual(records[0]["canonical_url"], paper.url)

    async def test_transient_error_is_retried(self) -> None:
        summarizer = PaperSummarizer(
            api_key="test",
            base_url="https://example.com/v1",
            model="anthropic/claude-opus-5",
            synthesis_retries=1,
            synthesis_retry_base_delay_sec=0,
        )
        summarizer.client.achat_stream = AsyncMock(side_effect=[TimeoutError("slow"), "ok"])
        with patch("summarizer.asyncio.sleep", new=AsyncMock()) as sleep:
            result = await summarizer._call_with_retry(
                [{"role": "user", "content": "test"}],
                max_tokens=10,
                purpose="test",
            )
        self.assertEqual(result, "ok")
        self.assertEqual(summarizer.client.achat_stream.await_count, 2)
        sleep.assert_awaited_once()

    async def test_authentication_error_is_not_retried(self) -> None:
        class AuthenticationError(Exception):
            pass

        summarizer = PaperSummarizer(
            api_key="test",
            base_url="https://example.com/v1",
            model="anthropic/claude-opus-5",
            synthesis_retries=2,
        )
        summarizer.client.achat_stream = AsyncMock(side_effect=AuthenticationError("bad key"))
        with self.assertRaises(SynthesisError):
            await summarizer._call_with_retry(
                [{"role": "user", "content": "test"}],
                max_tokens=10,
                purpose="test",
            )
        self.assertEqual(summarizer.client.achat_stream.await_count, 1)

    def test_unknown_model_item_id_is_rejected(self) -> None:
        summarizer = PaperSummarizer(api_key="test", base_url="https://example.com/v1")
        payload = valid_payload()
        payload["editors_choice"][0]["item_id"] = "p99"
        with self.assertRaisesRegex(SynthesisValidationError, "unknown paper item_id"):
            summarizer._validate_digest_payload(payload, {"p01": make_paper()}, {})

    def test_error_page_content_is_rejected(self) -> None:
        with self.assertRaises(SynthesisValidationError):
            PaperSummarizer._parse_json_object("Error generating report: Request timed out.")

    def test_missing_required_section_is_rejected(self) -> None:
        summarizer = PaperSummarizer(api_key="test", base_url="https://example.com/v1")
        payload = valid_payload()
        payload.pop("worth_skimming")
        with self.assertRaisesRegex(SynthesisValidationError, "worth_skimming"):
            summarizer._validate_digest_payload(payload, {"p01": make_paper()}, {})

    async def test_report_remains_compatible_with_feedback_manifest(self) -> None:
        paper = make_paper()
        with tempfile.TemporaryDirectory() as directory:
            summarizer = PaperSummarizer(
                api_key="test",
                base_url="https://example.com/v1",
                model="anthropic/claude-opus-5",
                extraction_settings=ExtractionSettings(enabled=False),
                diagnostic_output_dir=directory,
            )
            summarizer.client.achat_stream = AsyncMock(return_value=json.dumps(valid_payload()))
            report = await summarizer.generate_report([paper], use_pdf_multimodal=False)
            artifacts = export_run_feedback_manifest(
                [paper],
                report,
                output_dir=directory,
                run_id="run-test",
                resolver_enabled=False,
            )
            self.assertIsNotNone(artifacts)
            manifest = json.loads(Path(artifacts[0]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["papers"][0]["url"], paper.url)
            self.assertEqual(manifest["papers"][0]["semantic_paper_id"], "CorpusId:1")

    async def test_structured_report_drives_semantic_memory_visibility(self) -> None:
        paper = make_paper()
        with tempfile.TemporaryDirectory() as directory:
            summarizer = PaperSummarizer(
                api_key="test",
                base_url="https://example.com/v1",
                model="anthropic/claude-opus-5",
                extraction_settings=ExtractionSettings(enabled=False),
                diagnostic_output_dir=directory,
            )
            summarizer.client.achat_stream = AsyncMock(return_value=json.dumps(valid_payload()))
            report = await summarizer.generate_report([paper], use_pdf_multimodal=False)
            store = SemanticMemoryStore(str(Path(directory) / "memory.json"), max_ids=20)
            store.load()
            config = SimpleNamespace(
                semantic_memory_enabled=True,
                _semantic_memory_store=store,
                semantic_seen_ttl_days=30,
            )
            update_semantic_memory_from_report([paper], report, config)
            self.assertTrue(store.recently_seen("arxiv:2601.00001", ttl_days=30))
            self.assertTrue(store.recently_seen("semantic:CorpusId:1", ttl_days=30))


class StreamingClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_compatible_stream_accumulates_content(self) -> None:
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hel"))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]),
        ]

        class FakeStream:
            def __aiter__(self):
                self._iterator = iter(chunks)
                return self

            async def __anext__(self):
                try:
                    return next(self._iterator)
                except StopIteration as error:
                    raise StopAsyncIteration from error

        client = LLMClient(
            api_key="test",
            base_url="https://example.com/v1",
            model="anthropic/claude-opus-5",
            max_retries=0,
        )
        create = AsyncMock(return_value=FakeStream())
        client.async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        result = await client.achat_stream(
            [{"role": "user", "content": "test"}],
            max_tokens=10,
        )
        self.assertEqual(result, "hello")
        self.assertTrue(create.await_args.kwargs["stream"])


if __name__ == "__main__":
    unittest.main()
