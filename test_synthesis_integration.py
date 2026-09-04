from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from config import Config
from main import run_pipeline
from models import Paper, PaperSource
from summarizer import SynthesisError


class SynthesisConfigTests(unittest.TestCase):
    def test_structured_defaults_load_from_yaml(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_yaml("config.yaml")
        self.assertEqual(config.synthesis_mode, "structured")
        self.assertEqual(config.paper_evidence_chars, 18000)
        self.assertEqual(config.synthesis_aggregate_chars, 180000)
        self.assertEqual(config.extraction_quality_threshold, 70)
        self.assertFalse(config.tex_source_enabled)
        self.assertTrue(config.synthesis_streaming)

    def test_environment_overrides_parse_types_and_ignore_empty_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "synthesis_aggregate_chars: 180000\n"
                "tex_source_enabled: false\n"
                "extraction_quality_max_unreadable_ratio: 0.02\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "SYNTHESIS_AGGREGATE_CHARS": "90000",
                    "TEX_SOURCE_ENABLED": "true",
                    "EXTRACTION_QUALITY_MAX_UNREADABLE_RATIO": "0.05",
                    "PAPER_EXTRACTION_MODE": "",
                },
                clear=True,
            ):
                config = Config.from_yaml(str(path))
        self.assertEqual(config.synthesis_aggregate_chars, 90000)
        self.assertTrue(config.tex_source_enabled)
        self.assertEqual(config.extraction_quality_max_unreadable_ratio, 0.05)
        self.assertEqual(config.paper_extraction_mode, "markdown")


class DegradedPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_synthesis_failure_skips_all_persistent_state(self) -> None:
        paper = Paper(
            title="Retry Tomorrow",
            abstract="evidence",
            url="https://arxiv.org/abs/2601.00001",
            source=PaperSource.ARXIV,
        )
        config = SimpleNamespace(
            papers_enabled=True,
            email_to="owner@example.com",
            synthesis_failure_notification=True,
        )
        with (
            patch("main.Config.from_yaml", return_value=config),
            patch("main.fetch_papers", new=AsyncMock(return_value=[paper])),
            patch("main.fetch_blogs", new=AsyncMock(return_value=([], []))),
            patch("main.filter_papers_coarse", new=AsyncMock(return_value=[paper])),
            patch("main.enrich_papers", new=AsyncMock(return_value=[paper])),
            patch("main.filter_papers_fine", new=AsyncMock(return_value=[paper])),
            patch("main.summarize_papers", new=AsyncMock(side_effect=SynthesisError("timeout"))),
            patch("main.send_synthesis_failure_notification", new=AsyncMock(return_value=True)) as notify,
            patch("main.export_run_feedback_manifest") as export_manifest,
            patch("main.publish_feedback_run_to_d1") as publish_d1,
            patch("main.update_semantic_memory_from_report") as update_memory,
            patch("main.send_email", new=AsyncMock()) as send_digest,
        ):
            await run_pipeline(config_path="unused.yaml", days_back=1, dry_run=False)

        notify.assert_awaited_once()
        export_manifest.assert_not_called()
        publish_d1.assert_not_called()
        update_memory.assert_not_called()
        send_digest.assert_not_awaited()

    async def test_dry_run_skips_d1_and_semantic_memory(self) -> None:
        paper = Paper(
            title="Dry Run Paper",
            abstract="evidence",
            url="https://arxiv.org/abs/2601.00002",
            source=PaperSource.ARXIV,
        )
        config = SimpleNamespace(
            papers_enabled=True,
            email_to="owner@example.com",
        )
        file_emailer = SimpleNamespace(send=AsyncMock(return_value=True))
        with (
            patch("main.Config.from_yaml", return_value=config),
            patch("main.fetch_papers", new=AsyncMock(return_value=[paper])),
            patch("main.fetch_blogs", new=AsyncMock(return_value=([], []))),
            patch("main.filter_papers_coarse", new=AsyncMock(return_value=[paper])),
            patch("main.enrich_papers", new=AsyncMock(return_value=[paper])),
            patch("main.filter_papers_fine", new=AsyncMock(return_value=[paper])),
            patch("main.summarize_papers", new=AsyncMock(return_value=f'<a href="{paper.url}">Paper</a>')),
            patch("main.export_run_feedback_manifest", return_value=(Path("manifest.json"), Path("template.json"))),
            patch("main.inject_feedback_actions_into_report", return_value="web report"),
            patch("main.get_run_id_from_manifest", return_value="run-test"),
            patch("main.build_feedback_run_view_url", return_value="https://feedback.example/run"),
            patch("main.publish_feedback_run_to_d1") as publish_d1,
            patch("main.update_semantic_memory_from_report") as update_memory,
            patch("main.FileEmailer", return_value=file_emailer),
            patch("main.send_email", new=AsyncMock()) as send_digest,
        ):
            await run_pipeline(config_path="unused.yaml", days_back=1, dry_run=True)

        publish_d1.assert_not_called()
        update_memory.assert_not_called()
        send_digest.assert_not_awaited()
        file_emailer.send.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
