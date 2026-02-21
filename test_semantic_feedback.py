from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models import Paper, PaperSource
from semantic_feedback import apply_feedback_to_seeds, export_run_feedback_manifest


class SemanticFeedbackTests(unittest.TestCase):
    def test_export_manifest_contains_report_visible_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            papers = [
                Paper(
                    title="Visible Paper",
                    abstract="a",
                    url="https://arxiv.org/abs/2501.00001",
                    source=PaperSource.SEMANTIC_SCHOLAR,
                    semantic_paper_id="123456",
                ),
                Paper(
                    title="Hidden Paper",
                    abstract="b",
                    url="https://arxiv.org/abs/2501.00002",
                    source=PaperSource.SEMANTIC_SCHOLAR,
                    semantic_paper_id="CorpusId:999999",
                ),
            ]
            html = '<a href="https://arxiv.org/abs/2501.00001">Visible Paper</a>'
            out = export_run_feedback_manifest(papers, html, output_dir=td, run_id="2026-02-21T08-00-00Z")

            self.assertIsNotNone(out)
            manifest_path, questionnaire_path = out
            data = json.loads(Path(manifest_path).read_text())
            questionnaire = json.loads(Path(questionnaire_path).read_text())
            self.assertEqual(data["run_id"], "2026-02-21T08-00-00Z")
            self.assertEqual(len(data["papers"]), 1)
            self.assertEqual(data["papers"][0]["item_id"], "p01")
            self.assertEqual(data["papers"][0]["semantic_paper_id"], "CorpusId:123456")
            self.assertEqual(questionnaire["run_id"], "2026-02-21T08-00-00Z")
            self.assertEqual(questionnaire["labels"][0]["item_id"], "p01")
            self.assertEqual(questionnaire["labels"][0]["label"], "undecided")

    def test_apply_feedback_latest_review_wins_and_moves_between_lists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            manifest = td_path / "manifest.json"
            feedback = td_path / "feedback.json"
            seeds = td_path / "seeds.json"

            manifest.write_text(
                json.dumps(
                    {
                        "version": "v1",
                        "run_id": "2026-02-21T08-00-00Z",
                        "generated_at": "2026-02-21T08:10:00Z",
                        "papers": [
                            {
                                "item_id": "p01",
                                "title": "A",
                                "url": "https://arxiv.org/abs/1",
                                "semantic_paper_id": "CorpusId:111",
                            },
                            {
                                "item_id": "p02",
                                "title": "B",
                                "url": "https://arxiv.org/abs/2",
                                "semantic_paper_id": "222",
                            },
                        ],
                    }
                )
                + "\n"
            )
            feedback.write_text(
                json.dumps(
                    {
                        "version": "v1",
                        "run_id": "2026-02-21T08-00-00Z",
                        "reviewer": "u",
                        "reviewed_at": "2026-02-21T09:00:00Z",
                        "labels": [
                            {"item_id": "p01", "label": "positive", "reviewed_at": "2026-02-21T09:01:00Z"},
                            {"item_id": "p01", "label": "negative", "reviewed_at": "2026-02-21T09:02:00Z"},
                            {"item_id": "p02", "label": "positive"},
                        ],
                    }
                )
                + "\n"
            )
            seeds.write_text(
                json.dumps(
                    {
                        "positive_paper_ids": ["CorpusId:111"],
                        "negative_paper_ids": [],
                    }
                )
                + "\n"
            )

            result = apply_feedback_to_seeds(
                feedback_path=str(feedback),
                manifest_path=str(manifest),
                seeds_path=str(seeds),
                dry_run=False,
            )
            self.assertEqual(result["applied_count"], 2)

            updated = json.loads(seeds.read_text())
            self.assertEqual(updated["positive_paper_ids"], ["CorpusId:222"])
            self.assertEqual(updated["negative_paper_ids"], ["CorpusId:111"])

    def test_apply_feedback_rejects_run_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            manifest = td_path / "manifest.json"
            feedback = td_path / "feedback.json"

            manifest.write_text(
                json.dumps(
                    {
                        "version": "v1",
                        "run_id": "run-a",
                        "generated_at": "2026-02-21T08:10:00Z",
                        "papers": [],
                    }
                )
                + "\n"
            )
            feedback.write_text(
                json.dumps(
                    {
                        "version": "v1",
                        "run_id": "run-b",
                        "reviewer": "u",
                        "reviewed_at": "2026-02-21T09:00:00Z",
                        "labels": [],
                    }
                )
                + "\n"
            )

            with self.assertRaises(ValueError):
                apply_feedback_to_seeds(
                    feedback_path=str(feedback),
                    manifest_path=str(manifest),
                    seeds_path=str(td_path / "seeds.json"),
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
