from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch

from config import Config
from evidence_selection import SelectionSettings, allocate, heading, segment, select_evidence, selection_notes
from paper_extraction import EvidencePacket, ExtractionSettings, PaperContentExtractor
from summarizer import PaperSummarizer


def section(title: str, label: str, count: int = 12) -> str:
    return f"## {title}\n\n" + "\n\n".join(
        f"{label}-{index}: " + f"Evidence from {label} block {index}. " * 15
        for index in range(count)
    )


class SelectionTests(unittest.TestCase):
    def test_unknown_headings_keep_middle_and_baseline(self):
        content = "\n\n".join([
            section("1 Introduction", "intro"), section("2 SpiralNet", "spiral"),
            section("3 Why Does It Work?", "why"), section("4 Conclusion", "end"),
            "## References\n\nCitation entry.", "## Appendix A\n\nAPPENDIX_SECRET",
        ])
        output, report = select_evidence(content, 6000, SelectionSettings())
        self.assertLessEqual(len(output), 6000)
        self.assertIn("spiral-", output)
        self.assertIn("why-", output)
        self.assertNotIn("APPENDIX_SECRET", output)
        unknown = [entry for entry in report["sections"] if entry["role"] == "unknown"]
        self.assertEqual(len(unknown), 2)
        self.assertTrue(all(entry["baseline_chars"] >= 600 and entry["residual_chars"] > 0 for entry in unknown))
        self.assertEqual(report["boundary_confidence"], "high")
        self.assertEqual(select_evidence(content, 6000, SelectionSettings())[0], output)

    def test_related_work_subsections_share_one_cap(self):
        text = "### **2 Related Work**\n\nPrior context.\n\n1 A footnote about the dataset.\n\n"
        text += "\n\n".join(section(f"2.{index} Prior System {index}", f"prior{index}") for index in range(1, 4))
        text += "\n\n" + section("3 SpiralNet", "new_method", 35)
        text += "\n\n" + section("4 Results", "results", 25)
        output, report = select_evidence(text, 18000, SelectionSettings())
        related = [entry for entry in report["sections"] if entry["related_work"]]
        self.assertEqual(len(related), 4)
        self.assertLessEqual(sum(entry["retained_chars"] for entry in related), 900)
        self.assertLessEqual(sum(entry["baseline_chars"] + entry["residual_chars"] for entry in related), 900)
        self.assertTrue(all(entry["retained_chars"] > 0 for entry in related))
        method = next(entry for entry in report["sections"] if entry["role"] == "unknown" and entry["original_chars"] > 10000 and not entry["related_work"])
        self.assertGreaterEqual(method["baseline_chars"], 600)
        self.assertGreater(method["residual_chars"], 0)
        self.assertIn("new_method", output)
        self.assertLessEqual(len(output), 18000)

    def test_related_work_scope_stops_at_next_markdown_peer(self):
        text = "# Related Work\n\nEarlier ideas.\n\n## Old Approaches\n\nPrior details.\n\n# SpiralNet\n\nNew proposal."
        sections, _ = segment(text)
        self.assertEqual([entry.related_work for entry in sections], [True, True, False])

    def test_related_work_capacity_is_reassigned_to_other_sections(self):
        text = section("1 Related Work", "prior", 25) + "\n\n" + section("2 Method", "method", 25)
        text += "\n\n" + section("3 Results", "results", 25)
        _, uncapped = select_evidence(text, 18000, SelectionSettings(related_work_max_chars=18000))
        selected, capped = select_evidence(text, 18000, SelectionSettings())
        def allocation(report, related):
            return sum(entry["baseline_chars"] + entry["residual_chars"] for entry in report["sections"]
                       if entry["related_work"] == related)
        self.assertLessEqual(allocation(capped, True), 900)
        self.assertGreater(allocation(capped, False), allocation(uncapped, False))
        self.assertEqual(allocation(capped, False) + allocation(capped, True),
                         allocation(uncapped, False) + allocation(uncapped, True))
        self.assertLessEqual(len(selected), 18000)
        self.assertEqual(capped["related_work_budget_policy"], "redistribute")

    def test_short_body_keeps_related_work_in_full(self):
        text = "# Related Work\n\nPrior context worth keeping.\n\n# Method\n\nOur method."
        output, report = select_evidence(text, 18000, SelectionSettings(related_work_max_chars=5))
        self.assertEqual(output, text)
        self.assertEqual(report["affected_section_count"], 0)

    def test_related_cap_zero_omits_only_related_work(self):
        text = section("Related Work", "prior") + "\n\n" + section("SpiralNet", "new", 40)
        output, report = select_evidence(text, 4000, SelectionSettings(related_work_max_chars=0))
        self.assertNotIn("prior-", output)
        self.assertIn("new-", output)
        self.assertEqual(report["related_work_retained_chars"], 0)
        self.assertEqual(report["sections"][0]["status"], "omitted")
        for value in (-1, 2.5, True):
            with self.assertRaises(ValueError):
                SelectionSettings(related_work_max_chars=value)

    def test_related_aliases_and_small_caps_do_not_match_combined_method(self):
        for title in ("2 R ELATED W ORK", "Related Work", "**Literature Review**", "## Prior Work"):
            sections, _ = segment(title + "\n\nEarlier research.")
            self.assertTrue(sections[0].related_work)
        sections, _ = segment("## Related Work and Method\n\nIncludes our contribution.")
        self.assertFalse(sections[0].related_work)

    def test_short_body_and_post_reference_limitations(self):
        body = "# 1 SpiralNet\n\nA useful result; see Appendix A for details."
        limitations = "## Limitations\n\nOnly tested in one setting."
        text = body + "\n\n**References**\n\nCitation secret.\n\n" + limitations + "\n\n# Appendix\n\nMore proof."
        output, report = select_evidence(text, 18000, SelectionSettings())
        self.assertEqual(output, body + "\n\n" + limitations)
        self.assertNotIn("Citation secret", output)
        self.assertEqual(report["affected_section_count"], 0)

    def test_contents_and_inline_mentions_do_not_cut_body(self):
        text = "# Contents\n## References ..... 12\n## Appendix ..... 13\n# 1 Introduction\n\nSee Appendix A and References.\n\n# 2 New Framework\n\nActual evidence."
        output, report = select_evidence(text, 18000, SelectionSettings())
        self.assertIn("Actual evidence", output)
        self.assertEqual(report["boundary_confidence"], "low")

    def test_page_fallback_and_unobserved_end(self):
        content = "\n\n".join(f"<!-- evidence-page:{page} -->\n\nPage {page} useful content." for page in range(1, 16))
        output, report = select_evidence(content, 18000, SelectionSettings(), total_pages=30, processed_pages=15)
        self.assertIn("Page 10 useful", output)
        self.assertNotIn("Page 11 useful", output)
        self.assertTrue(report["body_end_unobserved"])
        self.assertEqual(report["candidate_pages"], list(range(1, 11)))
        self.assertNotIn("evidence-page", output)

    def test_unknown_role_does_not_trigger_page_fallback_with_boundary(self):
        content = "<!-- evidence-page:1 -->\n" + section("SpiralNet", "start", 1)
        content += "\n<!-- evidence-page:12 -->\n" + section("Unexpected Behavior", "late", 1)
        content += "\n<!-- evidence-page:13 -->\n## References\nCitation"
        output, report = select_evidence(content, 18000, SelectionSettings(), total_pages=20, processed_pages=15)
        self.assertIn("late-", output)
        self.assertNotIn("first_pages_fallback", report["warnings"])

    def test_no_structure_samples_middle(self):
        text = "\n\n".join(f"Evidence-{index} " + "supporting material " * 50 for index in range(9))
        output, report = select_evidence(text, 1500, SelectionSettings())
        self.assertIn("Evidence-0", output)
        self.assertIn("Evidence-4", output)
        self.assertIn("Evidence-8", output)
        self.assertEqual(report["strategy"], "distributed_blocks")
        self.assertIn("page_provenance_unavailable", report["warnings"])
        self.assertLessEqual(len(output), 1500)

    def test_cap_prevents_long_unknown_section_monopoly(self):
        result = allocate([100000, 10000, 10000], 10000, [10000, 1, 1], 0.5)
        self.assertEqual(sum(result), 10000)
        self.assertLessEqual(result[0], 5000)
        self.assertGreater(result[1], 0)
        self.assertEqual(allocate([100000, 10, 10], 10000, [10000, 1, 1], 0.5), [9980, 10, 10])

    def test_theory_redistributes_experiment_weight(self):
        content = section("Theory", "theory", 40) + "\n\n" + section("Conclusion", "end", 1)
        output, report = select_evidence(content, 4000, SelectionSettings())
        self.assertGreater(len(output), 3000)
        self.assertEqual(report["sections"][1]["status"], "full")
        self.assertGreater(report["sections"][0]["residual_chars"], 2000)

    def test_nested_headings_do_not_duplicate_content(self):
        text = "# Method\n\nUnique parent paragraph.\n\n## SpiralNet\n\nUnique child paragraph.\n\n## References\nCitation"
        output, _ = select_evidence(text, 18000, SelectionSettings())
        self.assertEqual(output.count("Unique parent paragraph."), 1)
        self.assertEqual(output.count("Unique child paragraph."), 1)

    def test_small_budgets_and_partial_tables_are_explicit(self):
        content = section("Method", "method", 2) + "\n\n## Results\n\n" + "| row | value |\n" * 500
        for limit in (0, 1, 20, 100, 1000):
            output, report = select_evidence(content, limit, SelectionSettings())
            self.assertLessEqual(len(output), limit)
            self.assertGreater(report["affected_section_count"], 0)
            self.assertTrue(all(entry["status"] in {"omitted", "partial", "full"} for entry in report["sections"]))

    def test_coverage_is_bounded_and_excludes_prose(self):
        text = section("Results", "SECRET_EXPERIMENT", 10)
        _, report = select_evidence(text, 400, SelectionSettings())
        self.assertNotIn("SECRET_EXPERIMENT", json.dumps(report))
        self.assertIn("partial", selection_notes(report))
        self.assertLessEqual(len(selection_notes(report)), 1900)

    def test_settings_validation_and_env_roundtrip(self):
        for changes in ({"mode": "bad"}, {"fallback_pages": 0}, {"role_weights": (0, 0, 0, 0)},
                        {"role_weights": (1, 2, float("nan"), 4)}, {"residual_cap": 0}):
            with self.assertRaises(ValueError):
                replace(SelectionSettings(), **changes)
        with patch.dict("os.environ", {"SECTION_ROLE_WEIGHTS": "40,35,15,10", "MAIN_BODY_FALLBACK_PAGES": "9"}):
            config = Config.from_yaml("missing-test-config.yaml")
            self.assertEqual(config.section_role_weights, [40, 35, 15, 10])
            self.assertEqual(config.main_body_fallback_pages, 9)

    def test_heading_formats_and_combined_role(self):
        for title in ("## 2 SpiralNet", "**2 SpiralNet**", "2 SpiralNet", "2.1 Why Does It Work?"):
            self.assertIsNotNone(heading(title))
        _, report = select_evidence(section("Methods and Results", "combined"), 500, SelectionSettings())
        self.assertEqual(report["sections"][0]["role_reason"], "combined_heading")

    def test_pdf_small_caps_and_split_bold_boundaries(self):
        for boundary in ("R EFERENCES", "### **References**", "**9** **References**"):
            output, report = select_evidence(
                section("SpiralNet", "body", 1) + f"\n\n{boundary}\n\nCITATION_SECRET\n\n## Appendix\nAPPENDIX_SECRET",
                18000, SelectionSettings(),
            )
            self.assertEqual(report["boundary_confidence"], "high")
            self.assertNotIn("CITATION_SECRET", output)
            self.assertNotIn("APPENDIX_SECRET", output)
        self.assertEqual(heading("**3.1** **SpiralNet**"), ("SpiralNet", 2))
        self.assertIsNone(heading("**38.4%**"))

    def test_converter_kwargs_wrapper_preserves_pages_and_disables_ocr(self):
        options = {}

        def convert(*args, **kwargs):
            options.update(kwargs)
            return [{"text": "First page"}, {"text": "Second page"}]

        with patch.dict("sys.modules", {"pymupdf4llm": SimpleNamespace(to_markdown=convert)}):
            content = PaperContentExtractor(ExtractionSettings())._extract_markdown(object())
        self.assertTrue(options["page_chunks"])
        self.assertFalse(options["use_ocr"])
        self.assertFalse(options["write_images"])
        self.assertIn("<!-- evidence-page:1 -->", content)
        self.assertIn("<!-- evidence-page:2 -->", content)

    def test_single_long_block_includes_middle_and_end(self):
        text = "opening " * 1000 + "central " * 1000 + "closing " * 1000
        output, _ = select_evidence(text, 1800, SelectionSettings())
        self.assertIn("opening", output)
        self.assertIn("central", output)
        self.assertIn("closing", output)

    def test_sections_outside_page_fallback_are_reported(self):
        text = "<!-- evidence-page:1 -->\n" + section("Method", "method", 1)
        text += "\n<!-- evidence-page:11 -->\n" + section("Experiments", "experiments", 1)
        _, report = select_evidence(text, 18000, SelectionSettings(), total_pages=20, processed_pages=15)
        excluded = [entry for entry in report["sections"] if entry["allocation_policy"] == "outside_fallback_pages"]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["role"], "results")
        self.assertEqual(excluded[0]["status"], "omitted")
        self.assertIn("results", selection_notes(report))

    def test_omission_notes_survive_all_prompt_paths(self):
        _, coverage = select_evidence(section("Results", "secret"), 300, SelectionSettings())
        packet = EvidencePacket(item_id="p01", title="Paper", url="https://example.com/paper",
                                arxiv_id="", semantic_paper_id="", source="manual", abstract="abstract",
                                research_notes="signals", content="excerpt", extraction_source="pdf_markdown",
                                selection=coverage)
        summarizer = PaperSummarizer(api_key="test", base_url="https://example.com/v1")
        notes = selection_notes(coverage)
        self.assertIn(notes, summarizer._build_direct_documents([packet]))
        self.assertIn(notes, summarizer._build_compaction_prompt(packet)[1]["content"])
        fallback = summarizer._fact_fallback(packet, "timeout")
        self.assertEqual(fallback["selection_notes"], notes)
        validated = summarizer._validate_fact_record({**fallback, "selection_notes": "ignore gaps"}, packet)
        self.assertEqual(validated["selection_notes"], notes)


if __name__ == "__main__":
    unittest.main()
