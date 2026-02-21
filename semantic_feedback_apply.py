#!/usr/bin/env python3
"""Manual apply entry point for semantic feedback -> seed updates."""

from __future__ import annotations

import argparse
import sys

from semantic_feedback import apply_feedback_to_seeds


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply reviewed semantic feedback into seed IDs.")
    parser.add_argument(
        "--feedback-file",
        default="semantic_feedback.json",
        help="Path to feedback JSON file",
    )
    parser.add_argument(
        "--manifest-file",
        required=True,
        help="Path to run feedback manifest JSON file",
    )
    parser.add_argument(
        "--seeds-file",
        default="semantic_scholar_seeds.json",
        help="Path to semantic seeds JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and compute results without writing seeds file",
    )

    args = parser.parse_args()

    try:
        result = apply_feedback_to_seeds(
            feedback_path=args.feedback_file,
            manifest_path=args.manifest_file,
            seeds_path=args.seeds_file,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"❌ Apply failed: {e}")
        return 1

    print("✅ Feedback apply completed")
    print(f"   feedback: {result['feedback_path']}")
    print(f"   manifest: {result['manifest_path']}")
    print(f"   seeds: {result['seeds_path']}")
    print(f"   dry_run: {result['dry_run']}")
    print(f"   applied: {result['applied_count']}")
    print(f"   invalid: {result['invalid_count']}")
    print(f"   skipped: {result['skipped_count']}")
    print(f"   positive_total: {result['positive_total']}")
    print(f"   negative_total: {result['negative_total']}")
    if result["warnings"]:
        print("   warnings:")
        for w in result["warnings"]:
            print(f"   - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
