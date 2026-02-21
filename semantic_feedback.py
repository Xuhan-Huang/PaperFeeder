"""Human-in-the-loop feedback manifest export and seed apply utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlsplit, urlunsplit


ALLOWED_LABELS = {"positive", "negative", "undecided"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
        scheme = parts.scheme.lower() if parts.scheme else "https"
        netloc = parts.netloc.lower()
        path = parts.path.rstrip("/")
        return urlunsplit((scheme, netloc, path, "", ""))
    except Exception:
        return url.strip().lower().rstrip("/")


def _normalize_title(title: str) -> str:
    return " ".join((title or "").strip().lower().split())


def normalize_paper_id(value: Any) -> str:
    """Normalize seed paper IDs (numeric corpus IDs -> CorpusId:<id>)."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if s.isdigit():
        return f"CorpusId:{s}"
    return s


def _extract_report_urls(report_html: str) -> set[str]:
    import re

    if not report_html:
        return set()
    raw_urls = re.findall(r'href=["\']([^"\']+)["\']', report_html, flags=re.IGNORECASE)
    return {normalize_url(u) for u in raw_urls if normalize_url(u)}


def build_run_id(now: datetime | None = None) -> str:
    dt = now or _utc_now()
    return dt.strftime("%Y-%m-%dT%H-%M-%SZ")


def export_run_feedback_manifest(
    final_papers: Iterable[Any],
    report_html: str,
    output_dir: str = "artifacts",
    run_id: str | None = None,
) -> Tuple[Path, Path] | None:
    """Export final report paper mappings for human feedback."""
    papers = list(final_papers or [])
    if not papers:
        return None

    visible_urls = _extract_report_urls(report_html)
    entries: List[Dict[str, Any]] = []

    for p in papers:
        url = getattr(p, "url", "") or ""
        norm_url = normalize_url(url)
        if visible_urls and norm_url not in visible_urls:
            continue
        entries.append(
            {
                "title": getattr(p, "title", "") or "",
                "url": url,
                "semantic_paper_id": normalize_paper_id(getattr(p, "semantic_paper_id", "")) or None,
            }
        )

    if not entries:
        return None

    rid = run_id or build_run_id()
    for idx, e in enumerate(entries, 1):
        e["item_id"] = f"p{idx:02d}"

    payload = {
        "version": "v1",
        "run_id": rid,
        "generated_at": _to_iso(_utc_now()),
        "papers": entries,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"run_feedback_manifest_{rid}.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    # Emit a starter questionnaire for direct copy/edit by reviewers.
    feedback_template = {
        "version": "v1",
        "run_id": rid,
        "reviewer": "",
        "reviewed_at": _to_iso(_utc_now()),
        "labels": [
            {
                "item_id": p["item_id"],
                "label": "undecided",
                "note": "",
            }
            for p in entries
        ],
    }
    questionnaire_path = out_dir / f"semantic_feedback_template_{rid}.json"
    questionnaire_path.write_text(json.dumps(feedback_template, indent=2) + "\n")

    return manifest_path, questionnaire_path


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {path}")
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def _sort_seed_ids(values: Iterable[str]) -> List[str]:
    def sort_key(v: str) -> Tuple[int, str]:
        s = normalize_paper_id(v)
        if s.startswith("CorpusId:"):
            tail = s.split(":", 1)[1]
            if tail.isdigit():
                return (0, f"{int(tail):020d}")
        return (1, s.lower())

    return sorted({normalize_paper_id(v) for v in values if normalize_paper_id(v)}, key=sort_key)


def apply_feedback_to_seeds(
    feedback_path: str = "semantic_feedback.json",
    manifest_path: str = "",
    seeds_path: str = "semantic_scholar_seeds.json",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Validate feedback and deterministically update seed file."""
    if not manifest_path:
        raise ValueError("manifest_path is required")

    manifest = _load_json(manifest_path)
    feedback = _load_json(feedback_path)

    m_run_id = str(manifest.get("run_id", "")).strip()
    f_run_id = str(feedback.get("run_id", "")).strip()
    if not m_run_id or not f_run_id:
        raise ValueError("Both manifest.run_id and feedback.run_id are required")
    if m_run_id != f_run_id:
        raise ValueError(f"run_id mismatch: manifest={m_run_id}, feedback={f_run_id}")

    reviewer = str(feedback.get("reviewer", "")).strip()
    reviewed_at = str(feedback.get("reviewed_at", "")).strip()
    if not reviewer:
        raise ValueError("feedback.reviewer is required")
    if _parse_iso(reviewed_at) is None:
        raise ValueError("feedback.reviewed_at must be ISO-8601 timestamp")

    papers = manifest.get("papers", [])
    if not isinstance(papers, list):
        raise ValueError("manifest.papers must be an array")

    labels = feedback.get("labels", [])
    if not isinstance(labels, list):
        raise ValueError("feedback.labels must be an array")

    by_item_id: Dict[str, Dict[str, Any]] = {}
    by_semantic_id: Dict[str, Dict[str, Any]] = {}
    by_title_url: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for p in papers:
        if not isinstance(p, dict):
            continue
        item_id = str(p.get("item_id", "")).strip()
        title = str(p.get("title", "")).strip()
        url = str(p.get("url", "")).strip()
        semantic_id = normalize_paper_id(p.get("semantic_paper_id"))
        if item_id:
            by_item_id[item_id] = p
        if semantic_id:
            by_semantic_id[semantic_id] = p
        if title or url:
            by_title_url[(_normalize_title(title), normalize_url(url))] = p

    latest: Dict[str, Tuple[datetime, str]] = {}
    warnings: List[str] = []
    invalid_count = 0
    skipped_count = 0

    for idx, entry in enumerate(labels, 1):
        if not isinstance(entry, dict):
            invalid_count += 1
            warnings.append(f"label[{idx}] invalid: entry must be an object")
            continue

        label = str(entry.get("label", "")).strip().lower()
        if label not in ALLOWED_LABELS:
            invalid_count += 1
            warnings.append(f"label[{idx}] invalid label: {label!r}")
            continue

        label_ts = _parse_iso(str(entry.get("reviewed_at", "")).strip()) or _parse_iso(reviewed_at)
        if label_ts is None:
            invalid_count += 1
            warnings.append(f"label[{idx}] invalid reviewed_at")
            continue

        resolved = None
        item_id = str(entry.get("item_id", "")).strip()
        if item_id:
            resolved = by_item_id.get(item_id)

        if resolved is None:
            sem = normalize_paper_id(entry.get("semantic_paper_id"))
            if sem:
                resolved = by_semantic_id.get(sem)

        if resolved is None:
            title = _normalize_title(str(entry.get("title", "")))
            url = normalize_url(str(entry.get("url", "")))
            resolved = by_title_url.get((title, url))

        if resolved is None:
            skipped_count += 1
            warnings.append(f"label[{idx}] skipped: no matching paper in manifest")
            continue

        semantic_id = normalize_paper_id(resolved.get("semantic_paper_id"))
        if not semantic_id:
            skipped_count += 1
            warnings.append(f"label[{idx}] skipped: matched paper has no semantic_paper_id")
            continue

        current = latest.get(semantic_id)
        if current is None or label_ts >= current[0]:
            latest[semantic_id] = (label_ts, label)

    seeds_file = Path(seeds_path)
    if seeds_file.exists():
        seeds = _load_json(seeds_path)
    else:
        seeds = {}

    positive = set(normalize_paper_id(v) for v in seeds.get("positive_paper_ids", []) or [])
    negative = set(normalize_paper_id(v) for v in seeds.get("negative_paper_ids", []) or [])
    positive.discard("")
    negative.discard("")

    applied_count = 0
    for semantic_id, (_ts, label) in sorted(latest.items()):
        if label == "positive":
            positive.add(semantic_id)
            negative.discard(semantic_id)
            applied_count += 1
        elif label == "negative":
            negative.add(semantic_id)
            positive.discard(semantic_id)
            applied_count += 1
        else:
            # undecided keeps seeds unchanged
            pass

    output = {
        "positive_paper_ids": _sort_seed_ids(positive),
        "negative_paper_ids": _sort_seed_ids(negative),
    }

    if not dry_run:
        seeds_file.write_text(json.dumps(output, indent=2) + "\n")

    return {
        "feedback_path": feedback_path,
        "manifest_path": manifest_path,
        "seeds_path": seeds_path,
        "dry_run": dry_run,
        "applied_count": applied_count,
        "invalid_count": invalid_count,
        "skipped_count": skipped_count,
        "warnings": warnings,
        "positive_total": len(output["positive_paper_ids"]),
        "negative_total": len(output["negative_paper_ids"]),
    }
