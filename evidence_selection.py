"""Deterministic, bounded main-body selection without model calls."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional


ROLES = ("method", "results", "background", "conclusion")
ROLE_PATTERNS = (
    r"\b(method\w*|approach|algorithm|theor\w*|proof\w*)\b",
    r"\b(experiment\w*|evaluat\w*|result\w*|ablation\w*)\b",
    r"\b(abstract|introduction|background|related work|preliminar\w*)\b",
    r"\b(conclusion\w*|discussion|limitation\w*)\b",
)
PAGE_MARKER = re.compile(r"^<!-- evidence-page:(\d+) -->$")
OMISSION = "\n[... omitted ...]\n"


@dataclass(frozen=True)
class SelectionSettings:
    mode: str = "section_balanced"
    fallback_pages: int = 10
    role_weights: tuple[float, ...] = (35, 40, 15, 10)
    baseline_chars: int = 600
    residual_cap: float = 0.5
    related_work_max_chars: int = 900

    def __post_init__(self) -> None:
        if self.mode not in {"section_balanced", "head_tail"}:
            raise ValueError("paper_evidence_selection_mode must be section_balanced or head_tail")
        if any(type(value) is not int or value < 1 for value in (self.fallback_pages, self.baseline_chars)):
            raise ValueError("main_body_fallback_pages and section_baseline_chars must be positive")
        if (
            len(self.role_weights) != 4
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in self.role_weights)
            or not math.isfinite(sum(self.role_weights))
            or sum(self.role_weights) <= 0
        ):
            raise ValueError("section_role_weights requires four finite nonnegative weights with positive sum")
        if type(self.residual_cap) not in (int, float) or not math.isfinite(self.residual_cap) or not 0 < self.residual_cap <= 1:
            raise ValueError("section_residual_cap must be in (0, 1]")
        if type(self.related_work_max_chars) is not int or self.related_work_max_chars < 0:
            raise ValueError("related_work_max_chars must be a nonnegative integer")


@dataclass
class Section:
    title: str
    level: int
    lines: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    start_line: int = 0
    role: str = "unknown"
    confidence: str = "low"
    reason: str = "unclassified"
    related_work: bool = False

    @property
    def content(self) -> str:
        return "\n".join(self.lines).strip()


def heading(line: str) -> Optional[tuple[str, int]]:
    stripped = line.strip()
    if not stripped or len(stripped) > 180 or re.search(r"\.{3,}\s*\d+\s*$", stripped):
        return None
    markdown = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", stripped)
    bold = re.fullmatch(r"\*\*(.+?)\*\*", stripped)
    value = (markdown.group(2) if markdown else bold.group(1) if bold else stripped).strip()
    value = value.replace("**", "").strip()
    for word in ("REFERENCES", "BIBLIOGRAPHY", "ABSTRACT", "INTRODUCTION", "CONCLUSION",
                 "CONCLUSIONS", "APPENDIX", "APPENDICES", "LIMITATIONS", "METHOD", "RESULTS", "RELATED", "WORK"):
        value = re.sub(rf"\b{word[0]}\s+{word[1:]}\b", word, value)
    numbered = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+([A-Z][^\n]+)$", value)
    if not numbered and (markdown or bold):
        numbered = re.match(r"^([A-Z](?:\.\d+)*)[.)]?\s+([A-Z][^\n]+)$", value)
    title = numbered.group(2) if numbered else value
    level = len(markdown.group(1)) if markdown else numbered.group(1).count(".") + 1 if numbered else 1
    plain_section = re.fullmatch(
        r"abstract|introduction|conclusions?|discussion|limitations?|references|bibliography|related works?|prior work|literature review|"
        r"acknowledg(?:e)?ments?|appendi(?:x|ces)(?:\s+[A-Z])?|supplementary material",
        title, re.I,
    )
    if not markdown and not numbered and not plain_section:
        if not bold or len(value.split()) > 12 or not re.search(r"[A-Za-z]", value) or value.endswith((".", ",", ";")):
            return None
    if markdown or bold or numbered or plain_section:
        return title.strip(), level
    return None


def classify(section: Section) -> None:
    matches = [role for role, pattern in zip(ROLES, ROLE_PATTERNS) if re.search(pattern, section.title, re.I)]
    if len(matches) == 1:
        section.role, section.confidence, section.reason = matches[0], "high", "heading_role"
    elif len(matches) > 1:
        section.reason = "combined_heading"
    else:
        opening = section.content[:800]
        cues = {
            "method": r"\bwe (?:propose|introduce|derive)\b.*?\b(?:algorithm|method|framework|theorem)\b",
            "results": r"\bwe (?:evaluate|benchmark|compare)\b.*?\b(?:dataset|baseline|benchmark)s?\b",
        }
        inferred = [role for role, pattern in cues.items() if re.search(pattern, opening, re.I | re.S)]
        if len(inferred) == 1:
            section.role, section.confidence, section.reason = inferred[0], "medium", "opening_cue"


def mark_related_work(sections: list[Section]) -> None:
    parent_number = None
    parent_level = None
    for section in sections:
        title = re.sub(r"\s+", "", section.title).lower()
        raw_heading = re.sub(r"^#{1,6}\s*", "", section.lines[0]).replace("**", "").strip()
        match = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+[A-Z]", raw_heading)
        number = match.group(1) if match else None
        section.related_work = False
        if title in {"relatedwork", "relatedworks", "priorwork", "literaturereview"}:
            section.related_work = True
            parent_number, parent_level = number, section.level
        elif parent_level is not None:
            if number and parent_number:
                if number.startswith(parent_number + "."):
                    section.related_work = True
                elif int(number.split(".")[0]) > int(parent_number.split(".")[0]):
                    parent_number, parent_level = None, None
            elif section.level > parent_level:
                section.related_work = True
            else:
                parent_number, parent_level = None, None


def segment(content: str) -> tuple[list[Section], bool]:
    sections = []
    current = Section("", 0)
    page = None
    contents_level = None
    for line_index, line in enumerate(content.splitlines()):
        marker = PAGE_MARKER.fullmatch(line.strip())
        if marker:
            page = int(marker.group(1))
            continue
        parsed = heading(line)
        if parsed and parsed[0].lower() in {"contents", "table of contents"}:
            contents_level = parsed[1]
            continue
        if contents_level is not None:
            if not parsed or parsed[1] > contents_level or re.search(r"\s\d+\s*$", parsed[0]):
                continue
            contents_level = None
        if parsed:
            title, level = parsed
            normalized = title.lower()
            if current.title.lower() == normalized:
                continue
            if current.content:
                classify(current)
                sections.append(current)
            current = Section(title, level, start_line=line_index)
        current.lines.append(line)
        if page is not None and page not in current.pages:
            current.pages.append(page)
    if current.content:
        classify(current)
        sections.append(current)
    mark_related_work(sections)
    return sections, page is not None


def main_body(sections: list[Section]) -> tuple[list[Section], bool]:
    eligible = []
    excluded = False
    boundary_found = False
    for section in sections:
        title = section.title.strip().lower()
        boundary = bool(re.match(r"^(references|bibliography|acknowledg(?:e)?ments?)\b", title))
        appendix = bool(re.match(r"^(appendix|appendices|supplementary material)(?:\b|:)", title))
        lettered = bool(re.match(r"^(?:#{1,6}\s*)?(?:\*\*)?[A-Z](?:\.\d+)*[.)]?\s+", section.lines[0]))
        if eligible and (appendix or (excluded and lettered)):
            boundary_found = True
            break
        if boundary and eligible:
            excluded = True
            if title.startswith(("references", "bibliography")):
                boundary_found = True
            continue
        if excluded:
            if section.confidence == "high" and section.role == "conclusion":
                eligible.append(section)
            continue
        eligible.append(section)
    return eligible, boundary_found


def allocate(lengths: list[int], budget: int, weights: list[float], cap: float = 1.0) -> list[int]:
    allocations = [0] * len(lengths)
    remaining = max(0, budget)
    ceiling = max(1, math.ceil(remaining * max(cap, 1 / max(1, len(lengths)))))
    ceilings = [min(length, ceiling) for length in lengths]
    while remaining:
        active = [index for index, length in enumerate(ceilings) if allocations[index] < length]
        if not active:
            unmet = [index for index, length in enumerate(lengths) if allocations[index] < length]
            if not unmet:
                break
            extra_cap = max(1, math.ceil(remaining * max(cap, 1 / len(unmet))))
            for index in unmet:
                ceilings[index] = min(lengths[index], allocations[index] + extra_cap)
            continue
        total_weight = sum(weights[index] for index in active)
        round_budget = remaining
        grants = []
        for index in active:
            share = weights[index] / total_weight if total_weight else 1 / len(active)
            grants.append(min(ceilings[index] - allocations[index], int(round_budget * share)))
        if not any(grants):
            grants = [1] * len(active)
        for index, grant in zip(active, grants):
            granted = min(remaining, grant)
            allocations[index] += granted
            remaining -= granted
    return allocations


def sample_blocks(content: str, limit: int) -> tuple[str, int, bool]:
    if len(content) <= limit:
        return content, len(re.split(r"\n\s*\n", content)), False
    if limit <= len(OMISSION):
        return "", 0, True
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]
    if len(blocks) == 1:
        block = blocks[0]
        split_size = max(1, math.ceil(len(block) / 3))
        blocks = [block[offset:offset + split_size] for offset in range(0, len(block), split_size)]
    selected: dict[int, str] = {}
    candidates = list(dict.fromkeys([0, len(blocks) // 2, len(blocks) - 1]))
    candidates += [index for index in range(len(blocks)) if index not in candidates]
    remaining = limit - len(OMISSION)
    for position, index in enumerate(candidates):
        reserve = 2
        slots = max(1, min(3, len(candidates)) - position)
        allowance = max(0, remaining // slots - reserve)
        block = blocks[index]
        if len(block) <= allowance:
            selected[index] = block
        elif allowance > 60 and position < 3:
            marker = " [partial block]"
            fragment = block[:allowance - len(marker)]
            sentence = max(fragment.rfind(". "), fragment.rfind("。"))
            if sentence > len(fragment) // 2:
                fragment = fragment[:sentence + 1]
            selected[index] = fragment.rstrip() + marker
        else:
            continue
        remaining -= len(selected[index]) + reserve
    result = "\n\n".join(selected[index] for index in sorted(selected)) + OMISSION
    return result, len(selected), True


def select_evidence(content: str, limit: int, settings: SelectionSettings, *,
                    total_pages: int = 0, processed_pages: int = 0) -> tuple[str, dict]:
    sections, has_pages = segment(content)
    body, boundary_found = main_body(sections)
    observed_body = {section.start_line: section for section in body}
    notes = []
    if not boundary_found:
        notes.append("body_boundary_uncertain")
        if has_pages:
            page = 0
            prefix = []
            for line in content.splitlines():
                marker = PAGE_MARKER.fullmatch(line.strip())
                if marker:
                    page = int(marker.group(1))
                if page <= settings.fallback_pages:
                    prefix.append(line)
            body, _ = main_body(segment("\n".join(prefix))[0])
            notes.append("first_pages_fallback")
        else:
            notes.append("page_provenance_unavailable")
    if processed_pages < total_pages and not boundary_found:
        notes.append("body_end_unobserved")
    structured = any(section.level for section in body)
    if not structured:
        notes.append("unclassified_sampling")
    bodies = [section.content for section in body]
    lengths = [len(value) for value in bodies]
    oversized = sum(lengths) + max(0, len(body) - 1) * 2 > limit
    allocation_lengths = [length + len(f"[section s{index + 1:02d}]\n") if oversized else length
                          for index, length in enumerate(lengths)]
    related_indices = [index for index, section in enumerate(body) if section.related_work]
    budget = max(0, limit - max(0, len(body) - 1) * 2)
    baseline = allocate([min(length, settings.baseline_chars) for length in allocation_lengths], budget, [1.0] * len(body))
    demands = [length - initial for length, initial in zip(allocation_lengths, baseline)]
    weights = []
    role_total = sum(settings.role_weights)
    for section, demand in zip(body, demands):
        preference = settings.role_weights[ROLES.index(section.role)] / role_total * 4 if section.confidence == "high" else 1.0
        weights.append(math.sqrt(demand) * preference)
    residual = allocate(demands, budget - sum(baseline), weights, settings.residual_cap)
    original_allocations = [initial + extra for initial, extra in zip(baseline, residual)]
    if oversized and related_indices:
        related_lengths = [original_allocations[index] for index in related_indices]
        related_baseline = allocate(
            [min(length, 200, settings.baseline_chars) for length in related_lengths],
            settings.related_work_max_chars, [1.0] * len(related_indices),
        )
        related_demands = [length - initial for length, initial in zip(related_lengths, related_baseline)]
        related_extra = allocate(related_demands, settings.related_work_max_chars - sum(related_baseline),
                                 [math.sqrt(demand) for demand in related_demands])
        for index, initial, extra in zip(related_indices, related_baseline, related_extra):
            related_limit = initial + extra
            baseline[index] = min(baseline[index], related_limit)
            residual[index] = related_limit - baseline[index]
    saved_budget = sum(original_allocations) - sum(baseline) - sum(residual)
    output = []
    coverage = []
    for index, (section, initial, extra) in enumerate(zip(body, baseline, residual)):
        label = f"[section s{index + 1:02d}]\n" if oversized else ""
        selected, blocks, changed = sample_blocks(section.content, max(0, initial + extra - len(label)))
        if selected:
            selected = label + selected
            output.append(selected)
        observed = observed_body.get(section.start_line, section)
        changed = changed or len(section.content) < len(observed.content)
        coverage.append({
            "section_id": f"s{index + 1:02d}", "start_line": section.start_line,
            "pages": section.pages, "role": section.role,
            "role_confidence": section.confidence, "role_reason": section.reason,
            "related_work": section.related_work,
            "allocation_policy": "related_work_cap" if oversized and section.related_work else "role_weighted" if section.confidence == "high" else "capped_length",
            "original_chars": len(observed.content), "retained_chars": len(selected),
            "baseline_chars": initial, "residual_chars": extra, "retained_blocks": blocks,
            "allocation_before_related_cap": original_allocations[index],
            "status": "omitted" if not selected else "partial" if changed else "full",
        })
    included_lines = {section.start_line for section in body}
    for section in observed_body.values():
        if section.start_line in included_lines:
            continue
        coverage.append({
            "section_id": f"s{len(coverage) + 1:02d}", "start_line": section.start_line,
            "pages": section.pages, "role": section.role, "role_confidence": section.confidence,
            "related_work": section.related_work,
            "role_reason": section.reason, "allocation_policy": "outside_fallback_pages",
            "original_chars": len(section.content), "retained_chars": 0,
            "baseline_chars": 0, "residual_chars": 0, "retained_blocks": 0, "status": "omitted",
        })
    selected_content = "\n\n".join(output)
    affected = [entry for entry in coverage if entry["status"] != "full"]
    if affected:
        notes.append("sections_omitted_or_partial")
    pages = sorted({page for section in body for page in section.pages})
    diagnostics = {
        "strategy": "section_balanced" if structured else "distributed_blocks",
        "boundary_confidence": "high" if boundary_found else "low",
        "boundary_reason": "structural_end_marker" if boundary_found else "no_confident_end_marker",
        "body_end_unobserved": "body_end_unobserved" in notes,
        "page_provenance": has_pages, "candidate_pages": pages,
        "candidate_chars": sum(lengths), "selected_chars": len(selected_content),
        "baseline_target": settings.baseline_chars, "residual_cap": settings.residual_cap,
        "role_weights": list(settings.role_weights), "section_count": len(coverage),
        "related_work_max_chars": settings.related_work_max_chars,
        "related_work_budget_policy": "save_without_refill",
        "related_work_budget_saved_chars": saved_budget,
        "effective_content_limit": max(0, limit - saved_budget),
        "related_work_retained_chars": sum(entry["retained_chars"] for entry in coverage if entry["related_work"]),
        "related_work_candidate_chars": sum(len(section.content) for section in body if section.related_work),
        "affected_section_count": len(affected), "sections": coverage[:80],
        "diagnostics_sections_omitted": max(0, len(coverage) - 80), "warnings": notes,
    }
    return selected_content, diagnostics


def selection_notes(diagnostics: dict) -> str:
    if not diagnostics:
        return ""
    warnings = ", ".join(diagnostics.get("warnings", []))
    sections = diagnostics.get("sections", [])
    affected = [f"{entry['section_id']}({entry['role']}):{entry['status']}" for entry in sections if entry["status"] != "full"]
    summary = (
        f"Selection={diagnostics.get('strategy')}; boundary={diagnostics.get('boundary_confidence', 'unknown')}; "
        f"warnings={warnings}; affected_sections={diagnostics.get('affected_section_count', 0)}. "
        + "; ".join(affected)
    )
    return summary[:1800] + "\nOmitted excerpt evidence is not proof of missing work in the original paper."
