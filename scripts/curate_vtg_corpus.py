#!/usr/bin/env python3
"""Curate a deterministic VTG corpus candidate set from an enriched inventory.

The output is still a candidate set: records whose venue was inferred from a
filename/PDF rather than a bibliographic registry are explicitly marked and
must pass the venue gate before a release is frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any


CCF_A_VENUES = {
    "AAAI",
    "NEURIPS",
    "ACL",
    "CVPR",
    "ICCV",
    "ICML",
    "ICLR",
    "ACM MM",
    "TPAMI",
    "IJCV",
    "TIP",
    "TMM",
}

TITLE_SCOPE = re.compile(
    r"(?:"
    r"video temporal grounding|temporal video grounding|temporal sentence grounding|"
    r"video sentence grounding|video moment (?:retrieval|locali[sz]ation)|"
    r"moment locali[sz]ation (?:with|of|via) natural language|"
    r"natural language video locali[sz]ation|temporal grounding|"
    r"video grounding|spatio[- ]temporal (?:video )?grounding|"
    r"language[- ]based temporal locali[sz]ation|query[- ]based video locali[sz]ation|"
    r"temporal activity locali[sz]ation via language|language grounding in videos?|"
    r"temporal language locali[sz]ation in videos?|temporal sentence locali[sz]ation|"
    r"activity locali[sz]ation in videos via sentence query|video temporal locali[sz]ation|"
    r"video sentence locali[sz]ation|"
    r"grounding language queries in videos?|locali[sz]ing moments? in (?:long )?video|"
    r"moment retrieval|video paragraph grounding"
    r")",
    re.I,
)

# Named task papers whose titles use non-standard wording.
TITLE_ALLOWLIST = re.compile(
    r"(?:"
    r"learning 2d temporal adjacent networks|dense events grounding in video|"
    r"local-global video-text interactions|dense regression network for video grounding|"
    r"mad: a scalable dataset for language grounding in videos|"
    r"span-based localizing network|snag: scalable and accurate video grounding|"
    r"univtg: towards unified video-language temporal grounding|"
    r"vtime?llm|revisionllm|timeexpert|moment quantization|"
    r"you can ground earlier than see|identity-text video corpus grounding|"
    r"prior knowledge integration via llm encoding"
    r"|relational network via cascade crf for video language grounding"
    r"|siamese learning with joint alignment and regression for weakly supervised video"
    r")",
    re.I,
)

TITLE_EXCLUDE = (
    ("survey", re.compile(r"\bsurvey\b|elements of temporal sentence grounding", re.I)),
    ("video_question_grounding", re.compile(r"video question grounding|question answering", re.I)),
    ("generic_video_reasoning", re.compile(r"video-skill-cot", re.I)),
    ("generic_video_description", re.compile(r"comprehensive visual grounding for video description", re.I)),
    ("spatial_only", re.compile(r"open-vocabulary visual grounding", re.I)),
    ("non_ccf_a_wacv", re.compile(r"\bmac mining activity concepts for language based temporal locali[sz]ation\b", re.I)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalize_title(value: str) -> str:
    value = value.casefold().replace("&", " and ")
    value = re.sub(r"\b(?:paper|conference)\b", " ", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def canonical_title(record: dict[str, Any]) -> str:
    match = record.get("openalex_match")
    if isinstance(match, dict) and float(match.get("score") or 0.0) >= 0.86:
        return str(match.get("title") or record["title_inferred"])
    return str(record["title_inferred"])


def venue(record: dict[str, Any]) -> str | None:
    return record.get("venue_verified_value") or record.get("venue_inferred")


def exclusion_reason(record: dict[str, Any], title: str) -> str | None:
    searchable_title = normalize_title(title)
    if "workshop" in (record.get("exclusion_signals") or []):
        return "workshop_or_non_regular_track"
    for label, pattern in TITLE_EXCLUDE:
        if pattern.search(title) or pattern.search(searchable_title):
            return label
    if venue(record) not in CCF_A_VENUES:
        return "not_ccf_a_candidate"
    if not (
        TITLE_SCOPE.search(title)
        or TITLE_SCOPE.search(searchable_title)
        or TITLE_ALLOWLIST.search(title)
        or TITLE_ALLOWLIST.search(searchable_title)
    ):
        return "outside_title_scope"
    return None


def paradigm_tags(title: str) -> list[str]:
    tags: list[str] = []
    if re.search(
        r"weakly[- ]supervised|point[- ]supervised|point annotations?|single[- ]frame supervision|"
        r"sparse supervision|semi[- ]supervised|self[- ]supervised|unsupervised|"
        r"unlabeled videos?|cross[- ]supervision",
        title,
        re.I,
    ):
        tags.append("weakly_supervised")
    if re.search(
        r"zero[- ]shot|training[- ]free|language[- ]free training|open[- ]vocabulary",
        title,
        re.I,
    ):
        tags.append("zero_shot")
    if re.search(
        r"\b(?:llm|mllm|lmm|vlm)s?\b|large (?:vision[- ]language|language) models?|"
        r"vision[- ]language model|generative multi[- ]modal",
        title,
        re.I,
    ):
        tags.append("llm_based")
    return tags or ["fully_supervised"]


def primary_paradigm(tags: list[str]) -> str:
    for label in ("llm_based", "zero_shot", "weakly_supervised", "fully_supervised"):
        if label in tags:
            return label
    raise AssertionError("unreachable")


def record_quality(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(bool(record.get("venue_verified"))),
        int(record.get("pages") or 0),
        int(record.get("size_bytes") or 0),
    )


def merge_near_duplicate_groups(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for record in records:
        title = normalize_title(record["title"])
        destination: list[dict[str, Any]] | None = None
        for group in groups:
            other = normalize_title(group[0]["title"])
            if title == other or (
                min(len(title), len(other)) >= 24
                and SequenceMatcher(None, title, other).ratio() >= 0.96
            ):
                destination = group
                break
        (destination if destination is not None else groups.append([record]))
        if destination is not None:
            destination.append(record)
    return groups


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for source in load_jsonl(args.inventory):
        if source["sha256"] in seen_hashes:
            excluded.append({**source, "exclusion_reason": "exact_duplicate"})
            continue
        seen_hashes.add(source["sha256"])
        title = canonical_title(source)
        reason = exclusion_reason(source, title)
        if reason:
            excluded.append({**source, "title": title, "exclusion_reason": reason})
            continue
        tags = paradigm_tags(title)
        candidates.append(
            {
                **source,
                "title": title,
                "venue": venue(source),
                "venue_evidence": "openalex_registry" if source.get("venue_verified") else "local_inference_pending_verification",
                "paradigm_primary": primary_paradigm(tags),
                "paradigm_tags": tags,
                "task_scope": "video_temporal_grounding",
            }
        )

    selected: list[dict[str, Any]] = []
    for group in merge_near_duplicate_groups(candidates):
        best = max(group, key=record_quality)
        selected.append(best)
        for duplicate in group:
            if duplicate is not best:
                excluded.append({**duplicate, "exclusion_reason": "same_paper_different_file"})
    selected.sort(key=lambda item: (int(item.get("year_inferred") or 9999), item["title"].casefold()))
    excluded.sort(key=lambda item: (item["exclusion_reason"], item.get("source_filename", "").casefold()))

    write_jsonl(args.selected, selected)
    write_jsonl(args.excluded, excluded)
    evidence_counts = Counter(item["venue_evidence"] for item in selected)
    paradigm_counts = Counter(item["paradigm_primary"] for item in selected)
    summary = {
        "selected_unique_papers": len(selected),
        "excluded_file_records": len(excluded),
        "venue_evidence": dict(sorted(evidence_counts.items())),
        "paradigm_primary": dict(sorted(paradigm_counts.items())),
        "release_gate_passes": len(selected) >= 100 and not evidence_counts.get("local_inference_pending_verification"),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
