#!/usr/bin/env python3
"""Build a reproducible inventory for a local video temporal grounding corpus.

This script is deliberately conservative: filename/content heuristics only create
review candidates. They never mark a paper as venue-verified or eligible for a
frozen corpus release.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


CCF_A_VENUES = (
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
)

VENUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("TPAMI", re.compile(r"\b(?:TPAMI|T-PAMI|TRANSACTIONS ON PATTERN ANALYSIS)\b", re.I)),
    ("IJCV", re.compile(r"\b(?:IJCV|INTERNATIONAL JOURNAL OF COMPUTER VISION)\b", re.I)),
    ("TMM", re.compile(r"\b(?:TMM|TRANSACTIONS ON MULTIMEDIA)\b", re.I)),
    ("TIP", re.compile(r"\b(?:TIP|TRANSACTIONS ON IMAGE PROCESSING)\b", re.I)),
    ("ACM MM", re.compile(r"\b(?:ACM\s*MM|ACM MULTIMEDIA|ACMMM)\b", re.I)),
    ("NEURIPS", re.compile(r"\b(?:NEURIPS|NIPS|NEURAL INFORMATION PROCESSING SYSTEMS)\b", re.I)),
    ("CVPR", re.compile(r"\bCVPR\b|COMPUTER VISION AND PATTERN RECOGNITION", re.I)),
    ("ICCV", re.compile(r"\bICCV\b|INTERNATIONAL CONFERENCE ON COMPUTER VISION", re.I)),
    ("ICML", re.compile(r"\bICML\b|INTERNATIONAL CONFERENCE ON MACHINE LEARNING", re.I)),
    ("ICLR", re.compile(r"\bICLR\b|INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS", re.I)),
    ("AAAI", re.compile(r"\bAAAI\b", re.I)),
    ("ACL", re.compile(r"\bACL\b|ASSOCIATION FOR COMPUTATIONAL LINGUISTICS", re.I)),
)

TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("video_temporal_grounding", re.compile(r"\bvideo temporal grounding\b", re.I)),
    ("temporal_sentence_grounding", re.compile(r"\btemporal sentence grounding\b", re.I)),
    ("video_moment_localization", re.compile(r"\bvideo moment locali[sz]ation\b", re.I)),
    ("video_moment_retrieval", re.compile(r"\bvideo moment retrieval\b", re.I)),
    ("natural_language_video_localization", re.compile(r"\bnatural language video locali[sz]ation\b", re.I)),
    ("temporal_language_grounding", re.compile(r"\btemporal(?:ly)? language grounding\b", re.I)),
    ("sentence_grounding_in_video", re.compile(r"\bsentence grounding in videos?\b", re.I)),
    ("moment_localization_nl", re.compile(r"\bmoment locali[sz]ation (?:with|of|via) natural language\b", re.I)),
    ("temporal_activity_localization_language", re.compile(r"\btemporal activity locali[sz]ation via language\b", re.I)),
    ("query_based_video_localization", re.compile(r"\bquery[- ]based video locali[sz]ation\b", re.I)),
    ("natural_language_query_video", re.compile(r"\b(?:temporal|moment) locali[sz]ation of (?:a )?natural[- ]language quer(?:y|ies) in video", re.I)),
    ("language_based_temporal_localization", re.compile(r"\blanguage[- ]based temporal locali[sz]ation\b", re.I)),
    ("video_sentence_grounding", re.compile(r"\bvideo sentence grounding\b", re.I)),
    ("video_grounding", re.compile(r"\bvideo grounding\b", re.I)),
)

METHOD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("weakly_supervised", re.compile(r"\bweakly[- ]supervised\b|\bpoint annotations?\b|\bsingle[- ]frame supervision\b|\bsparse supervision\b", re.I)),
    ("zero_shot", re.compile(r"\bzero[- ]shot\b|\btraining[- ]free\b|\blanguage[- ]free training\b", re.I)),
    ("llm_based", re.compile(r"\b(?:LLM|MLLM|large language model|large vision[- ]language model|video[- ]LLM)\b", re.I)),
    ("fully_supervised", re.compile(r"\bfully[- ]supervised\b|\bstrongly[- ]supervised\b", re.I)),
)

EXCLUSION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("workshop", re.compile(r"\b(?:WORKSHOP|ICCVW|CVPRW)\b", re.I)),
    ("pure_text_video_retrieval", re.compile(r"\btext[- ]to[- ]video retrieval\b|\btext video retrieval\b", re.I)),
    ("video_qa", re.compile(r"\bvideo question answering\b|\bvideo QA\b", re.I)),
    ("pure_visual_grounding", re.compile(r"\bvisual grounding\b", re.I)),
    ("anomaly_detection", re.compile(r"\bvideo anomaly\b|\blocali[sz]ing anomalies\b", re.I)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=2)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(args: list[str]) -> str:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def pdf_page_count(path: Path) -> int | None:
    info = command_output(["pdfinfo", str(path)])
    match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    return int(match.group(1)) if match else None


def pdf_text(path: Path, pages: int) -> str:
    return command_output(
        ["pdftotext", "-f", "1", "-l", str(pages), "-layout", str(path), "-"]
    )


def canonical_title(filename: str) -> str:
    title = Path(filename).stem
    title = re.sub(r"^(?:AAAI|ACL|CVPR|ICCV|ICML|ICLR|NEURIPS|ACMMM|MM|TPAMI|TMM|TCSVT|TIP|ARXIV|PR)[-_ ]*\d{2,4}[-_ ]*", "", title, flags=re.I)
    title = re.sub(r"(?:_CVPR_\d{4}_paper|-Paper-Conference|\((?:AAAI|CVPR|ICCV|ICLR|TMM)?\d{4}\))$", "", title, flags=re.I)
    title = re.sub(r"[_]+", " ", title)
    return " ".join(title.split()).strip(" -")


def normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def infer_matches(text: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> list[str]:
    return [label for label, pattern in patterns if pattern.search(text)]


def infer_venue(filename: str, text: str) -> tuple[str | None, str]:
    for venue, pattern in VENUE_PATTERNS:
        if pattern.search(filename):
            return venue, "filename"
    for venue, pattern in VENUE_PATTERNS:
        if pattern.search(text):
            return venue, "first_pages"
    return None, "unknown"


def infer_year(filename: str, text: str) -> int | None:
    filename_years = re.findall(r"(?<!\d)(20\d{2})(?!\d)", filename)
    if filename_years:
        return int(filename_years[0])
    short = re.search(r"\b(?:AAAI|ACL|CVPR|ICCV|ICML|ICLR|NEURIPS|MM|TPAMI|TMM|TIP)[-_ ]?(\d{2})\b", filename, re.I)
    if short:
        return 2000 + int(short.group(1))
    text_years = re.findall(r"(?:©|Copyright|Proceedings of|Conference on).*?\b(20\d{2})\b", text, re.I | re.S)
    return int(text_years[0]) if text_years else None


def build_inventory(source: Path, pages: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(source.glob("*.pdf"), key=lambda item: item.name.casefold()):
        text = pdf_text(path, pages)
        title = canonical_title(path.name)
        title_text = f"{path.name}\n{title}"
        combined = f"{title_text}\n{text}"
        venue, venue_source = infer_venue(path.name, text)
        title_topics = infer_matches(title_text, TOPIC_PATTERNS)
        text_topics = infer_matches(text, TOPIC_PATTERNS)
        methods = infer_matches(combined, METHOD_PATTERNS)
        exclusions = infer_matches(combined, EXCLUSION_PATTERNS)
        digest = sha256(path)
        records.append(
            {
                "paper_key": digest[:16],
                "source_filename": path.name,
                "source_path": str(path),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "pages": pdf_page_count(path),
                "title_inferred": title,
                "title_normalized": normalized_title(title),
                "year_inferred": infer_year(path.name, text),
                "venue_inferred": venue,
                "venue_inference_source": venue_source,
                "venue_verified": False,
                "ccf_version": "2026-v7",
                "ccf_a_candidate": venue in CCF_A_VENUES,
                "topic_title_signals": title_topics,
                "topic_text_signals": text_topics,
                "topic_signals": list(dict.fromkeys(title_topics + text_topics)),
                "method_signals": methods,
                "exclusion_signals": exclusions,
                "review_status": "pending",
            }
        )
    return records


def exact_duplicates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["sha256"]].append(record)
    return [
        {
            "kind": "exact_sha256",
            "sha256": digest,
            "files": [item["source_filename"] for item in group],
        }
        for digest, group in sorted(groups.items())
        if len(group) > 1
    ]


def likely_title_duplicates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(record["sha256"], record)
    values = list(unique.values())
    groups: list[dict[str, Any]] = []
    for index, left in enumerate(values):
        if len(left["title_normalized"]) < 18:
            continue
        for right in values[index + 1 :]:
            if len(right["title_normalized"]) < 18:
                continue
            ratio = SequenceMatcher(
                None, left["title_normalized"], right["title_normalized"]
            ).ratio()
            if ratio >= 0.94:
                groups.append(
                    {
                        "kind": "likely_same_title",
                        "similarity": round(ratio, 4),
                        "files": [left["source_filename"], right["source_filename"]],
                        "sha256": [left["sha256"], right["sha256"]],
                    }
                )
    return groups


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = build_inventory(args.source, args.pages)
    inventory_path = args.output / "inventory.jsonl"
    with inventory_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    duplicates = exact_duplicates(records) + likely_title_duplicates(records)
    write_json(args.output / "duplicates.json", duplicates)
    unique_hashes = {record["sha256"] for record in records}
    summary = {
        "source": str(args.source),
        "pdf_files": len(records),
        "unique_hashes": len(unique_hashes),
        "ccf_a_candidates": sum(record["ccf_a_candidate"] for record in records),
        "topic_candidates": sum(bool(record["topic_signals"]) for record in records),
        "ccf_a_and_topic_candidates": sum(
            record["ccf_a_candidate"] and bool(record["topic_signals"])
            for record in records
        ),
        "exact_duplicate_groups": sum(item["kind"] == "exact_sha256" for item in duplicates),
        "likely_title_duplicate_pairs": sum(item["kind"] == "likely_same_title" for item in duplicates),
    }
    write_json(args.output / "inventory-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
