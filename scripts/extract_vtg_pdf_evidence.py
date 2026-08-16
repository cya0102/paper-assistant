#!/usr/bin/env python3
"""Extract auditable venue/DOI evidence from the first pages of selected PDFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any


VENUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("TPAMI", re.compile(r"IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE", re.I)),
    ("IJCV", re.compile(r"INTERNATIONAL JOURNAL OF COMPUTER VISION", re.I)),
    ("TIP", re.compile(r"IEEE TRANSACTIONS ON IMAGE PROCESSING", re.I)),
    ("TMM", re.compile(r"IEEE TRANSACTIONS ON MULTIMEDIA", re.I)),
    ("ACM MM", re.compile(r"ACM INTERNATIONAL CONFERENCE ON MULTIMEDIA|PROCEEDINGS OF THE ACM.*MULTIMEDIA", re.I | re.S)),
    ("NEURIPS", re.compile(r"NEURAL INFORMATION PROCESSING SYSTEMS|\bNEURIPS\b", re.I)),
    ("CVPR", re.compile(r"CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION|\bCVPR\b", re.I)),
    ("ICCV", re.compile(r"INTERNATIONAL CONFERENCE ON COMPUTER VISION|\bICCV\b", re.I)),
    ("ICML", re.compile(r"INTERNATIONAL CONFERENCE ON MACHINE LEARNING|\bICML\b", re.I)),
    ("ICLR", re.compile(r"INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS|\bICLR\b", re.I)),
    ("AAAI", re.compile(r"AAAI CONFERENCE ON ARTIFICIAL INTELLIGENCE|\bAAAI-?\d{2}\b", re.I)),
    ("ACL", re.compile(r"ANNUAL MEETING OF THE ASSOCIATION FOR COMPUTATIONAL LINGUISTICS|\bACL 20\d{2}\b", re.I)),
)

DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=3)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pdf_text(path: Path, pages: int) -> str:
    result = subprocess.run(
        ["pdftotext", "-f", "1", "-l", str(pages), "-layout", str(path), "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else ""


def clean_doi(value: str) -> str:
    return value.rstrip(".,;:)]}")


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    for record in records:
        text = pdf_text(Path(record["source_path"]), args.pages)
        doi_values = list(dict.fromkeys(clean_doi(item) for item in DOI_PATTERN.findall(text)))
        venue_values = [venue for venue, pattern in VENUE_PATTERNS if pattern.search(text)]
        expected = record.get("venue_verified_value") or record.get("venue") or record.get("venue_inferred")
        record["pdf_evidence"] = {
            "doi_candidates": doi_values,
            "venue_markers": venue_values,
            "expected_venue_marker_found": expected in venue_values,
            "pages_examined": min(args.pages, int(record.get("pages") or args.pages)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "records": len(records),
        "with_doi": sum(bool(item["pdf_evidence"]["doi_candidates"]) for item in records),
        "with_expected_venue_marker": sum(item["pdf_evidence"]["expected_venue_marker_found"] for item in records),
        "pending_external_but_pdf_marker_found": sum(
            item.get("venue_evidence") == "local_inference_pending_verification"
            and item["pdf_evidence"]["expected_venue_marker_found"]
            for item in records
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
