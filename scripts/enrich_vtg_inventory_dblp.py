#!/usr/bin/env python3
"""Enrich VTG inventory candidates with reproducible DBLP bibliographic matches."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib.parse import urlencode


DBLP_ENDPOINT = "https://dblp.org/search/publ/api"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.4)
    return parser.parse_args()


def normalize_title(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = value.casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def venue_to_ccf_a(venue: str) -> str | None:
    normalized = " ".join(venue.split()).strip()
    exact = {
        "AAAI": "AAAI",
        "NeurIPS": "NEURIPS",
        "NIPS": "NEURIPS",
        "CVPR": "CVPR",
        "ICCV": "ICCV",
        "ICML": "ICML",
        "ICLR": "ICLR",
        "ACM Multimedia": "ACM MM",
        "IEEE Trans. Pattern Anal. Mach. Intell.": "TPAMI",
        "Int. J. Comput. Vis.": "IJCV",
        "IEEE Trans. Image Process.": "TIP",
        "IEEE Trans. Multim.": "TMM",
    }
    if normalized in exact:
        return exact[normalized]
    if re.fullmatch(r"ACL(?: \(\d+\))?", normalized):
        return "ACL"
    return None


def author_names(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return []
    authors = raw.get("author", [])
    if isinstance(authors, dict):
        authors = [authors]
    if not isinstance(authors, list):
        return []
    values: list[str] = []
    for author in authors:
        if isinstance(author, dict) and author.get("text"):
            values.append(str(author["text"]))
        elif isinstance(author, str):
            values.append(author)
    return values


def query_dblp(title: str) -> list[dict[str, Any]]:
    url = DBLP_ENDPOINT + "?" + urlencode({"q": title, "format": "json", "h": 8})
    result = subprocess.run(
        [
            "curl",
            "--http1.1",
            "--silent",
            "--show-error",
            "--fail",
            "--retry",
            "2",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--max-time",
            "20",
            "--user-agent",
            "paper-assistant-corpus-builder/1.0",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=75,
    )
    payload = json.loads(result.stdout)
    raw_hits = payload.get("result", {}).get("hits", {}).get("hit", [])
    if isinstance(raw_hits, dict):
        raw_hits = [raw_hits]
    return [item.get("info", {}) for item in raw_hits if isinstance(item, dict)]


def best_match(title: str, hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    source = normalize_title(title)
    best: tuple[float, dict[str, Any]] | None = None
    for hit in hits:
        hit_title = str(hit.get("title", ""))
        target = normalize_title(hit_title)
        if not source or not target:
            continue
        ratio = SequenceMatcher(None, source, target).ratio()
        containment = min(len(source), len(target)) / max(len(source), len(target)) if source in target or target in source else 0.0
        score = max(ratio, containment)
        if best is None or score > best[0]:
            best = (score, hit)
    if best is None:
        return None
    score, hit = best
    venue = str(hit.get("venue", ""))
    ee = hit.get("ee")
    if isinstance(ee, list):
        ee = ee[0] if ee else None
    return {
        "score": round(score, 4),
        "title": re.sub(r"<[^>]+>", "", str(hit.get("title", ""))).rstrip("."),
        "authors": author_names(hit.get("authors")),
        "venue": venue,
        "ccf_a_venue": venue_to_ccf_a(venue),
        "year": int(hit["year"]) if str(hit.get("year", "")).isdigit() else None,
        "type": hit.get("type"),
        "doi": hit.get("doi"),
        "dblp_url": hit.get("url"),
        "electronic_edition": ee,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.inventory)
    cache = load_cache(args.cache)
    candidates = [item for item in records if not item.get("venue_verified")]
    seen_hashes: set[str] = set()
    for index, record in enumerate(candidates, start=1):
        digest = record["sha256"]
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        if digest not in cache or "error" in cache[digest]:
            title = str(record.get("title") or record["title_inferred"])
            try:
                cache[digest] = {"query_title": title, "match": best_match(title, query_dblp(title))}
            except Exception as error:  # network failures stay auditable and resumable
                cache[digest] = {"query_title": title, "error": f"{type(error).__name__}: {error}"}
            save_cache(args.cache, cache)
            time.sleep(args.delay)
        if index % 20 == 0:
            print(f"processed={index}/{len(candidates)}", flush=True)

    for record in records:
        cached = cache.get(record["sha256"], {})
        match = cached.get("match") if isinstance(cached, dict) else None
        record["dblp_match"] = match
        if record.get("venue_verified"):
            record["bibliographic_match_status"] = "verified_ccf_a"
        elif isinstance(match, dict):
            score = float(match.get("score", 0.0))
            ccf_venue = match.get("ccf_a_venue")
            record["venue_verified"] = bool(score >= 0.86 and ccf_venue)
            record["venue_verified_value"] = ccf_venue if record["venue_verified"] else None
            record["bibliographic_match_status"] = (
                "verified_ccf_a" if record["venue_verified"] else "matched_requires_review"
            )
        else:
            record["bibliographic_match_status"] = "not_found"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "records": len(records),
        "unverified_candidates": len(candidates),
        "unique_queried": len(seen_hashes),
        "verified_ccf_a": sum(bool(item.get("venue_verified")) for item in records),
        "matched_requires_review": sum(item.get("bibliographic_match_status") == "matched_requires_review" for item in records),
        "not_found": sum(item.get("bibliographic_match_status") == "not_found" for item in records),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
