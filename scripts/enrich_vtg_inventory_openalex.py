#!/usr/bin/env python3
"""Enrich VTG corpus candidates with OpenAlex publication metadata."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib.parse import urlencode


OPENALEX_ENDPOINT = "https://api.openalex.org/works"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.0)
    return parser.parse_args()


def normalize_title(value: str) -> str:
    value = value.casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def title_score(source: str, target: str) -> float:
    left, right = normalize_title(source), normalize_title(target)
    if not left or not right:
        return 0.0
    ratio = SequenceMatcher(None, left, right).ratio()
    containment = min(len(left), len(right)) / max(len(left), len(right)) if left in right or right in left else 0.0
    return max(ratio, containment)


def source_names(work: dict[str, Any]) -> list[str]:
    values: list[str] = []
    locations = [work.get("primary_location"), *(work.get("locations") or [])]
    for location in locations:
        if not isinstance(location, dict):
            continue
        raw_name = location.get("raw_source_name")
        if raw_name:
            values.append(str(raw_name))
        source = location.get("source")
        if isinstance(source, dict) and source.get("display_name"):
            values.append(str(source["display_name"]))
    return list(dict.fromkeys(values))


def venue_to_ccf_a(names: list[str]) -> str | None:
    joined = "\n".join(names)
    patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("TPAMI", re.compile(r"IEEE Transactions on Pattern Analysis and Machine Intelligence", re.I)),
        ("IJCV", re.compile(r"^International Journal of Computer Vision$", re.I | re.M)),
        ("TIP", re.compile(r"IEEE Transactions on Image Processing", re.I)),
        ("TMM", re.compile(r"IEEE Transactions on Multimedia", re.I)),
        ("ACM MM", re.compile(r"(?:Proceedings of )?(?:the )?ACM International Conference on Multimedia", re.I)),
        ("NEURIPS", re.compile(r"(?:Advances in|Conference on) Neural Information Processing Systems", re.I)),
        ("CVPR", re.compile(r"(?:IEEE/CVF |IEEE )?Conference on Computer Vision and Pattern Recognition|\bCVPR\b", re.I)),
        ("ICCV", re.compile(r"International Conference on Computer Vision|\bICCV\b", re.I)),
        ("ICML", re.compile(r"International Conference on Machine Learning|\bICML\b", re.I)),
        ("ICLR", re.compile(r"International Conference on Learning Representations|\bICLR\b", re.I)),
        ("AAAI", re.compile(r"AAAI Conference on Artificial Intelligence", re.I)),
        ("ACL", re.compile(r"Annual Meeting of the Association for Computational Linguistics", re.I)),
    )
    for venue, pattern in patterns:
        if pattern.search(joined):
            return venue
    return None


def run_curl(url: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "curl",
            "--http1.1",
            "--silent",
            "--show-error",
            "--fail",
            "--retry",
            "2",
            "--retry-delay",
            "1",
            "--max-time",
            "25",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=35,
    )
    return json.loads(result.stdout)


def query_openalex(title: str) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "search": title,
            "per-page": 8,
            "select": "id,title,publication_year,doi,type,authorships,primary_location,locations,open_access",
        }
    )
    payload = run_curl(f"{OPENALEX_ENDPOINT}?{query}")
    results = payload.get("results", [])
    return results if isinstance(results, list) else []


def authors(work: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") if isinstance(authorship, dict) else None
        if isinstance(author, dict) and author.get("display_name"):
            values.append(str(author["display_name"]))
    return values


def location_value(work: dict[str, Any], key: str) -> Any:
    primary = work.get("primary_location")
    return primary.get(key) if isinstance(primary, dict) else None


def best_match(title: str, works: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [(title_score(title, str(work.get("title", ""))), work) for work in works]
    if not scored:
        return None
    score, work = max(scored, key=lambda item: item[0])
    names = source_names(work)
    return {
        "score": round(score, 4),
        "openalex_id": work.get("id"),
        "title": work.get("title"),
        "authors": authors(work),
        "publication_year": work.get("publication_year"),
        "doi": work.get("doi"),
        "type": work.get("type"),
        "source_names": names,
        "ccf_a_venue": venue_to_ccf_a(names),
        "landing_page_url": location_value(work, "landing_page_url"),
        "pdf_url": location_value(work, "pdf_url"),
        "location_version": location_value(work, "version"),
        "is_accepted": location_value(work, "is_accepted"),
        "is_published": location_value(work, "is_published"),
        "open_access": work.get("open_access"),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_cache(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch_one(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    digest, title = record["sha256"], str(record["title_inferred"])
    try:
        match = best_match(title, query_openalex(title))
        return digest, {"query_title": title, "match": match}
    except Exception as error:
        return digest, {"query_title": title, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.inventory)
    cache = load_cache(args.cache)
    unique_candidates: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("topic_signals"):
            unique_candidates.setdefault(record["sha256"], record)
    pending = [
        record
        for digest, record in unique_candidates.items()
        if digest not in cache or "error" in cache[digest]
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_one, record) for record in pending]
        for index, future in enumerate(as_completed(futures), start=1):
            digest, value = future.result()
            cache[digest] = value
            if args.delay:
                time.sleep(args.delay)
            if index % 10 == 0 or index == len(futures):
                save_cache(args.cache, cache)
                print(f"processed={index}/{len(futures)}", flush=True)

    for record in records:
        cached = cache.get(record["sha256"], {})
        match = cached.get("match") if isinstance(cached, dict) else None
        record["openalex_match"] = match
        if isinstance(match, dict):
            verified = bool(
                float(match.get("score", 0.0)) >= 0.86
                and match.get("ccf_a_venue")
                and match.get("is_published") is not False
            )
            record["venue_verified"] = verified
            record["venue_verified_value"] = match.get("ccf_a_venue") if verified else None
            record["bibliographic_match_status"] = (
                "verified_ccf_a" if verified else "matched_requires_review"
            )
        elif record.get("topic_signals"):
            record["bibliographic_match_status"] = "not_found"
        else:
            record["bibliographic_match_status"] = "not_queried"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "records": len(records),
        "unique_topic_candidates": len(unique_candidates),
        "verified_ccf_a": sum(bool(item.get("venue_verified")) for item in records),
        "matched_requires_review": sum(item.get("bibliographic_match_status") == "matched_requires_review" for item in records),
        "not_found": sum(item.get("bibliographic_match_status") == "not_found" for item in records),
        "cache_errors": sum("error" in value for value in cache.values()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
