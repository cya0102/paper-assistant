#!/usr/bin/env python3
"""Build a frozen, local-only VTG corpus release from curated records."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


VERSION = "1.0.0"
CORPUS_ID = "vtg-ccf-a"
CCF_EDITION = "2026-v7"
CCF_URL = "https://www.ccf.org.cn/Academic_Evaluation/By_category/"
MANUAL_OFFICIAL_EVIDENCE = {
    "vtg llm integrating timestamp knowledge into video llms for enhanced video temporal grounding": {
        "kind": "official_proceedings_page",
        "url": "https://ojs.aaai.org/index.php/AAAI/article/view/32341",
        "note": "AAAI-25 main technical track; DOI 10.1609/AAAI.V39I3.32341",
    }
}
TITLE_CORRECTIONS = {
    "vtg llm intergrating timestamp knowledge into video llms for enhanced video temporal grounding":
        "VTG-LLM: Integrating Timestamp Knowledge into Video LLMs for Enhanced Video Temporal Grounding",
    "knowing your target targer aware transformer makes better spatio temporal video grounding":
        "Knowing Your Target: Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding",
    "end to end multi modal video temporal grounding paper":
        "End-to-end Multi-modal Video Temporal Grounding",
    "measure twice cut once a semanticoriented approach to video temporal localization with video llms":
        "Measure Twice, Cut Once: A Semantic-Oriented Approach to Video Temporal Localization with Video LLMs",
    "temporal sentence grounding with relevance":
        "Temporal Sentence Grounding with Relevance Feedback in Videos",
    "prot g untrimmed pretraining for video temporal grounding by video temporal grounding":
        "ProTeGe: Untrimmed Pretraining for Video Temporal Grounding by Video Temporal Grounding",
}
YEAR_CORRECTIONS = {
    "invert4tvg a temporal video grounding framework with inversion tasks preserving action understanding ability": 2026,
    "knowing your target target aware transformer makes better spatio temporal video grounding": 2025,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalize_title(value: str) -> str:
    value = value.casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_metadata_title(path: Path) -> str | None:
    result = subprocess.run(["pdfinfo", str(path)], check=False, capture_output=True, text=True)
    if result.returncode:
        return None
    match = re.search(r"^Title:\s*(.+)$", result.stdout, re.M)
    value = " ".join(match.group(1).split()) if match else ""
    if len(value) < 18 or value.casefold() in {"untitled", "microsoft word"}:
        return None
    return value


def canonical_title(selected: dict[str, Any], evidence: dict[str, Any]) -> str:
    dblp = evidence.get("dblp_match")
    if isinstance(dblp, dict) and float(dblp.get("score") or 0.0) >= 0.86:
        value = str(dblp.get("title"))
    else:
        openalex = selected.get("openalex_match")
        if isinstance(openalex, dict) and float(openalex.get("score") or 0.0) >= 0.86:
            value = str(openalex.get("title"))
        else:
            value = pdf_metadata_title(Path(selected["source_path"])) or str(selected["title"])
    return TITLE_CORRECTIONS.get(normalize_title(value), value)


def publication_year(selected: dict[str, Any], evidence: dict[str, Any], title: str) -> int | None:
    corrected = YEAR_CORRECTIONS.get(normalize_title(title))
    if corrected:
        return corrected
    dblp = evidence.get("dblp_match")
    openalex = selected.get("openalex_match")
    values = [
        dblp.get("year") if isinstance(dblp, dict) else None,
        openalex.get("publication_year") if isinstance(openalex, dict) else None,
        selected.get("year_inferred"),
    ]
    return next((int(value) for value in values if value), None)


def authors(selected: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    dblp = evidence.get("dblp_match")
    openalex = selected.get("openalex_match")
    if isinstance(dblp, dict) and dblp.get("authors"):
        return list(dblp["authors"])
    if isinstance(openalex, dict) and openalex.get("authors"):
        return list(openalex["authors"])
    return []


def venue_evidence(selected: dict[str, Any], evidence: dict[str, Any], title: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    openalex = selected.get("openalex_match")
    if selected.get("venue_verified") and isinstance(openalex, dict):
        values.append(
            {
                "kind": "openalex_registry",
                "id": openalex.get("openalex_id"),
                "url": openalex.get("landing_page_url"),
                "matched_title": openalex.get("title"),
                "score": openalex.get("score"),
            }
        )
    dblp = evidence.get("dblp_match")
    if evidence.get("venue_verified") and isinstance(dblp, dict):
        values.append(
            {
                "kind": "dblp_registry",
                "url": dblp.get("dblp_url"),
                "electronic_edition": dblp.get("electronic_edition"),
                "matched_title": dblp.get("title"),
                "score": dblp.get("score"),
            }
        )
    pdf = evidence.get("pdf_evidence") or {}
    if pdf.get("expected_venue_marker_found"):
        values.append(
            {
                "kind": "published_pdf_marker",
                "venue_markers": pdf.get("venue_markers") or [],
                "doi_candidates": pdf.get("doi_candidates") or [],
                "pages_examined": pdf.get("pages_examined"),
            }
        )
    manual = MANUAL_OFFICIAL_EVIDENCE.get(normalize_title(title))
    if manual:
        values.append(manual)
    return values


def slug(value: str, limit: int = 72) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized[:limit].rstrip("-") or "paper"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def evenly_pick(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count >= len(records):
        return list(records)
    if count == 1:
        return [records[len(records) // 2]]
    indices = [round(index * (len(records) - 1) / (count - 1)) for index in range(count)]
    return [records[index] for index in indices]


def stratified_split(records: list[dict[str, Any]], quotas: dict[str, int]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for paradigm, count in quotas.items():
        group = [item for item in records if item["paradigm_primary"] == paradigm]
        picked.extend(evenly_pick(group, count))
    return sorted(picked, key=lambda item: (item["year"] or 9999, item["title"].casefold()))


def relative_split_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": record["paper_id"],
        "title": record["title"],
        "year": record["year"],
        "venue": record["venue"],
        "paradigm_primary": record["paradigm_primary"],
        "file": record["file"],
        "sha256": record["sha256"],
    }


def main() -> None:
    args = parse_args()
    selected = load_jsonl(args.selected)
    evidence_by_hash = {item["sha256"]: item for item in load_jsonl(args.evidence)}
    excluded = load_jsonl(args.excluded)
    root = args.output
    pdf_root = root / "pdfs"
    split_root = root / "splits"
    pdf_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for selected_record in selected:
        evidence = evidence_by_hash[selected_record["sha256"]]
        source_path = Path(selected_record["source_path"])
        title = canonical_title(selected_record, evidence)
        proof = venue_evidence(selected_record, evidence, title)
        if not proof:
            raise SystemExit(f"venue gate failed: {title}")
        year = publication_year(selected_record, evidence, title)
        venue = selected_record["venue"]
        paper_id = f"{CORPUS_ID}:{selected_record['sha256'][:16]}"
        destination_name = f"{year or 'unknown'}_{slug(venue, 12)}_{slug(title)}_{selected_record['sha256'][:8]}.pdf"
        destination = pdf_root / destination_name
        shutil.copy2(source_path, destination)
        if file_sha256(destination) != selected_record["sha256"]:
            raise SystemExit(f"checksum mismatch after copy: {destination}")
        openalex = selected_record.get("openalex_match") or {}
        dblp = evidence.get("dblp_match") or {}
        doi_candidates = list(
            dict.fromkeys(
                value
                for value in [
                    dblp.get("doi"),
                    openalex.get("doi"),
                    *((evidence.get("pdf_evidence") or {}).get("doi_candidates") or []),
                ]
                if value
            )
        )
        manifest.append(
            {
                "corpus_id": CORPUS_ID,
                "corpus_version": VERSION,
                "paper_id": paper_id,
                "title": title,
                "authors": authors(selected_record, evidence),
                "year": year,
                "venue": venue,
                "ccf_edition": CCF_EDITION,
                "ccf_category": "A",
                "task_scope": "video_temporal_grounding",
                "paradigm_primary": selected_record["paradigm_primary"],
                "paradigm_tags": selected_record["paradigm_tags"],
                "taxonomy_note": "point/sparse/single-frame supervision is grouped under weakly_supervised",
                "file": f"pdfs/{destination_name}",
                "sha256": selected_record["sha256"],
                "size_bytes": selected_record["size_bytes"],
                "pages": selected_record["pages"],
                "doi_candidates": doi_candidates,
                "openalex_id": openalex.get("openalex_id"),
                "source_original_path": str(source_path),
                "source_storage": "local_only",
                "venue_evidence": proof,
                "review_status": "included",
            }
        )
    manifest.sort(key=lambda item: (item["year"] or 9999, item["title"].casefold()))

    smoke = stratified_split(
        manifest,
        {"fully_supervised": 6, "weakly_supervised": 3, "zero_shot": 1, "llm_based": 2},
    )
    regression = stratified_split(
        manifest,
        {"fully_supervised": 24, "weakly_supervised": 9, "zero_shot": 3, "llm_based": 4},
    )
    smoke_ids = {item["paper_id"] for item in smoke}
    regression_ids = {item["paper_id"] for item in regression}
    if not smoke_ids.issubset(regression_ids):
        # Preserve the exact 40-paper budget while forcing smoke to be a subset.
        additions = [item for item in smoke if item["paper_id"] not in regression_ids]
        removable = [item for item in reversed(regression) if item["paper_id"] not in smoke_ids]
        for addition, removal in zip(additions, removable):
            regression.remove(removal)
            regression.append(addition)
        regression.sort(key=lambda item: (item["year"] or 9999, item["title"].casefold()))

    write_jsonl(root / "manifest.jsonl", manifest)
    write_jsonl(split_root / "smoke-12.jsonl", [relative_split_record(item) for item in smoke])
    write_jsonl(split_root / "regression-40.jsonl", [relative_split_record(item) for item in regression])
    write_jsonl(split_root / "full-101.jsonl", [relative_split_record(item) for item in manifest])

    hard_negative_pool = [
        item
        for item in excluded
        if item.get("exclusion_reason") in {
            "outside_title_scope",
            "video_question_grounding",
            "generic_video_reasoning",
            "generic_video_description",
            "survey",
            "workshop_or_non_regular_track",
            "non_ccf_a_wacv",
        }
    ]
    hard_negatives = sorted(
        hard_negative_pool,
        key=lambda item: (item.get("exclusion_reason", ""), item.get("title_inferred", "").casefold()),
    )[:20]
    write_jsonl(root / "exclusions.jsonl", excluded)
    write_jsonl(split_root / "hard-negatives-20.jsonl", hard_negatives)

    policy = f"""# VTG CCF-A corpus inclusion policy

Version: `{VERSION}`  
CCF list: `{CCF_EDITION}` ({CCF_URL})

## Task boundary

Include papers whose model or benchmark takes a natural-language query and a video and localizes the corresponding temporal interval. Spatio-temporal grounding is included only when temporal localization is a core output. Video paragraph grounding is included because it predicts temporal intervals for multiple related sentence queries.

Exclude pure text-video retrieval, captioning, VQA, spatial-only/object grounding, anomaly localization, surveys, workshop/short/demo/findings papers, and venues outside the CCF-A whitelist.

## Supervision taxonomy

- `fully_supervised`: timestamp/boundary supervision, unless a more specific tag applies.
- `weakly_supervised`: weak, point, sparse, single-frame, semi-supervised, self-supervised, unsupervised, or unlabeled-data supervision. Point/sparse supervision is never a separate top-level class.
- `zero_shot`: zero-shot, training-free, language-free, or open-vocabulary grounding.
- `llm_based`: LLM/MLLM/VLM-based grounding. This is the primary class when a paper is also weak or zero-shot; all applicable labels remain in `paradigm_tags`.

## Venue gate

A paper must have at least one auditable proof: OpenAlex/DBLP bibliographic match, a CCF-A venue marker or DOI in the published PDF, or an official proceedings page. CCF-A eligibility is evaluated against the 2026 seventh-edition list; only Full/Regular conference papers are eligible.

## Versioning and storage

This release is immutable. Any PDF, metadata, taxonomy, or split change requires a new semantic version. PDFs are copied into the release and addressed by SHA-256. The corpus is local-only and is not a redistribution package; copyright remains with the paper authors/publishers.
"""
    (root / "inclusion-policy.md").write_text(policy, encoding="utf-8")

    readme = f"""# {CORPUS_ID} v{VERSION}

Frozen local corpus for regression testing a paper/RAG agent on video temporal grounding.

- Core papers: {len(manifest)} unique PDFs
- Splits: smoke 12, regression 40, full {len(manifest)}
- CCF policy: 2026 seventh edition, category A, Full/Regular papers only
- Integrity: `checksums.sha256`
- Metadata: `manifest.jsonl`
- Inclusion and taxonomy rules: `inclusion-policy.md`

Use the smoke split for every commit, regression-40 for pre-merge quality gates, and full-{len(manifest)} for release/nightly evaluation. `hard-negatives-20.jsonl` contains metadata-only near-miss examples and does not add non-task PDFs to the core corpus.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    checksum_lines = [f"{item['sha256']}  {item['file']}" for item in manifest]
    (root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    counts = Counter(item["paradigm_primary"] for item in manifest)
    venue_counts = Counter(item["venue"] for item in manifest)
    evidence_counts = Counter(
        next(proof["kind"] for proof in item["venue_evidence"] if proof.get("kind"))
        for item in manifest
    )
    lock = {
        "corpus_id": CORPUS_ID,
        "version": VERSION,
        "created_on": date.today().isoformat(),
        "paper_count": len(manifest),
        "ccf_edition": CCF_EDITION,
        "ccf_source": CCF_URL,
        "paradigm_primary_counts": dict(sorted(counts.items())),
        "venue_counts": dict(sorted(venue_counts.items())),
        "primary_evidence_counts": dict(sorted(evidence_counts.items())),
        "splits": {"smoke": 12, "regression": 40, "full": len(manifest), "hard_negatives_metadata_only": len(hard_negatives)},
        "manifest_sha256": file_sha256(root / "manifest.jsonl"),
        "policy_sha256": file_sha256(root / "inclusion-policy.md"),
        "gate_passed": len(manifest) >= 100 and all(item["venue_evidence"] for item in manifest),
    }
    (root / "corpus.lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
