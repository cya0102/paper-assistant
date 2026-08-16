from uuid import uuid4

import hashlib

from paper_agent.agent.tool_adapters import ReadPaperToolAdapter
from paper_agent.domain.enums import ElementType
from paper_agent.domain.reading import ReadElement, ReadPaperResult, ReadPassage


def _result() -> ReadPaperResult:
    paper_id, version_id, section_id = uuid4(), uuid4(), uuid4()
    return ReadPaperResult(
        paper_id=paper_id,
        version_id=version_id,
        title="Scene Codebook Paper",
        passages=(
            ReadPassage(
                chunk_id=uuid4(),
                section_id=section_id,
                section_path="3 Method > 3.2 Codebook",
                page_start=5,
                page_end=6,
                chunk_order=0,
                text="The codebook clusters scene features.",
                source_group_ids=(uuid4(),),
                source_block_ids=("b1",),
                element_ids=(),
            ),
        ),
        elements=(
            ReadElement(
                element_id=uuid4(),
                element_type=ElementType.FIGURE,
                section_id=section_id,
                section_path="3 Method > 3.2 Codebook",
                label="Figure 1",
                caption="Codebook overview",
                content="diagram",
                page=5,
                source_block_ids=("b2",),
            ),
        ),
    )


def test_read_serialize_keeps_single_source_of_text_and_citations():
    payload = ReadPaperToolAdapter._serialize(_result())

    assert len(payload["passages"]) == 1
    assert payload["passages"][0]["citation"].startswith("P")
    assert payload["passages"][0]["paper_title"] == "Scene Codebook Paper"
    assert payload["elements"][0]["citation"].startswith("P")
    assert payload["elements"][0]["section_path"] == "3 Method > 3.2 Codebook"
    assert payload["elements"][0]["page_start"] == payload["elements"][0]["page_end"] == 5

    # No duplicated unified evidence list: passages/elements are the only text
    # carrier and the Citation Manifest is derived from them by the materializer.
    assert "evidence" not in payload

    from paper_agent.artifacts.materializer import extract_citation_manifest

    manifest = extract_citation_manifest("read_paper", payload)
    assert len(manifest) == 2
    for entry in manifest:
        assert entry.citation_label.startswith("P")
        assert entry.paper_title == "Scene Codebook Paper"
        assert entry.section_path == "3 Method > 3.2 Codebook"
        assert entry.page_start <= entry.page_end
    passage_ref = next(item for item in manifest if item.chunk_id is not None)
    element_ref = next(item for item in manifest if item.element_id is not None)
    assert passage_ref.evidence_hash == hashlib.sha256(
        "The codebook clusters scene features.".encode()
    ).hexdigest()
    assert element_ref.chunk_id is None
