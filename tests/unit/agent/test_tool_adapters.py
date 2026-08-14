from uuid import uuid4

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


def test_read_serialize_includes_unified_evidence_and_citations():
    payload = ReadPaperToolAdapter._serialize(_result())

    assert len(payload["passages"]) == 1
    assert payload["passages"][0]["citation"].startswith("P")
    assert payload["passages"][0]["paper_title"] == "Scene Codebook Paper"
    assert payload["elements"][0]["citation"].startswith("P")
    assert payload["elements"][0]["section_path"] == "3 Method > 3.2 Codebook"
    assert payload["elements"][0]["page_start"] == payload["elements"][0]["page_end"] == 5

    assert len(payload["evidence"]) == 2
    for entry in payload["evidence"]:
        assert entry["citation"].startswith("P")
        assert entry["paper_id"] and entry["version_id"]
        assert entry["paper_title"] == "Scene Codebook Paper"
        assert entry["section_path"] == "3 Method > 3.2 Codebook"
        assert entry["page_start"] <= entry["page_end"]
        assert entry["text"]

    passage_evidence = next(item for item in payload["evidence"] if item["chunk_id"])
    element_evidence = next(item for item in payload["evidence"] if item["element_id"])
    assert passage_evidence["text"] == "The codebook clusters scene features."
    assert passage_evidence["element_id"] is None
    assert element_evidence["text"] == "diagram"
    assert element_evidence["chunk_id"] is None
