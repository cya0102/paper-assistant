from pathlib import PurePosixPath

from paper_agent.domain.errors import ErrorCode
from paper_agent.ingestion.scanner import DirectoryScanner


def test_scanner_discovers_pdfs_recursively_in_stable_order(tmp_path) -> None:
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers" / "B.PDF").write_bytes(b"b")
    (tmp_path / "papers" / "a.pdf").write_bytes(b"a")
    (tmp_path / "papers" / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / ".paper-agent" / "parsed").mkdir(parents=True)
    (tmp_path / ".paper-agent" / "parsed" / "hidden.pdf").write_bytes(b"hidden")

    result = DirectoryScanner().scan(tmp_path.resolve())

    assert [item.relative_path for item in result.files] == [
        PurePosixPath("papers/a.pdf"),
        PurePosixPath("papers/B.PDF"),
    ]
    assert result.issues == ()


def test_scanner_rejects_target_outside_project(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")

    result = DirectoryScanner().scan(project.resolve(), (outside.resolve(),))

    assert result.files == ()
    assert result.issues[0].code == ErrorCode.PATH_OUTSIDE_PROJECT

