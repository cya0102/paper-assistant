import json

import pytest

from paper_agent.cli import main
from paper_agent.project_manifest import ProjectManifestStore


def test_project_manifest_is_stable_and_round_trips(tmp_path) -> None:
    store = ProjectManifestStore(tmp_path)

    first = store.load_or_create()
    second = store.load_or_create()

    assert first == second
    payload = json.loads((tmp_path / ".paper-agent" / "project.json").read_text())
    assert payload["project_id"] == str(first.project_id)


def test_cli_help_is_available(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "Incrementally ingest local PDFs" in capsys.readouterr().out

