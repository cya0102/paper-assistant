import json
from pathlib import Path
from uuid import uuid4

import pytest

from paper_agent.cli import main
from paper_agent.application import build_agent_runtime
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
    output = capsys.readouterr().out
    assert "Incrementally ingest local PDFs" in output
    assert "profile-extract" in output
    assert "compare" in output


def test_ask_help_exposes_rod_mode_and_trace(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["ask", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--rag-mode" in output
    assert "retrieve-offload-delegate" in output
    assert "--trace" in output


def test_agent_runtime_defaults_to_only_composite_rod_tool(
    tmp_path: Path, monkeypatch
) -> None:
    class NeverCalledModel:
        def start(self, checkpoint, tools):
            raise AssertionError("model should not be called while assembling")

        def continue_with_tools(self, checkpoint, results, tools):
            raise AssertionError("model should not be called while assembling")

    monkeypatch.setattr(
        "paper_agent.application._language_model",
        lambda **kwargs: NeverCalledModel(),
    )
    runtime = build_agent_runtime(
        project_id=uuid4(),
        database_url="sqlite+pysqlite:///:memory:",
        redis_url="redis://localhost:6379/15",
        model="fake",
        project_root=tmp_path,
        user_id=uuid4(),
        session_id=uuid4(),
    )

    assert tuple(item["name"] for item in runtime._tools.model_specs()) == (
        "retrieve_and_analyze_knowledge",
    )
