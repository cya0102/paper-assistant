from paper_agent.database import upgrade_database


def test_upgrade_database_marks_explicit_url_as_override(monkeypatch) -> None:
    captured = {}

    def fake_upgrade(config, revision):
        captured["revision"] = revision
        captured["configured_url"] = config.get_main_option("sqlalchemy.url")
        captured["override_url"] = config.attributes.get("database_url_override")

    monkeypatch.setattr("alembic.command.upgrade", fake_upgrade)

    database_url = "postgresql+psycopg://test-user:test-pass@localhost/test-db"
    upgrade_database(database_url)

    assert captured == {
        "revision": "head",
        "configured_url": database_url,
        "override_url": database_url,
    }
