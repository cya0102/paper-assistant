"""Paper Agent ingestion, search, read, and question-answering CLI."""

import argparse
import json
import os
from pathlib import Path
from typing import NoReturn
from uuid import UUID, uuid4

from paper_agent.application import (
    build_agent_runtime,
    build_ingestion_pipeline,
    build_read_paper_service,
    build_search_knowledge_service,
)
from paper_agent.database import database_status, upgrade_database
from paper_agent.domain.ingestion import IngestionRequest
from paper_agent.domain.enums import ElementType
from paper_agent.domain.reading import ReadPaperRequest
from paper_agent.domain.retrieval import SearchRequest, SearchScope
from paper_agent.project_manifest import ProjectManifestStore


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return _init(args)
        if args.command == "ingest":
            return _ingest(args)
        if args.command == "status":
            return _status(args)
        if args.command == "search":
            return _search(args)
        if args.command == "read":
            return _read(args)
        if args.command == "ask":
            return _ask(args)
        if args.command == "db-upgrade":
            upgrade_database(_database_url(args.database_url))
            print("Database upgraded to head.")
            return 0
    except Exception as error:
        _fail(str(error))
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-agent")
    subparsers = parser.add_subparsers(dest="command")
    init = subparsers.add_parser("init", help="Initialize project identity and database schema")
    _root_argument(init)
    _database_argument(init)
    ingest = subparsers.add_parser("ingest", help="Incrementally ingest local PDFs")
    _root_argument(ingest)
    _database_argument(ingest)
    ingest.add_argument("paths", nargs="*", type=Path)
    ingest.add_argument("--no-recursive", action="store_true")
    ingest.add_argument("--force-reindex", action="store_true")
    ingest.add_argument("--parser", choices=("auto", "pymupdf", "poppler"), default="auto")
    status = subparsers.add_parser("status", help="Show project ingestion counts")
    _root_argument(status)
    _database_argument(status)
    search = subparsers.add_parser("search", help="Search indexed paper evidence")
    _root_argument(search)
    _database_argument(search)
    search.add_argument("query")
    search.add_argument("--max-evidence", type=int, default=5)
    read = subparsers.add_parser("read", help="Read a paper Section, pages, or Element")
    _database_argument(read)
    _root_argument(read)
    read.add_argument("paper_id", type=UUID)
    read.add_argument("--section-id", type=UUID)
    read.add_argument("--pages", nargs=2, type=int)
    read.add_argument("--element-id", type=UUID)
    read.add_argument("--element-type", action="append", choices=tuple(item.value for item in ElementType))
    ask = subparsers.add_parser("ask", help="Run the recoverable Agent tool loop")
    _root_argument(ask)
    _database_argument(ask)
    ask.add_argument("query")
    ask.add_argument("--user-id", type=UUID)
    ask.add_argument("--session-id", type=UUID)
    ask.add_argument("--redis-url")
    ask.add_argument("--model")
    ask.add_argument("--provider", choices=("openai", "mimo"))
    upgrade = subparsers.add_parser("db-upgrade", help="Apply all database migrations")
    _database_argument(upgrade)
    return parser


def _root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path.cwd())


def _database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database-url")


def _init(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = ProjectManifestStore(root).load_or_create()
    upgrade_database(_database_url(args.database_url))
    print(json.dumps({"project_id": str(manifest.project_id), "root": str(root)}, indent=2))
    return 0


def _ingest(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    manifest = ProjectManifestStore(root).load_or_create()
    pipeline = build_ingestion_pipeline(
        project_root=root,
        database_url=_database_url(args.database_url),
        parser_name=args.parser,
    )
    report = pipeline.ingest(
        IngestionRequest(
            project_id=manifest.project_id,
            project_root=root,
            paths=tuple(args.paths),
            recursive=not args.no_recursive,
            force_reindex=args.force_reindex,
        )
    )
    payload = {
        "run_id": str(report.run_id),
        "scanned": report.scanned,
        "missing": report.missing,
        "counts": {key.value: value for key, value in report.counts.items()},
        "failed": [
            {
                "path": item.relative_path.as_posix(),
                "error_code": item.error_code.value if item.error_code else None,
                "error": item.error_message,
            }
            for item in report.items
            if item.error_message
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if payload["failed"] else 0


def _status(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    manifest = ProjectManifestStore(root).load()
    payload = {"project_id": str(manifest.project_id), **database_status(_database_url(args.database_url))}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _search(args: argparse.Namespace) -> int:
    manifest = ProjectManifestStore(args.root.resolve()).load()
    result = build_search_knowledge_service(
        database_url=_database_url(args.database_url)
    ).search_knowledge(
        SearchRequest(
            query=args.query,
            scope=SearchScope(project_id=manifest.project_id),
            max_evidence=args.max_evidence,
        )
    )
    payload = {
        "status": result.status.value,
        "reason": result.reason,
        "evidence": [
            {
                "evidence_id": str(item.evidence_id),
                "paper_id": str(item.paper_id),
                "paper_title": item.paper_title,
                "section_id": str(item.section_id),
                "section_path": item.section_path,
                "pages": [item.page_start, item.page_end],
                "chunk_id": str(item.chunk_id),
                "element_ids": [str(value) for value in item.element_ids],
                "text": item.text,
                "relevance": item.relevance,
            }
            for item in result.evidence
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _read(args: argparse.Namespace) -> int:
    manifest = ProjectManifestStore(args.root.resolve()).load()
    pages = (args.pages[0], args.pages[1]) if args.pages else None
    result = build_read_paper_service(
        database_url=_database_url(args.database_url)
    ).read_paper(
        ReadPaperRequest(
            paper_id=args.paper_id,
            project_id=manifest.project_id,
            section_id=args.section_id,
            page_range=pages,
            element_id=args.element_id,
            element_types=tuple(ElementType(value) for value in (args.element_type or [])),
        )
    )
    payload = {
        "paper_id": str(result.paper_id),
        "version_id": str(result.version_id),
        "title": result.title,
        "passages": [
            {
                "citation": f"P{int(item.chunk_id.hex[:12], 16)}",
                "chunk_id": str(item.chunk_id),
                "section_path": item.section_path,
                "pages": [item.page_start, item.page_end],
                "source_group_ids": [str(value) for value in item.source_group_ids],
                "source_block_ids": list(item.source_block_ids),
                "text": item.text,
            }
            for item in result.passages
        ],
        "elements": [
            {
                "citation": f"P{int(item.element_id.hex[:12], 16)}",
                "element_id": str(item.element_id),
                "type": item.element_type.value,
                "section_path": item.section_path,
                "label": item.label,
                "caption": item.caption,
                "content": item.content,
                "page": item.page,
                "source_block_ids": list(item.source_block_ids),
            }
            for item in result.elements
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _ask(args: argparse.Namespace) -> int:
    manifest = ProjectManifestStore(args.root.resolve()).load()
    redis_url = args.redis_url or os.environ.get(
        "PAPER_AGENT_REDIS_URL", "redis://localhost:6379/0"
    )
    model = (
        args.model
        or os.environ.get("PAPER_AGENT_LLM_MODEL")
        or os.environ.get("OPENAI_CHAT_MODEL")
    )
    if not model:
        raise ValueError("Set PAPER_AGENT_LLM_MODEL or pass --model")
    provider = (
        args.provider
        or os.environ.get("PAPER_AGENT_LLM_PROVIDER")
        or ("mimo" if model.lower().startswith("mimo-") else "openai")
    )
    session_id = args.session_id or uuid4()
    user_id = args.user_id or uuid4()
    answer = build_agent_runtime(
        project_id=manifest.project_id,
        database_url=_database_url(args.database_url),
        redis_url=redis_url,
        model=model,
        provider=provider,
    ).run(
        session_id=session_id,
        user_id=user_id,
        project_id=manifest.project_id,
        query=args.query,
    )
    print(
        json.dumps(
            {
                "session_id": str(session_id),
                "user_id": str(user_id),
                "answer": answer.text,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _database_url(argument: str | None) -> str:
    value = argument or os.environ.get("PAPER_AGENT_DATABASE_URL")
    if not value:
        raise ValueError("Set PAPER_AGENT_DATABASE_URL or pass --database-url")
    return value


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"paper-agent: {message}")


if __name__ == "__main__":
    raise SystemExit(main())