"""Project-scoped, deterministic PDF discovery."""

from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from paper_agent.domain.errors import ErrorCode
from paper_agent.domain.ingestion import DiscoveredFile, ScanIssue, ScanResult


class DirectoryScanner:
    def __init__(self, *, follow_symlinks: bool = False, ignored_directories: tuple[str, ...] = (".paper-agent",)):
        self._follow_symlinks = follow_symlinks
        self._ignored_directories = frozenset(ignored_directories)

    def scan(self, project_root: Path, paths: tuple[Path, ...] = (), *, recursive: bool = True) -> ScanResult:
        root = project_root.resolve()
        targets = paths or (root,)
        discovered: dict[PurePosixPath, DiscoveredFile] = {}
        issues: list[ScanIssue] = []

        for target_input in targets:
            target = target_input if target_input.is_absolute() else root / target_input
            try:
                target = target.resolve(strict=True)
            except FileNotFoundError:
                issues.append(
                    ScanIssue(target, ErrorCode.FILE_NOT_FOUND, f"Scan target does not exist: {target}")
                )
                continue
            except OSError as error:
                issues.append(ScanIssue(target, ErrorCode.INVALID_PATH, str(error)))
                continue

            if not target.is_relative_to(root):
                issues.append(
                    ScanIssue(
                        target,
                        ErrorCode.PATH_OUTSIDE_PROJECT,
                        f"Scan target is outside project root: {target}",
                    )
                )
                continue

            for candidate in self._iter_candidates(target, recursive=recursive, issues=issues):
                relative = PurePosixPath(candidate.relative_to(root).as_posix())
                if any(part in self._ignored_directories for part in relative.parts):
                    continue
                try:
                    stat = candidate.stat(follow_symlinks=self._follow_symlinks)
                except OSError as error:
                    issues.append(ScanIssue(candidate, ErrorCode.INVALID_PATH, str(error)))
                    continue
                discovered[relative] = DiscoveredFile(
                    absolute_path=candidate,
                    relative_path=relative,
                    file_size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )

        ordered_files = tuple(
            discovered[path] for path in sorted(discovered, key=lambda path: str(path).casefold())
        )
        return ScanResult(files=ordered_files, issues=tuple(issues))

    def _iter_candidates(
        self,
        target: Path,
        *,
        recursive: bool,
        issues: list[ScanIssue],
    ) -> Iterable[Path]:
        if target.is_symlink() and not self._follow_symlinks:
            return
        if target.is_file():
            if target.suffix.casefold() == ".pdf":
                yield target
            return
        if not target.is_dir():
            issues.append(ScanIssue(target, ErrorCode.INVALID_PATH, f"Unsupported scan target: {target}"))
            return

        try:
            entries = sorted(target.iterdir(), key=lambda path: path.name.casefold())
        except OSError as error:
            issues.append(ScanIssue(target, ErrorCode.INVALID_PATH, str(error)))
            return

        for entry in entries:
            if entry.name in self._ignored_directories:
                continue
            if entry.is_symlink() and not self._follow_symlinks:
                continue
            if entry.is_file() and entry.suffix.casefold() == ".pdf":
                yield entry
            elif recursive and entry.is_dir():
                yield from self._iter_candidates(entry, recursive=True, issues=issues)
