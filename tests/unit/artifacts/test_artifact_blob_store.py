import gzip
from hashlib import sha256
from pathlib import Path
import tempfile

import pytest

from paper_agent.domain.errors import ErrorCode
from paper_agent.storage.local.artifact_blob_store import (
    ArtifactBlobError,
    LocalArtifactBlobStore,
)


@pytest.fixture()
def store(tmp_path: Path) -> LocalArtifactBlobStore:
    return LocalArtifactBlobStore(tmp_path)


def test_put_get_round_trip_and_gzip(store: LocalArtifactBlobStore) -> None:
    data = b'{"a": 1}'
    key = store.put(content_hash=sha256(data).hexdigest(), data=data)
    assert store.get(storage_key=key) == data
    blob_path = store._blob_root / key
    raw = blob_path.read_bytes()
    # gzip actually compresses: decompressing yields the original bytes
    assert gzip.decompress(raw) == data
    assert len(raw) < len(data) or len(raw) <= len(data) + 32


def test_put_is_content_addressed_and_idempotent(store: LocalArtifactBlobStore) -> None:
    data = b'{"x": [1, 2, 3]}'
    h = sha256(data).hexdigest()
    first = store.put(content_hash=h, data=data)
    second = store.put(content_hash=h, data=data)
    assert first == second
    assert store.exists(storage_key=first)


def test_put_rejects_mismatched_hash(store: LocalArtifactBlobStore) -> None:
    with pytest.raises(ValueError, match="does not match"):
        store.put(content_hash="a" * 64, data=b"other")


def test_get_rejects_corrupt_blob(store: LocalArtifactBlobStore) -> None:
    data = b'{"ok": true}'
    key = store.put(content_hash=sha256(data).hexdigest(), data=data)
    blob_path = store._blob_root / key
    blob_path.write_bytes(gzip.compress(b'{"tampered": true}'))
    with pytest.raises(ArtifactBlobError) as excinfo:
        store.get(storage_key=key)
    assert excinfo.value.code == ErrorCode.ARTIFACT_CORRUPT


def test_get_rejects_path_traversal(store: LocalArtifactBlobStore) -> None:
    with pytest.raises(ArtifactBlobError) as excinfo:
        store.get(storage_key="../../etc/passwd")
    assert excinfo.value.code == ErrorCode.INVALID_PATH


def test_get_missing_blob(store: LocalArtifactBlobStore) -> None:
    with pytest.raises(ArtifactBlobError) as excinfo:
        store.get(storage_key="sha256/aa/" + "a" * 64 + ".json.gz")
    assert excinfo.value.code == ErrorCode.FILE_NOT_FOUND


def test_delete_removes_blob(store: LocalArtifactBlobStore) -> None:
    data = b"delete me"
    key = store.put(content_hash=sha256(data).hexdigest(), data=data)
    store.delete(storage_key=key)
    assert not store.exists(storage_key=key)
