"""Mastery preparation must accept an index the provider already calls ready.

GraphRAG, LightRAG and PageIndex publish a synthetic provider signature.
LlamaIndex records the embedding that built the store. Looking that store up
with the process-global embedding signature rejects it before page retrieval
(#2895).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deeptutor.learning import lesson_evidence as lessons
from deeptutor.learning.models import (
    KnowledgePoint,
    LearningModule,
    LearningProgress,
    MasteryTopic,
    TopicMetadata,
    TopicSource,
    TopicSourceKind,
)
from deeptutor.services.rag.index_probe import has_ready_provider_index
from deeptutor.services.rag.index_versioning import (
    EmbeddingSignature,
    resolve_storage_dir_for_read,
)

_GLOBAL = EmbeddingSignature(
    binding="openai",
    model="global-default",
    dimension=1536,
    base_url="https://example.invalid/v1",
    api_version="",
)
_BOUND = EmbeddingSignature(
    binding="openai",
    model="qwen/qwen3-embedding-8b",
    dimension=4096,
    base_url="https://kb.example.invalid/v1",
    api_version="",
)


def _catalog(document_key: str) -> dict:
    return {
        "schema_version": 1,
        "pages": [{"document_key": document_key, "page_index": 0}],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_llamaindex(version: Path, signature: str) -> None:
    version.mkdir(parents=True, exist_ok=True)
    _write_json(version / "docstore.json", {"docstore/data": {"doc-1": {}}})
    (version / "index_store.json").write_text("{}", encoding="utf-8")
    _write_json(
        version / "meta.json",
        {
            "version": version.name,
            "provider": "llamaindex",
            "signature": signature,
            "layout": "flat",
        },
    )


def _prepare_pageindex(version: Path) -> None:
    _write_json(
        version / "meta.json",
        {
            "version": version.name,
            "provider": "pageindex",
            "signature": "pageindex",
            "layout": "flat",
        },
    )
    _write_json(
        version / "pageindex_docs.json",
        {"provider": "pageindex", "docs": {"lesson.pdf": {"doc_id": "doc-123"}}},
    )


def _prepare_pageindex_oss(version: Path) -> None:
    _write_json(
        version / "meta.json",
        {
            "version": version.name,
            "provider": "pageindex-oss",
            "signature": "pageindex-oss",
            "layout": "flat",
        },
    )
    _write_json(
        version / "pageindex_docs.json",
        {"provider": "pageindex-oss", "docs": {"lesson.pdf": {"doc_id": "pi-local"}}},
    )
    doc_dir = version / "pageindex" / "docs" / "pi-local"
    doc_dir.mkdir(parents=True)
    for name in ("doc.json", "tree.json", "pages.json"):
        (doc_dir / name).write_text("{}", encoding="utf-8")


def _prepare_graphrag(version: Path) -> None:
    _write_json(
        version / "meta.json",
        {
            "version": version.name,
            "provider": "graphrag",
            "signature": "graphrag",
            "layout": "flat",
        },
    )
    output = version / "output"
    output.mkdir()
    (output / "entities.parquet").write_bytes(b"placeholder")


def _prepare_lightrag(version: Path) -> None:
    # The stored embedding fingerprint matches the configured model. The version
    # signature is still the synthetic provider name, so a global-embedding
    # lookup cannot see this store.
    _write_json(
        version / "meta.json",
        {
            "version": version.name,
            "provider": "lightrag",
            "signature": "lightrag",
            "layout": "flat",
            "embedding_signature": _GLOBAL.hash(),
            "embedding_model": _GLOBAL.model,
            "embedding_dim": _GLOBAL.dimension,
        },
    )
    _write_json(
        version / "kv_store_doc_status.json",
        {"doc-1": {"status": "processed", "chunks_list": ["chunk-1"]}},
    )


def _lesson(monkeypatch: pytest.MonkeyPatch, provider: str, version: Path):
    base = version.parent.parent
    _write_json(
        base / "kb_config.json",
        {"knowledge_bases": {"selected": {"rag_provider": provider, "status": "ready"}}},
    )
    _write_json(version / "source_pages.json", _catalog("lesson.pdf"))
    monkeypatch.setattr(
        "deeptutor.services.rag.embedding_signature.signature_from_embedding_config",
        lambda: _GLOBAL,
    )
    monkeypatch.setattr(
        "deeptutor.multi_user.knowledge_access.resolve_for_rag",
        lambda kb: SimpleNamespace(base_dir=base, name="selected") if kb == "selected" else None,
    )
    progress = LearningProgress(
        book_id="path",
        modules=[
            LearningModule(
                id="m",
                name="Cells",
                order=0,
                knowledge_points=[
                    KnowledgePoint(
                        id="kp",
                        name="Membrane",
                        type="memory",
                        module_id="m",
                        topic_source_ids=["s"],
                    )
                ],
            )
        ],
    )
    topic = MasteryTopic(
        metadata=TopicMetadata(path_id="path"),
        sources=[
            TopicSource(
                id="s",
                kind=TopicSourceKind.FILE,
                label="Lesson",
                source_id="lesson.pdf",
                metadata={"kb_name": "selected", "path": "lesson.pdf"},
            )
        ],
    )
    return base, progress, topic


@pytest.mark.parametrize(
    ("provider", "prepare"),
    [
        ("pageindex", _prepare_pageindex),
        ("pageindex-oss", _prepare_pageindex_oss),
        ("graphrag", _prepare_graphrag),
        ("lightrag", _prepare_lightrag),
    ],
)
def test_scope_snapshot_accepts_provider_ready_synthetic_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, prepare
) -> None:
    version = tmp_path / "kbs" / "selected" / "version-1"
    version.mkdir(parents=True)
    prepare(version)
    kb_dir = version.parent
    _, progress, topic = _lesson(monkeypatch, provider, version)

    assert has_ready_provider_index(kb_dir, provider) is True
    assert resolve_storage_dir_for_read(kb_dir, _GLOBAL) is None

    _fingerprint, snapshots = lessons.scope_snapshot(topic, progress, "kp")

    assert snapshots["selected"]["version"] == "version-1"
    assert snapshots["selected"]["keys"] == ["lesson.pdf"]


def test_scope_snapshot_accepts_llamaindex_built_with_another_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version = tmp_path / "kbs" / "selected" / "version-1"
    _write_llamaindex(version, _BOUND.hash())
    kb_dir = version.parent
    _, progress, topic = _lesson(monkeypatch, "llamaindex", version)

    assert _BOUND.hash() != _GLOBAL.hash()
    assert has_ready_provider_index(kb_dir, "llamaindex") is True
    assert resolve_storage_dir_for_read(kb_dir, _GLOBAL) is None

    _fingerprint, snapshots = lessons.scope_snapshot(topic, progress, "kp")

    assert snapshots["selected"]["version"] == "version-1"
    assert snapshots["selected"]["keys"] == ["lesson.pdf"]


def test_scope_snapshot_prefers_the_bound_llamaindex_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb_dir = tmp_path / "kbs" / "selected"
    older = kb_dir / "version-1"
    newer = kb_dir / "version-2"
    _write_llamaindex(older, _BOUND.hash())
    _write_llamaindex(newer, "other-embedding")
    _write_json(older / "source_pages.json", _catalog("lesson.pdf"))
    _write_json(newer / "source_pages.json", _catalog("other.pdf"))
    base = kb_dir.parent
    _write_json(
        base / "kb_config.json",
        {
            "knowledge_bases": {
                "selected": {
                    "rag_provider": "llamaindex",
                    "status": "ready",
                    "embedding_selection": {"profile_id": "profile", "model_id": "kb-model"},
                }
            }
        },
    )
    monkeypatch.setattr(
        "deeptutor.services.rag.embedding_signature.signature_from_embedding_config",
        lambda: _GLOBAL,
    )

    def bound_signature(entry: dict) -> EmbeddingSignature:
        assert entry["embedding_selection"]["model_id"] == "kb-model"
        return _BOUND

    monkeypatch.setattr("deeptutor.services.rag.embedding_binding.entry_signature", bound_signature)
    monkeypatch.setattr(
        "deeptutor.multi_user.knowledge_access.resolve_for_rag",
        lambda kb: SimpleNamespace(base_dir=base, name="selected") if kb == "selected" else None,
    )
    progress = LearningProgress(
        book_id="path",
        modules=[
            LearningModule(
                id="m",
                name="Cells",
                order=0,
                knowledge_points=[
                    KnowledgePoint(
                        id="kp",
                        name="Membrane",
                        type="memory",
                        module_id="m",
                        topic_source_ids=["s"],
                    )
                ],
            )
        ],
    )
    topic = MasteryTopic(
        metadata=TopicMetadata(path_id="path"),
        sources=[
            TopicSource(
                id="s",
                kind=TopicSourceKind.FILE,
                label="Lesson",
                source_id="lesson.pdf",
                metadata={"kb_name": "selected", "path": "lesson.pdf"},
            )
        ],
    )

    assert resolve_storage_dir_for_read(kb_dir, _GLOBAL) is None
    assert resolve_storage_dir_for_read(kb_dir, _BOUND) == older

    _fingerprint, snapshots = lessons.scope_snapshot(topic, progress, "kp")

    assert snapshots["selected"]["version"] == "version-1"
    assert snapshots["selected"]["keys"] == ["lesson.pdf"]


def test_scope_snapshot_still_rejects_an_unready_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version = tmp_path / "kbs" / "selected" / "version-1"
    version.mkdir(parents=True)
    _write_json(version / "docstore.json", {"docstore/data": {}})
    _write_json(
        version / "meta.json",
        {
            "version": version.name,
            "provider": "llamaindex",
            "signature": _BOUND.hash(),
            "layout": "flat",
        },
    )
    _write_json(version / "source_pages.json", _catalog("lesson.pdf"))
    kb_dir = version.parent
    _, progress, topic = _lesson(monkeypatch, "llamaindex", version)

    assert has_ready_provider_index(kb_dir, "llamaindex") is False
    with pytest.raises(lessons.LessonUnavailable, match="compatible completed index"):
        lessons.scope_snapshot(topic, progress, "kp")
