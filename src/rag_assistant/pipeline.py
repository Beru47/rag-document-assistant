"""Wiring helpers shared by the API and the eval harness."""

from pathlib import Path

from .config import Settings
from .embeddings import get_embeddings
from .ingest import ingest_files
from .store import HybridIndex


def build_index(settings: Settings | None = None) -> HybridIndex:
    s = settings or Settings()
    return HybridIndex(get_embeddings(s.embeddings), s.persist_dir)


def ingest_paths(index: HybridIndex, paths: list[Path], s: Settings | None = None):
    s = s or Settings()
    docs, skipped = ingest_files([(p.name, p.read_bytes()) for p in paths], s.chunk_size, s.chunk_overlap)
    return index.add(docs), skipped
