"""Settings read from environment variables (see .env.example)."""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    embeddings: str = field(default_factory=lambda: _env("RAG_EMBEDDINGS", "minilm"))
    persist_dir: str | None = field(default_factory=lambda: os.environ.get("RAG_PERSIST_DIR") or None)
    llm_provider: str = field(default_factory=lambda: _env("RAG_LLM_PROVIDER", "fake"))
    llm_model: str = field(default_factory=lambda: _env("RAG_LLM_MODEL", ""))
    chunk_size: int = field(default_factory=lambda: int(_env("RAG_CHUNK_SIZE", "400")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("RAG_CHUNK_OVERLAP", "80")))
    top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "4")))
    min_coverage: float = field(default_factory=lambda: float(_env("RAG_MIN_COVERAGE", "0.5")))
