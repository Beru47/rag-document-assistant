"""Chroma vector store + BM25 index with reciprocal-rank-fusion (hybrid retrieval)."""

import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from .text import tokens

RRF_K = 60  # standard constant from the original RRF paper


class HybridIndex:
    def __init__(self, embeddings: Embeddings, persist_dir: str | None = None, name: str = "docs"):
        # unique name for in-memory stores so several indexes can coexist in one process
        collection = name if persist_dir else f"{name}-{uuid.uuid4().hex[:8]}"
        self._vs = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._docs: dict[str, Document] = {}
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._dirty = True
        got = self._vs.get()
        for i, text, meta in zip(got["ids"], got["documents"], got["metadatas"], strict=False):
            self._docs[i] = Document(page_content=text, metadata={**meta, "id": i})

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        ids = [uuid.uuid4().hex[:12] for _ in docs]
        for i, d in zip(ids, docs, strict=False):
            d.metadata["id"] = i
            self._docs[i] = d
        self._vs.add_documents(docs, ids=ids)
        self._dirty = True
        return len(docs)

    def _bm25_index(self) -> BM25Okapi | None:
        if self._dirty:  # ponytail: full rebuild per ingest, incremental index if corpora get large
            self._ids = list(self._docs)
            corpus = [tokens(self._docs[i].page_content) for i in self._ids]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._dirty = False
        return self._bm25

    def vector_search(self, query: str, k: int) -> list[Document]:
        if not self._docs:
            return []
        return [self._docs[d.metadata["id"]] for d in self._vs.similarity_search(query, k=k)]

    def bm25_search(self, query: str, k: int) -> list[Document]:
        bm25 = self._bm25_index()
        if bm25 is None:
            return []
        scores = bm25.get_scores(tokens(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self._docs[self._ids[i]] for i in order if scores[i] > 0]

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[Document]:
        if mode == "vector":
            return self.vector_search(query, k)
        if mode == "bm25":
            return self.bm25_search(query, k)
        pool = max(k * 3, 10)
        fused: dict[str, float] = {}
        for ranking in (self.vector_search(query, pool), self.bm25_search(query, pool)):
            for rank, d in enumerate(ranking):
                fused[d.metadata["id"]] = fused.get(d.metadata["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self._docs[i] for i in best]
