"""Embedding backends behind LangChain's `Embeddings` interface."""

import hashlib
import math

from langchain_core.embeddings import Embeddings

from .text import content_tokens


class HashingEmbeddings(Embeddings):
    """Deterministic signed feature hashing of unigrams + bigrams. No model download: for tests/CI."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        toks = content_tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        vec = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(name: str = "minilm") -> Embeddings:
    if name == "hashing":
        return HashingEmbeddings()
    if name == "minilm":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install the 'minilm' extra: pip install -e '.[minilm]'") from e
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"unknown embeddings backend: {name!r} (use 'minilm' or 'hashing')")
