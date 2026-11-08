"""Tiny shared tokenizer used by BM25, the hashing embedder, refusal and eval checks."""

import re

STOPWORDS = frozenset(
    "a an the of to in on at by for from with and or is are was were be been do does did can could "
    "should would will may i you we they it its my our your their what which who whom when where why "
    "how many much long often this that these those there as about into than then so if not no".split()
)

_TOKEN = re.compile(r"[a-z0-9@.\-]+")


def tokens(text: str) -> list[str]:
    return [t.strip(".-") for t in _TOKEN.findall(text.lower()) if t.strip(".-")]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]
