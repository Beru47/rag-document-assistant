"""Retrieval-augmented answering with citations and refusal."""

import re
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .llm import REFUSAL_TOKEN
from .store import HybridIndex
from .text import content_tokens

REFUSAL_MESSAGE = "I could not find enough information in the indexed documents to answer that."

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using ONLY the numbered context passages. Cite every claim with its passage "
            "number in square brackets, e.g. [1]. If the passages do not contain the answer, "
            f"reply exactly {REFUSAL_TOKEN}.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    answer: str
    refused: bool
    sources: list[dict] = field(default_factory=list)
    retrieved: list = field(default_factory=list)


def coverage(question: str, text: str) -> float:
    """Fraction of the question's content words that appear in `text`."""
    q = set(content_tokens(question))
    return len(q & set(content_tokens(text))) / len(q) if q else 0.0


def format_context(docs) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata['source']}) {' '.join(d.page_content.split())}"
        for i, d in enumerate(docs, start=1)
    )


def ask(index: HybridIndex, llm: Runnable, question: str, k: int = 4, min_coverage: float = 0.5) -> Answer:
    docs = index.search(question, k=k)
    # Gate 1: no retrieved passage mentions enough of the question's terms -> do not call the LLM.
    if not docs or max(coverage(question, d.page_content) for d in docs) < min_coverage:
        return Answer(REFUSAL_MESSAGE, True, retrieved=docs)
    text = (
        (PROMPT | llm | StrOutputParser())
        .invoke({"question": question, "context": format_context(docs)})
        .strip()
    )
    # Gate 2: the model itself says the context is insufficient.
    if REFUSAL_TOKEN in text:
        return Answer(REFUSAL_MESSAGE, True, retrieved=docs)
    cited = sorted({int(n) for n in _CITE.findall(text) if 0 < int(n) <= len(docs)})
    sources = [
        {
            "ref": n,
            "source": docs[n - 1].metadata["source"],
            "page": docs[n - 1].metadata.get("page"),
            "text": docs[n - 1].page_content,
        }
        for n in cited
    ]
    return Answer(text, False, sources, docs)
