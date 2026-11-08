"""Evaluation harness: retrieval hit@k / MRR, citation-overlap faithfulness, refusal behaviour.

Run: python -m rag_assistant.eval [--embeddings hashing|minilm] [--llm fake|openai|anthropic]
"""

import argparse
import json
import re
from pathlib import Path

from .config import Settings
from .embeddings import get_embeddings
from .ingest import ingest_files
from .llm import get_llm
from .qa import ask
from .store import HybridIndex
from .text import content_tokens

DATA = Path(__file__).resolve().parents[2] / "data"
KS = (1, 3, 5)
MIN_FAITHFUL_OVERLAP = 0.6


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def is_gold(doc, item: dict) -> bool:
    """A retrieved chunk is relevant if it comes from the gold file and contains the evidence phrase."""
    return doc.metadata["source"] == item["source"] and _norm(item["evidence"]) in _norm(doc.page_content)


def faithful(answer: str, sources: list[dict]) -> bool:
    """Citation-overlap check: >=1 valid citation and most answer words appear in the cited chunks."""
    if not sources:
        return False
    claim = content_tokens(re.sub(r"\[\d+\]", " ", answer))
    cited = set(content_tokens(" ".join(s["text"] for s in sources)))
    return bool(claim) and sum(t in cited for t in claim) / len(claim) >= MIN_FAITHFUL_OVERLAP


def retrieval_metrics(index: HybridIndex, qa: list[dict], mode: str) -> dict:
    hits = {k: 0 for k in KS}
    rr = 0.0
    for item in qa:
        docs = index.search(item["question"], k=max(KS), mode=mode)
        rank = next((i for i, d in enumerate(docs, 1) if is_gold(d, item)), None)
        if rank:
            rr += 1 / rank
            for k in KS:
                hits[k] += rank <= k
    n = len(qa)
    return {**{f"hit@{k}": hits[k] / n for k in KS}, "mrr": rr / n}


def answer_metrics(index: HybridIndex, llm, qa: list[dict], k: int, min_coverage: float) -> dict:
    ans = [(i, ask(index, llm, i["question"], k, min_coverage)) for i in qa]
    answerable = [(i, a) for i, a in ans if i["source"]]
    unanswerable = [(i, a) for i, a in ans if not i["source"]]
    given = [(i, a) for i, a in answerable if not a.refused]
    return {
        "answered": f"{len(given)}/{len(answerable)}",
        "faithfulness": sum(faithful(a.answer, a.sources) for _, a in given) / len(given) if given else 0.0,
        "correct_refusals": f"{sum(a.refused for _, a in unanswerable)}/{len(unanswerable)}",
    }


def run(embeddings: str, provider: str, model: str = "") -> str:
    s = Settings()
    files = [(p.name, p.read_bytes()) for p in sorted((DATA / "corpus").glob("*.md"))]
    docs, _ = ingest_files(files, s.chunk_size, s.chunk_overlap)
    index = HybridIndex(get_embeddings(embeddings))
    index.add(docs)
    qa = json.loads((DATA / "qa.json").read_text(encoding="utf-8"))
    answerable = [q for q in qa if q["source"]]

    lines = [
        f"corpus: {len(files)} docs, {len(docs)} chunks | questions: {len(answerable)} answerable + "
        f"{len(qa) - len(answerable)} unanswerable | embeddings={embeddings} llm={provider}",
        "",
        "retrieval (answerable questions)",
        f"{'mode':<8} " + " ".join(f"hit@{k:<3}" for k in KS) + " mrr",
    ]
    for mode in ("vector", "bm25", "hybrid"):
        m = retrieval_metrics(index, answerable, mode)
        lines.append(f"{mode:<8} " + " ".join(f"{m[f'hit@{k}']:<6.3f}" for k in KS) + f" {m['mrr']:.3f}")
    a = answer_metrics(index, get_llm(provider, model), qa, s.top_k, s.min_coverage)
    lines += [
        "",
        f"generation (hybrid, k={s.top_k})",
        f"answered answerable questions : {a['answered']}",
        f"faithfulness (citation overlap): {a['faithfulness']:.3f}",
        f"correct refusals (unanswerable): {a['correct_refusals']}",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", default="hashing", choices=["hashing", "minilm"])
    p.add_argument("--llm", default="fake", choices=["fake", "openai", "anthropic"])
    p.add_argument("--model", default="")
    args = p.parse_args()
    print(run(args.embeddings, args.llm, args.model))


if __name__ == "__main__":
    main()
