"""LLM providers (OpenAI-compatible, Anthropic) plus a deterministic extractive stand-in for tests/CI."""

import os
import re

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from .text import content_tokens

REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"
_CTX = re.compile(r"^\[(\d+)\] \(source: [^)]*\) (.*)$", re.M)
_QUESTION = re.compile(r"^Question: (.*)$", re.M)
_SENT = re.compile(r"(?<=[.!?])\s+")


def extractive_llm(min_overlap: float = 0.5) -> Runnable:
    """Answers with the context sentence that best overlaps the question, cited as [n].

    It is not a language model: it makes tests and the eval harness runnable without API keys.
    """

    def run(prompt_value) -> AIMessage:
        text = prompt_value.to_messages()[-1].content
        q = set(content_tokens(_QUESTION.search(text).group(1)))
        best, best_score = None, 0.0
        for n, body in _CTX.findall(text):
            for sent in _SENT.split(body):
                score = len(q & set(content_tokens(sent))) / (len(q) or 1)
                if score > best_score:
                    best, best_score = f"{sent.strip()} [{n}]", score
        return AIMessage(content=best if best_score >= min_overlap else REFUSAL_TOKEN)

    return RunnableLambda(run)


def get_llm(provider: str = "fake", model: str = "") -> Runnable:
    if provider == "fake":
        return extractive_llm()
    if provider == "openai":  # also any OpenAI-compatible server via OPENAI_BASE_URL
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "gpt-4o-mini", temperature=0, base_url=os.environ.get("OPENAI_BASE_URL")
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model or os.environ["RAG_LLM_MODEL"], temperature=0)
    raise ValueError(f"unknown LLM provider: {provider!r} (fake | openai | anthropic)")
