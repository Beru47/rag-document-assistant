"""FastAPI service: /ingest, /ask, /health."""

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

from .config import Settings
from .ingest import ingest_files
from .llm import get_llm
from .pipeline import build_index
from .qa import ask


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=4, ge=1, le=20)


def create_app(index=None, llm=None, settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()
    app = FastAPI(title="RAG Document Assistant")
    state: dict = {"index": index, "llm": llm}

    def get_state():  # lazy: avoids loading models at import time
        if state["index"] is None:
            state["index"] = build_index(s)
        if state["llm"] is None:
            state["llm"] = get_llm(s.llm_provider, s.llm_model)
        return state["index"], state["llm"]

    @app.get("/health")
    def health():
        idx = state["index"]
        return {"status": "ok", "chunks": len(idx) if idx is not None else 0}

    @app.post("/ingest")
    async def ingest(files: list[UploadFile] = File(...)):
        idx, _ = get_state()
        payload = [(f.filename or "upload", await f.read()) for f in files]
        docs, skipped = ingest_files(payload, s.chunk_size, s.chunk_overlap)
        return {"chunks_added": idx.add(docs), "skipped": skipped}

    @app.post("/ask")
    def ask_endpoint(req: AskRequest):
        idx, model = get_state()
        res = ask(idx, model, req.question, req.k, s.min_coverage)
        return {"answer": res.answer, "refused": res.refused, "sources": res.sources}

    return app


app = create_app()
