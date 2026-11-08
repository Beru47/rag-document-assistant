import io

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from rag_assistant.api import create_app
from rag_assistant.embeddings import HashingEmbeddings
from rag_assistant.ingest import OCRUnavailable, chunk_document, clean_text, ingest_files, ocr_image
from rag_assistant.llm import REFUSAL_TOKEN, extractive_llm
from rag_assistant.qa import ask
from rag_assistant.store import HybridIndex

DOCS = {
    "vacation.txt": b"Employees receive 25 days of paid annual leave. Unused days expire on March 31.",
    "api.txt": b"The Routing API allows 600 requests per minute per key. Keys rotate every 90 days.",
    "backup.txt": b"Backups are encrypted with AES-256 and stored in two regions.",
}


@pytest.fixture
def index():
    idx = HybridIndex(HashingEmbeddings())
    docs, _ = ingest_files(list(DOCS.items()))
    idx.add(docs)
    return idx


# --- chunking / ingestion -------------------------------------------------------------------


def test_chunking_overlaps_and_keeps_metadata():
    text = " ".join(f"word{i}" for i in range(300)).encode()
    chunks = chunk_document("long.txt", text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 3
    assert all(len(c.page_content) <= 200 and c.metadata["source"] == "long.txt" for c in chunks)
    tail = chunks[0].page_content.split()[-2:]
    assert all(w in chunks[1].page_content for w in tail)  # overlap


def test_clean_text():
    assert clean_text("infor-\nmation  here\r\n\n\n\nx\x00") == "information here\n\nx"


def test_pdf_text_extraction():
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 700, "Hello PDF world")
    c.showPage()
    c.save()
    chunks = chunk_document("a.pdf", buf.getvalue())
    assert "Hello PDF world" in chunks[0].page_content and chunks[0].metadata["page"] == 1


def test_missing_ocr_is_skipped_not_fatal(monkeypatch):
    def boom(_):
        raise OCRUnavailable("no tesseract")

    monkeypatch.setattr("rag_assistant.ingest.ocr_image", boom)
    docs, skipped = ingest_files([("scan.png", b"x"), ("ok.txt", b"fine text here")])
    assert len(docs) == 1 and "scan.png" in skipped


def test_ocr_image_raises_unavailable_without_tesseract():
    import shutil

    if shutil.which("tesseract"):
        pytest.skip("tesseract installed")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, "PNG")
    with pytest.raises(OCRUnavailable):
        ocr_image(buf.getvalue())


# --- retrieval ------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_retrieval_finds_right_doc(index, mode):
    top = index.search("how many requests per minute does the API allow", k=1, mode=mode)[0]
    assert top.metadata["source"] == "api.txt"


def test_hybrid_matches_exact_rare_token(index):
    assert index.search("AES-256", k=1)[0].metadata["source"] == "backup.txt"


# --- answering ------------------------------------------------------------------------------


def test_answer_has_citation_and_sources(index):
    res = ask(index, extractive_llm(), "How many days of annual leave do employees receive?")
    assert not res.refused and "[" in res.answer
    assert {s["source"] for s in res.sources} == {"vacation.txt"}


def test_refuses_when_context_missing(index):
    res = ask(index, extractive_llm(), "Who won the football world cup in 1998?")
    assert res.refused and res.sources == []


def test_refuses_when_llm_declines(index):
    llm = RunnableLambda(lambda _: REFUSAL_TOKEN)
    assert ask(index, llm, "How many days of annual leave do employees receive?").refused


# --- API ------------------------------------------------------------------------------------


def test_api_end_to_end(index):
    client = TestClient(create_app(index=HybridIndex(HashingEmbeddings()), llm=extractive_llm()))
    assert client.get("/health").json() == {"status": "ok", "chunks": 0}

    files = [("files", (n, b, "text/plain")) for n, b in DOCS.items()]
    files.append(("files", ("scan.xyz", b"?", "application/octet-stream")))
    r = client.post("/ingest", files=files).json()
    assert r["chunks_added"] == 3 and "scan.xyz" in r["skipped"]

    r = client.post("/ask", json={"question": "How often do the API keys rotate?"}).json()
    assert not r["refused"] and r["sources"][0]["source"] == "api.txt"

    r = client.post("/ask", json={"question": "What is the CEO's favorite color?"}).json()
    assert r["refused"] and r["sources"] == []


def test_api_validates_input():
    client = TestClient(create_app(index=HybridIndex(HashingEmbeddings()), llm=extractive_llm()))
    assert client.post("/ask", json={"question": "x"}).status_code == 422
