"""Load PDF / text / image files, clean, and split into overlapping chunks."""

import io
import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)

TEXT_EXT = {".txt", ".md"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class OCRUnavailable(RuntimeError):
    """pytesseract or the tesseract binary is missing."""


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated at line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def ocr_image(data: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise OCRUnavailable("install the 'ocr' extra: pip install -e '.[ocr]'") from e
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(data)))
    except pytesseract.TesseractNotFoundError as e:
        raise OCRUnavailable("tesseract binary not found on PATH") from e


def load_pages(name: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return (page_number, raw_text) pairs. Raises OCRUnavailable if OCR is needed but missing."""
    ext = Path(name).suffix.lower()
    if ext in TEXT_EXT:
        return [(None, data.decode("utf-8", errors="replace"))]
    if ext in IMAGE_EXT:
        return [(None, ocr_image(data))]
    if ext == ".pdf":
        from pypdf import PdfReader

        pages = []
        for i, page in enumerate(PdfReader(io.BytesIO(data)).pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():  # scanned page: OCR its embedded images
                text = "\n".join(ocr_image(img.data) for img in page.images)
            pages.append((i, text))
        return pages
    raise ValueError(f"unsupported file type: {ext or name}")


def chunk_document(name: str, data: bytes, chunk_size: int = 400, chunk_overlap: int = 80) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: list[Document] = []
    for page, raw in load_pages(name, data):
        text = clean_text(raw)
        if not text:
            continue
        for piece in splitter.split_text(text):
            meta = {"source": name, "chunk": len(docs)}
            if page is not None:
                meta["page"] = page
            docs.append(Document(page_content=piece, metadata=meta))
    return docs


def ingest_files(
    files: list[tuple[str, bytes]], chunk_size: int = 400, chunk_overlap: int = 80
) -> tuple[list[Document], dict[str, str]]:
    """Chunk many files; files that cannot be read are reported in `skipped` instead of aborting."""
    docs: list[Document] = []
    skipped: dict[str, str] = {}
    for name, data in files:
        try:
            got = chunk_document(name, data, chunk_size, chunk_overlap)
        except (OCRUnavailable, ValueError) as e:
            log.warning("skipping %s: %s", name, e)
            skipped[name] = str(e)
            continue
        if not got:
            skipped[name] = "no extractable text"
        docs.extend(got)
    return docs, skipped
