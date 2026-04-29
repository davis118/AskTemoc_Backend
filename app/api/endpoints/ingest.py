"""
/api/ingest — receives scraper payloads, chunks text, embeds via OpenAI,
and stores Documents + Chunks + Embeddings in Postgres (pgvector).
"""

import uuid
import logging
from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from openai import OpenAI

from app.db.database import get_db
from app.db.models import Document, Chunk, Embedding
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Chunk sizes tuned for text-embedding-3-large (8191 token limit)
CHUNK_SIZE = 1000   # characters
CHUNK_OVERLAP = 150


# --------------------------------------------------------------------------- #
# Pydantic schema (matches publisher.py build_payload output)
# --------------------------------------------------------------------------- #

class IngestPayload(BaseModel):
    url: str
    page_title: str
    timestamp: Optional[str] = None
    text_content: Optional[str] = None
    html_content: Optional[str] = None
    chunks: Optional[list] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple character-based splitter with overlap, breaking at sentence ends."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        segment = text[start:end]

        if end < len(text):
            for sep in [". ", ".\n", "! ", "!\n", "? ", "?\n", "\n\n"]:
                boundary = segment.rfind(sep)
                if boundary > chunk_size // 2:
                    segment = segment[: boundary + 1]
                    end = start + len(segment)
                    break

        if segment.strip():
            chunks.append(segment.strip())
        start = end - overlap

    return chunks


def _embed_texts(texts: list[str], settings) -> list[list[float]]:
    """Call OpenAI embeddings API and return vectors."""
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.embeddings.create(
        input=texts,
        model=settings.OPENAI_EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


def _upsert_document(db: Session, url: str, title: str) -> Document:
    """Return existing document for this URL (deleting old chunks) or create new."""
    doc = db.query(Document).filter(Document.source == url, Document.is_deleted == False).first()

    if doc:
        # Delete old chunks cascade (embeddings are cascade-deleted via FK)
        for chunk in db.query(Chunk).filter(Chunk.document_id == doc.id).all():
            db.delete(chunk)
        doc.title = title
        doc.updated_at = datetime.now(timezone.utc)
        db.commit()
        return doc

    doc = Document(
        id=str(uuid.uuid4()),
        title=title,
        source=url,
        doc_metadata={},
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #

@router.post("/ingest", status_code=status.HTTP_200_OK)
def ingest(payload: IngestPayload, db: Session = Depends(get_db)):
    settings = get_settings()

    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    # Resolve raw text — prefer text_content, fall back to html_content stripped
    raw_text: Optional[str] = payload.text_content
    if not raw_text and payload.html_content:
        from html.parser import HTMLParser
        class _Strip(HTMLParser):
            def __init__(self):
                super().__init__()
                self._parts: list[str] = []
            def handle_data(self, data):
                if data.strip():
                    self._parts.append(data.strip())
            def text(self):
                return " ".join(self._parts)
        p = _Strip()
        p.feed(payload.html_content)
        raw_text = p.text()

    if not raw_text and not payload.chunks:
        raise HTTPException(status_code=422, detail="No text content or chunks provided")

    # Split into chunks
    if payload.chunks:
        texts = [c if isinstance(c, str) else str(c) for c in payload.chunks]
    else:
        texts = _split_text(raw_text)

    if not texts:
        raise HTTPException(status_code=422, detail="Text produced zero chunks after splitting")

    # Upsert document
    doc = _upsert_document(db, payload.url, payload.page_title)

    # Embed all chunks in one API call
    try:
        vectors = _embed_texts(texts, settings)
    except Exception as e:
        logger.error(f"OpenAI embedding failed for {payload.url}: {e}")
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}")

    # Persist chunks + embeddings
    model_name = settings.OPENAI_EMBEDDING_MODEL
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        chunk = Chunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=i,
            text=text,
            chunk_metadata={},
        )
        db.add(chunk)
        db.flush()  # get chunk.id before embedding insert

        embedding = Embedding(
            id=str(uuid.uuid4()),
            chunk_id=chunk.id,
            vector=vector,
            model=model_name,
            is_synced=True,  # pgvector IS the sync target
        )
        db.add(embedding)

    db.commit()

    logger.info(f"Ingested {len(texts)} chunks for {payload.url}")
    return {
        "status": "ok",
        "url": payload.url,
        "chunks_ingested": len(texts),
        "document_id": doc.id,
    }
