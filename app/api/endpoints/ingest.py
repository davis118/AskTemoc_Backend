"""
/api/ingest — receives scraper payloads, chunks text, embeds via OpenAI,
and stores Documents + Chunks + Embeddings in Postgres (pgvector).

Chunking stance (production RAG for chatbots, 2026):
- Recursive / structure-aware splitting is the usual default (paragraph → line → sentence)
  rather than blindly fixed characters; overlapping windows (~10–20%) preserves boundary context.
- Semantic or hierarchical chunking can win on messy multi-topic docs; parent-document patterns
  (small chunks indexed, larger parent fed to the LLM) are common once you measure retrieval misses.
- For well-scoped artifacts (single degree catalog page), a single embedding over the whole page
  often beats brittle splits—that is what we do for UT Dallas catalog program-detail URLs below.
"""

import uuid
import logging
from typing import Optional
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from openai import OpenAI

from app.db.database import get_db
from app.db.models import Document, Chunk, Embedding
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# text-embedding-3-large token limit is 8191 — use ~2000 tokens per chunk
# for good retrieval granularity while preserving meaningful context.
CHUNK_TOKENS = 2000
CHUNK_OVERLAP_TOKENS = 100
TIKTOKEN_MODEL = "text-embedding-3-large"

_EMBED_SAFE_MAX_TOKENS = 8000  # below OpenAI 8192 cap for text-embedding-3-large inputs


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


def _is_utd_catalog_degree_detail_page(url: str) -> bool:
    """
    True for a single-program catalog URL (degree page), not the program listing hub.

    Examples (detail → True):
      .../catalog.utdallas.edu/.../programs/aht/animation-and-games
    Examples (listing hub → False):
      .../undergraduate/programs
    """
    try:
        p = urlparse(url)
        if "catalog.utdallas.edu" not in (p.netloc or "").lower():
            return False
        parts = [x for x in (p.path or "").strip("/").split("/") if x]
        if "programs" not in parts:
            return False
        idx = parts.index("programs")
        # At least school code + degree slug after "programs", e.g. aht/ba-xyz
        return len(parts) >= idx + 3
    except Exception:
        return False


def _catalog_degree_embed_text(page_title: str, body: str) -> str:
    """Canonical text used for embedding: degree name first (retrieval aligns to title)."""
    return f"Degree program: {page_title.strip()}\n\n{(body or '').strip()}".strip()


def _approx_tokens(text: str) -> int:
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    return len(enc.encode(text))


def _split_text(text: str) -> list[str]:
    """
    Token-aware recursive splitter using tiktoken.
    Splits on paragraph → sentence → word boundaries before resorting to chars.
    Targets CHUNK_TOKENS tokens per chunk with CHUNK_OVERLAP_TOKENS overlap.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")  # same tokeniser family

    def token_len(t: str) -> int:
        return len(enc.encode(t))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )

    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


def _embed_texts(texts: list[str], settings) -> list[list[float]]:
    """Call OpenAI embeddings API and return vectors."""
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.embeddings.create(
        input=texts,
        model=settings.OPENAI_EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


def _upsert_document(
    db: Session, url: str, title: str, doc_metadata: Optional[dict] = None
) -> Document:
    """Return existing document for this URL (deleting old chunks) or create new."""
    doc = db.query(Document).filter(Document.source == url, Document.is_deleted == False).first()

    if doc:
        for chunk in db.query(Chunk).filter(Chunk.document_id == doc.id).all():
            db.delete(chunk)
        doc.title = title
        doc.updated_at = datetime.now(timezone.utc)
        if doc_metadata is not None:
            merged = dict(doc.doc_metadata or {})
            merged.update(doc_metadata)
            doc.doc_metadata = merged
        db.commit()
        return doc

    doc = Document(
        id=str(uuid.uuid4()),
        title=title,
        source=url,
        doc_metadata=doc_metadata or {},
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

    catalog_degree_page = _is_utd_catalog_degree_detail_page(payload.url)

    doc_metadata = None
    if catalog_degree_page:
        doc_metadata = {
            "catalog_degree_page": True,
            "degree_title": payload.page_title.strip(),
        }

    # Split into chunks
    if catalog_degree_page:
        # Exactly one retrieval unit per degree: never segment program pages.
        if payload.chunks:
            merged = "\n\n".join(
                str(c).strip() for c in payload.chunks if str(c).strip()
            ).strip()
        else:
            merged = (raw_text or "").strip()
        canon = _catalog_degree_embed_text(payload.page_title, merged)
        ntok = _approx_tokens(canon)
        if ntok > _EMBED_SAFE_MAX_TOKENS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Catalog degree page exceeds {_EMBED_SAFE_MAX_TOKENS} tokens ({ntok}): "
                    "cannot embed without splitting; trim source or raise model limit upstream."
                ),
            )
        texts = [canon]
    elif payload.chunks:
        texts = [c if isinstance(c, str) else str(c) for c in payload.chunks]
    else:
        texts = _split_text(raw_text)

    if not texts:
        raise HTTPException(status_code=422, detail="Text produced zero chunks after splitting")

    # Upsert document
    doc = _upsert_document(db, payload.url, payload.page_title, doc_metadata=doc_metadata)

    # Embed all chunks in one API call
    try:
        vectors = _embed_texts(texts, settings)
    except Exception as e:
        logger.error(f"OpenAI embedding failed for {payload.url}: {e}")
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}")

    # Persist chunks + embeddings
    chunk_meta_common = {}
    if catalog_degree_page:
        chunk_meta_common = {
            "catalog_degree_page": True,
            "degree_title": payload.page_title.strip(),
            "embedding_unit": "full_degree_page",
        }

    model_name = settings.OPENAI_EMBEDDING_MODEL
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        chunk = Chunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=i,
            text=text,
            chunk_metadata=dict(chunk_meta_common) if chunk_meta_common else {},
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

    logger.info(f"Ingested {len(texts)} chunks for {payload.url} (catalog_degree={catalog_degree_page})")
    return {
        "status": "ok",
        "url": payload.url,
        "chunks_ingested": len(texts),
        "document_id": doc.id,
        "catalog_degree_page": catalog_degree_page,
    }
