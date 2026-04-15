import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.db.services import DocumentService, ChunkService, EmbeddingService
from app.services.document_splitter import DocumentSplitter, EmbedBatch
from app.services.retriever_service import retriever_service
from app.services.ingest_service import IngestService
from langchain_chroma import Chroma

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

splitter = DocumentSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

vector_store = Chroma(
    client=retriever_service.client,
    collection_name=retriever_service.collection_name,
    embedding_function=retriever_service.embeddings,
)
def get_ingest_service():
    ingest_service = IngestService(
        splitter=splitter,
        embedding_function=retriever_service.embeddings.embed_query,
        vector_store=vector_store
    )
    return ingest_service

router = APIRouter()

@router.get("/")
async def list_documents(skip: int = 0,limit: int = 100,include_deleted: bool = False, db: Session = Depends(get_db)):
    """List all documents."""
    docs = DocumentService.list_documents(db, skip, limit, include_deleted)
    return {"documents": docs}

@router.get("/{doc_id}")
async def get_document(doc_id: str, db: Session = Depends(get_db)):
    """Get detailed document information including its chunks."""
    doc = DocumentService.get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = ChunkService.list_chunks_by_document(db, doc_id)
    return {"document": doc, "chunks": chunks}

@router.post("/upload/file")
async def upload_file(file: UploadFile = File(...),title: Optional[str] = Form(None), metadata: Optional[str] = Form(None), ingest_service: IngestService = Depends(get_ingest_service)):
    """Upload a file, split it, embed, and store in database & vector store."""
    # Parse metadata if provided as JSON string
    doc_metadata = {}
    if metadata:
        try:
            import json
            doc_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    # Save uploaded file to a temporary location
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Use the ingest service
        doc = ingest_service.ingest_file(
            source=tmp_path,
            source_url=file.filename,
            document_title=title,
            document_metadata=doc_metadata,
        )
    except Exception as e:
        # Clean up temp file
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

    # Clean up
    os.unlink(tmp_path)
    return {"document": doc}

@router.post("/upload/html")
async def upload_html(html: str = Form(...), source_url: Optional[str] = Form(None), title: Optional[str] = Form(None), metadata: Optional[str] = Form(None), ingest_service: IngestService = Depends(get_ingest_service),):
    """Ingest raw HTML content."""
    doc_metadata = {}
    if metadata:
        try:
            import json
            doc_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    doc = ingest_service.ingest_html(
        html=html,
        source_url=source_url,
        document_title=title,
        document_metadata=doc_metadata,
    )
    return {"document": doc}

@router.post("/upload/url")
async def upload_url(url: str = Form(...), metadata: Optional[str] = Form(None), ingest_service: IngestService = Depends(get_ingest_service)):
    """Ingest a webpage from a URL."""
    doc_metadata = {}
    if metadata:
        try:
            import json
            doc_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    doc = ingest_service.ingest_url(url, document_metadata=doc_metadata)
    return {"document": doc}

@router.delete("/{doc_id}")
async def delete_document(doc_id: str, hard_delete: bool = Query(False), ingest_service: IngestService = Depends(get_ingest_service)):
    """Delete a document (soft by default, hard if hard_delete=true)."""
    success = ingest_service.delete_file(doc_id, hard_delete=hard_delete)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted"}

@router.put("/{doc_id}")
async def update_document(doc_id: str, title: Optional[str] = Form(None), metadata: Optional[str] = Form(None), db: Session = Depends(get_db),):
    """Update document metadata (title, metadata)."""
    doc_metadata = {}
    if metadata:
        try:
            import json
            doc_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    doc = DocumentService.update_document(db, doc_id, title=title, metadata=doc_metadata)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document": doc}

@router.get("/{doc_id}/chunks")
async def get_document_chunks(doc_id: str, db: Session = Depends(get_db)):
    """List all chunks of a document."""
    doc = DocumentService.get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = ChunkService.list_chunks_by_document(db, doc_id)
    return {"chunks": chunks}

@router.get("/{doc_id}/embeddings")
async def get_document_embeddings(doc_id: str, db: Session = Depends(get_db)):
    """List all embeddings of a document (vectors not stored in DB)."""
    doc = DocumentService.get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    embeddings = EmbeddingService.get_embeddings_by_document(db, doc_id)
    return {"embeddings": embeddings}