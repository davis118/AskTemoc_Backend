from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
from typing import Optional
import tempfile
import os

from app.services.ingest_service import IngestService
from app.services.document_splitter import DocumentSplitter
from app.services.retriever_service import retriever_service
from app.schemas.document import DocumentResponse
from langchain_chroma import Chroma


router = APIRouter()


splitter = DocumentSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

vector_store = Chroma(
    client=retriever_service.client,
    collection_name=retriever_service.collection_name,
    embedding_function=retriever_service.embeddings,
)

ingest_service = IngestService(
    splitter=splitter,
    embedding_function=retriever_service.embeddings.embed_query,
    vector_store=vector_store
)


@router.get("/")
def health_check():
    return {"message": "Ingest service running"}


@router.post("/html", response_model=DocumentResponse)
def ingest_html(
    html: str = Form(...),
    source_url: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
):
    try:

        document = ingest_service.ingest_html(
            html=html,
            source_url=source_url,
            document_title=title
        )

        return document

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"HTML ingestion failed: {str(e)}"
        )


@router.post("/url", response_model=DocumentResponse)
def ingest_url(
    url: str = Form(...),
    timeout: int = Form(30),
):
    try:

        document = ingest_service.ingest_url(
            url=url,
            timeout=timeout
        )

        return document

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"URL ingestion failed: {str(e)}"
        )


@router.post("/file", response_model=DocumentResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source_url: Optional[str] = Form(None),
):

    ext = Path(file.filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:

        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        print(f"Document Name is: {file.filename}")
        document = ingest_service.ingest_file(
            source=tmp_path,
            source_url=source_url,
            document_title=file.filename
        )
        print(f"Document: {document}")
        return document

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"File ingestion failed: {str(e)}"
        )

    finally:

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
