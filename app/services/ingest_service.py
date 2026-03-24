from typing import Union, List, Optional, Dict, Any, Callable
from pathlib import Path
import logging
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.db.services import DocumentService, ChunkService, EmbeddingService
from app.db.models import Document, Chunk, Embedding
from app.services.document_splitter import DocumentSplitter, EmbedBatch
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class IngestService:
    def __init__(self, splitter: DocumentSplitter, embedding_function: Callable[[str], List[float]], vector_store, db_session_factory=SessionLocal):
        self.splitter = splitter
        self.embedding_function = embedding_function
        self.vector_store = vector_store
        self.db_session_factory = db_session_factory

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        clean = {}

        for k, v in metadata.items():

            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v

            elif isinstance(v, list):
                clean[k] = ", ".join(str(i) for i in v)

            elif isinstance(v, dict):
                clean[k] = str(v)

            else:
                clean[k] = str(v)

        return clean

    def ingest_file_old(self, source: Union[str, Path], source_url: Optional[str] = None, document_title: Optional[str] = None, document_metadata: Optional[Dict[str, Any]] = None) -> Document:
        # split the source into chunks
        batch: EmbedBatch = self.splitter.process_file(source, source_url)
        if not batch.items:
            logger.warning("No chunks generated from source: %s", source)
            # We'll assume at least one chunk is required.
            raise ValueError("No content extracted from source")

        # determine document source string (for uniqueness check)
        doc_source = source_url or str(source)

        # start a database session
        db: Session = self.db_session_factory()
        try:
            # check if document already exists by source
            existing_doc = DocumentService.get_document_by_source(db, doc_source)
            if existing_doc:
                logger.info("Document already exists, updating: %s", doc_source)
                # update existing document: soft delete old chunks & embeddings, then add new ones
                self._replace_document_chunks(db, existing_doc, batch, document_metadata)
                doc = existing_doc
            else:
                logger.info("Creating new document: %s", doc_source)
                # create new document
                title = document_title or self._derive_title_from_source(source, source_url)
                doc = DocumentService.create_document(
                    db,
                    title=title,
                    source=doc_source,
                    metadata=document_metadata or {},
                )
                # add new chunks
                self._add_chunks_to_document(db, doc, batch)

            db.commit()
            db.refresh(doc)
            return doc

        except Exception as e:
            db.rollback()
            logger.error("Ingestion failed for %s: %s", doc_source, e, exc_info=True)
            raise
        finally:
            db.close()

    def _ingest_batch(self, batch: EmbedBatch, source: str, document_title: Optional[str], document_metadata: Optional[Dict[str, Any]] = None,) -> Document:

        if not batch.items:
            raise ValueError("No content extracted from source")

        db: Session = self.db_session_factory()

        try:

            existing_doc = DocumentService.get_document_by_source(db, source)
            print(f"Source: {source}")
            print(f"Existing Doc: {existing_doc}")
            if existing_doc:
                logger.info("Document exists, replacing chunks: %s", source)

                self._replace_document_chunks(
                    db,
                    existing_doc,
                    batch,
                    document_metadata
                )

                doc = existing_doc

            else:
                logger.info("Creating new document: %s", source)
                print(f"Document Title: {document_title}")
                title = document_title or self._derive_title_from_source(source, None)
                print(f"Title: {title}")
                doc = DocumentService.create_document(
                    db,
                    title=title,
                    source=source,
                    metadata=document_metadata or {},
                )

                self._add_chunks_to_document(db, doc, batch)

            db.commit()
            db.refresh(doc)

            return doc

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()
    
    def ingest_file(self,source: Union[str, Path], source_url: Optional[str] = None, document_title: Optional[str] = None, document_metadata: Optional[Dict[str, Any]] = None,) -> Document:

        batch: EmbedBatch = self.splitter.process_file(source, source_url)

        doc_source = source_url or str(source)

        return self._ingest_batch(
            batch,
            doc_source,
            document_title,
            document_metadata
        )
    
    def ingest_html(self, html: str, source_url: Optional[str] = None, document_title: Optional[str] = None, document_metadata: Optional[Dict[str, Any]] = None,) -> Document:
        batch = self.splitter.process_html(html, source_url)

        source = source_url or "html_content"

        return self._ingest_batch(
            batch,
            source,
            document_title,
            document_metadata
        )
    
    def ingest_url( self, url: str, timeout: int = 30, document_metadata: Optional[Dict[str, Any]] = None,) -> Document:

        batch = self.splitter.process_html_from_url(url, timeout)

        return self._ingest_batch(
            batch,
            url,
            None,
            document_metadata
        )
    
    def delete_file(self, document_id: str, hard_delete: bool = False) -> bool:
        db: Session = self.db_session_factory()
        try:
            # fetch the document (including soft-deleted if hard_delete)
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                logger.warning("Document not found for deletion: %s", document_id)
                return False

            # If hard deleting, we need to delete vectors from vector store first
            if hard_delete:
                # collect all embedding chroma_ids
                chroma_ids = []
                for chunk in doc.chunks:
                    for emb in chunk.embeddings:
                        if emb.chroma_id:
                            chroma_ids.append(emb.chroma_id)
                if chroma_ids:
                    self.vector_store.delete(chroma_ids)
                    logger.info("Deleted %d vectors from vector store", len(chroma_ids))

            # use the service method to delete (soft or hard)
            success = DocumentService.delete_document(db, document_id, hard_delete)
            db.commit()
            return success

        except Exception as e:
            db.rollback()
            logger.error("Deletion failed for document %s: %s", document_id, e, exc_info=True)
            raise
        finally:
            db.close()

    def update_file(self, source: Union[str, Path], source_url: Optional[str] = None, document_title: Optional[str] = None, document_metadata: Optional[Dict[str, Any]] = None,) -> Document:
        return self.ingest_file(source, source_url, document_title, document_metadata)

    def read_files(self, skip: int = 0, limit: int = 100) -> List[Document]:
        db: Session = self.db_session_factory()
        try:
            return DocumentService.list_documents(db, skip=skip, limit=limit)
        finally:
            db.close()

    def _replace_document_chunks(self, db: Session, document: Document, batch: EmbedBatch, new_doc_metadata: Optional[Dict[str, Any]] = None) -> None:
        # collect all chroma_ids from existing embeddings for vector store deletion
        chroma_ids = []
        for chunk in document.chunks:
            for emb in chunk.embeddings:
                if emb.chroma_id:
                    chroma_ids.append(emb.chroma_id)

        # delete vectors from vector store
        if chroma_ids:
            self.vector_store.delete(chroma_ids)
            logger.info("Deleted %d old vectors from vector store for document %s", len(chroma_ids), document.id)

        # soft delete all existing chunks (cascade will also soft delete embeddings via relationship)
        for chunk in document.chunks:
            ChunkService.delete_chunk(db, chunk.id, hard_delete=False)

        # optionally update document metadata
        if new_doc_metadata:
            document.doc_metadata = {**(document.doc_metadata or {}), **new_doc_metadata}
            document.updated_at = datetime.utcnow()
            db.add(document)

        # add new chunks
        self._add_chunks_to_document(db, document, batch)

    def _add_chunks_to_document(self, db: Session, document: Document, batch: EmbedBatch) -> None:

        texts = []
        metadatas = []
        embedding_ids = []

        for idx, item in enumerate(batch.items):

            chunk = ChunkService.create_chunk(
                db,
                document_id=document.id,
                chunk_index=idx,
                text=item.text,
                metadata=item.metadata,
            )

            embedding = EmbeddingService.create_embedding(
                db,
                chunk_id=chunk.id,
                vector=None,
                model="nomic-embed-text",
            )            

            texts.append(item.text)

            metadata = {
                "chunk_id": chunk.id,
                "document_id": document.id,
                "source": document.source,
                **item.metadata,
            }

            metadatas.append(self._sanitize_metadata(metadata))

            embedding_ids.append(embedding.id)

        db.flush()

        try:
            
            chroma_ids = [str(uuid.uuid4()) for _ in texts]

            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=chroma_ids,
            )

        except Exception:
            logger.error("Vector store insertion failed")
            raise

        for emb_id, chroma_id in zip(embedding_ids, chroma_ids):
            EmbeddingService.mark_synced(db, emb_id, chroma_id)

    def _derive_title_from_source(self, source: Union[str, Path], source_url: Optional[str]) -> str:
        if source_url:
            # Use last part of URL path
            from urllib.parse import urlparse
            path = urlparse(source_url).path
            if path and path != "/":
                return Path(path).stem or source_url
            return source_url
        if isinstance(source, Path):
            return source.stem
        if isinstance(source, str):
            # Try to treat as file path
            p = Path(source)
            if p.exists():
                return p.stem
            # Fallback to the first 50 chars
            return source[:50]
        return "Untitled"