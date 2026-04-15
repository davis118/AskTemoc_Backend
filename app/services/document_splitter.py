from pathlib import Path  
from typing import Optional, List, Union  
from pydantic import BaseModel  
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat  
from docling.chunking import HybridChunker  
from docling.datamodel.backend_options import HTMLBackendOptions  
from app.models.requests import EmbedItem, EmbedBatch
from docling.datamodel.document import ConversionResult

from .html_processing_pipeline import HTMLProcessingPipeline, ChunkResult

from enum import Enum

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse
class SourceType(str, Enum):
    HTML_URL = "html_url"
    PDF_URL = "pdf_url"
    WORD_URL = "word_url"
    GENERIC_URL = "generic_url"

    HTML_FILE = "html_file"
    PDF_FILE = "pdf_file"
    WORD_FILE = "word_file"
    GENERIC_FILE = "generic_file"

    RAW_HTML = "raw_html"
    UNKNOWN = "unknown"
    

class DocumentSplitter:  
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):  
        """  
        Initialize the DocumentSplitter.  
          
        Args:  
            chunk_size: Maximum size of text chunks (default: 1000).  
            chunk_overlap: Number of characters to overlap between chunks (default: 200).  
        """  
        self.chunk_size = chunk_size  
        self.chunk_overlap = chunk_overlap  
          
        self.html_pipeline = HTMLProcessingPipeline()  
          
        html_options = HTMLBackendOptions(  
            fetch_images=False,  
            add_title=True,  
            infer_furniture=True  
        )  
          
        self.converter = DocumentConverter()  
           
        self.chunker = HybridChunker(  
            max_tokens=chunk_size,  
            merge_peers=True,  
            tokenizer=None  
        )  
    
    def _detect_source_type(self, source: Union[str, Path]) -> SourceType:
        """
        Detect the type of the input source.
        """

        if isinstance(source, Path):
            source = str(source)

        if not isinstance(source, str):
            return SourceType.UNKNOWN

        if source.startswith(("http://", "https://")):
            parsed = urlparse(source)
            path = parsed.path.lower()

            if path.endswith((".html", ".htm")):
                return SourceType.HTML_URL
            if path.endswith(".pdf"):
                return SourceType.PDF_URL
            if path.endswith((".doc", ".docx")):
                return SourceType.WORD_URL

            return SourceType.GENERIC_URL

        file_path = Path(source)

        if file_path.exists():
            suffix = file_path.suffix.lower()

            if suffix in [".html", ".htm"]:
                return SourceType.HTML_FILE
            if suffix == ".pdf":
                return SourceType.PDF_FILE
            if suffix in [".doc", ".docx"]:
                return SourceType.WORD_FILE

            return SourceType.GENERIC_FILE

        if "<html" in source.lower():
            return SourceType.RAW_HTML

        return SourceType.UNKNOWN

    def _chunk_result_to_embed_item(self, chunk_result: ChunkResult, idx: int) -> EmbedItem:  
        """Convert ChunkResult to EmbedItem."""
        chunk_id = f"{chunk_result.source}_{idx+1}"  
          
        metadata = {  
            "source": chunk_result.source,  
            **(chunk_result.metadata.dict() if chunk_result.metadata else {})  
        }  
          
        return EmbedItem(  
            chunk_id=chunk_id,  
            text=chunk_result.content,  
            metadata=metadata  
        )  
  
    def process_html(self, html_content: str, source_url: Optional[str] = None) -> EmbedBatch:  
        """Process HTML content using custom pipeline."""  
        chunks = self.html_pipeline.process(html_content, source_url)  
        items = [self._chunk_result_to_embed_item(chunk, i) for i, chunk in enumerate(chunks)]  
        return EmbedBatch(items=items)  
  
    def process_html_from_url(self, url: str, timeout: int = 30) -> EmbedBatch:
        """Fetch and process HTML from URL with retries and better headers."""

        session = requests.Session()

        # Retry strategy for dropped connections, 5xx errors, etc.
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        session.mount("http://", HTTPAdapter(max_retries=retries))

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            )
        }

        try:
            response = session.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RemoteDisconnected as e:
            raise RuntimeError(f"Server closed connection unexpectedly: {e}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch URL {url}: {e}") from e
        print(response.text)
        return self.process_html(response.text, url)
  
    def _create_embed_batch_from_docling(self, conv_result: ConversionResult, source_url: Optional[str] = None) -> EmbedBatch:  
        """Convert DoclingDocument to EmbedBatch using chunking."""  
        doc = conv_result.document  
        chunks = list(self.chunker.chunk(doc))  
          
        items = []  
        for i, chunk in enumerate(chunks):  
            chunk_id = f"{source_url or 'doc'}_{i+1}" if len(chunks) > 1 else str(source_url or 'doc_1')  
              
            metadata = {  
                "source_url": source_url,  
                "page_no": getattr(chunk.meta, "page_no", None),  
                "headings": getattr(chunk.meta, "headings", []),  
                "captions": getattr(chunk.meta, "captions", []),  
            }  
              
            items.append(EmbedItem(  
                chunk_id=chunk_id,  
                text=chunk.text,  
                metadata=metadata  
            ))  
          
        return EmbedBatch(items=items)  
  
    def process_pdf(self, file_path: str, source_url: Optional[str] = None) -> EmbedBatch:  
        """Process PDF file using Docling."""  
        if not source_url:  
            source_url = f"file://{file_path}"  
          
        conv_result = self.converter.convert(file_path)  
        return self._create_embed_batch_from_docling(conv_result, source_url)  
  
    def process_word(self, file_path: str, source_url: Optional[str] = None) -> EmbedBatch:  
        """Process DOCX file using Docling."""  
        if not source_url:  
            source_url = f"file://{file_path}"  
          
        conv_result = self.converter.convert(file_path)  
        return self._create_embed_batch_from_docling(conv_result, source_url)  
  
    def process_any(self, source: Union[str, Path], source_url: Optional[str] = None) -> EmbedBatch:  
        """Process any supported format using Docling's automatic format detection."""  
        if isinstance(source, str) and source.startswith(('http://', 'https://')):  
            conv_result = self.converter.convert(source)  
            source_url = source_url or source  
        else:  
            conv_result = self.converter.convert(source)  
            source_url = source_url or f"file://{source}"  
          
        return self._create_embed_batch_from_docling(conv_result, source_url)
    
    def process_file_old(self, source: Union[str, Path], source_url: Optional[str] = None) -> EmbedBatch:
        """
        Orchestrator method that detects file type and routes
        to the appropriate processing method.

        Args:
            source: File path, URL, or raw HTML string
            source_url: Optional explicit source URL for metadata

        Returns:
            EmbedBatch
        """

        if isinstance(source, Path):
            source = str(source)

        # handle URL's
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            parsed = urlparse(source)
            path = parsed.path.lower()

            if path.endswith((".html", ".htm")):
                return self.process_html_from_url(source)

            if path.endswith(".pdf"):
                return self.process_pdf(source, source_url=source)

            if path.endswith((".docx", ".doc")):
                return self.process_word(source, source_url=source)

            return self.process_any(source, source_url=source)

        # handle local files
        if isinstance(source, str):
            file_path = Path(source)

            if file_path.exists():
                suffix = file_path.suffix.lower()

                if suffix in [".html", ".htm"]:
                    html_content = file_path.read_text(encoding="utf-8")
                    return self.process_html(html_content, source_url or f"file://{file_path}")

                if suffix == ".pdf":
                    return self.process_pdf(str(file_path), source_url)

                if suffix in [".docx", ".doc"]:
                    return self.process_word(str(file_path), source_url)

                # Unknown file type -> fallback
                return self.process_any(str(file_path), source_url)

        # handle raw HTML
        if isinstance(source, str) and "<html" in source.lower():
            return self.process_html(source, source_url)

        # final fallback
        return self.process_any(source, source_url)
    
    def process_file(self, source: Union[str, Path], source_url: Optional[str] = None) -> EmbedBatch:
        """
        Main ingestion entry point.
        Automatically detects source type and routes to the correct processor.
        """

        source_type = self._detect_source_type(source)

        # url types
        if source_type == SourceType.HTML_URL:
            return self.process_html_from_url(source)

        if source_type == SourceType.PDF_URL:
            return self.process_pdf(source, source_url=source)

        if source_type == SourceType.WORD_URL:
            return self.process_word(source, source_url=source)

        if source_type == SourceType.GENERIC_URL:
            return self.process_any(source, source_url=source)

        # local file types
        if source_type == SourceType.HTML_FILE:
            file_path = Path(source)
            html_content = file_path.read_text(encoding="utf-8")
            return self.process_html(
                html_content,
                source_url or f"file://{file_path}"
            )

        if source_type == SourceType.PDF_FILE:
            return self.process_pdf(str(source), source_url)

        if source_type == SourceType.WORD_FILE:
            return self.process_word(str(source), source_url)

        if source_type == SourceType.GENERIC_FILE:
            return self.process_any(source, source_url)

        # Raw HTML
        if source_type == SourceType.RAW_HTML:
            return self.process_html(source, source_url)

        # final fallback
        return self.process_any(source, source_url)
    