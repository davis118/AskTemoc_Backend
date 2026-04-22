"""
Builds schema-compliant page payloads and sends them to the ingest server.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

INGEST_URL = "http://localhost:8000/api/ingest"


def build_payload(
    url: str,
    page_title: str,
    text_content: Optional[str] = None,
    html_content: Optional[str] = None,
    chunks: Optional[list] = None,
) -> dict:
    """
    Build a schema-compliant page payload.

    Provide either text_content/html_content (raw) or chunks (pre-chunked) — not both.
    """
    return {
        "url": url,
        "page_title": page_title,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_content": text_content if not chunks else None,
        "html_content": html_content if not chunks else None,
        "chunks": chunks or [],
    }


async def publish_page(
    url: str,
    page_title: str,
    text_content: Optional[str] = None,
    html_content: Optional[str] = None,
    chunks: Optional[list] = None,
) -> bool:
    """
    Build and POST a page payload to the ingest server.

    Returns True on success, False on failure.
    """
    payload = build_payload(url, page_title, text_content, html_content, chunks)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(INGEST_URL, json=payload, timeout=30)
            resp.raise_for_status()
            logger.info(f"Published: {url}")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"Server rejected payload for {url}: {e.response.status_code} {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Failed to reach ingest server for {url}: {e}")

    return False
