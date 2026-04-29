"""
POST /api/chat — streaming RAG over Neon Postgres (pgvector).

1) Embed query + pgvector retrieval (sync, off thread).
2) Stream LLM tokens via model `.astream()` (true token streaming, not replay).

SSE `data:` lines are JSON.

- **source** (before tokens): numbered rows with catalog **URL**, optional **title**; indices match prompt `Source n`.
- **text**: **delta** (token chunk), **message** (cumulative assistant text including `[n, "verbatim"]` citations).
- **done**: final **message** plus **citations**: parsed citation markers enriched with **url**/ **title**
  per `source_index` (hydrated from streamed **source** events).

Citation protocol is defined on `prompt_service.rag_prompt_template` (marker shape `[n, "EXACT_TEXT"]`).
"""

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.prompt_service import extract_inline_citations
from app.services.rag_chain_service import rag_chain_service

logger = logging.getLogger(__name__)

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class ChatRequest(BaseModel):
    message: str


async def stream_rag_response(message: str):
    accumulated = ""
    sources_by_idx: dict[int, dict] = {}
    try:
        async for ev in rag_chain_service.astream_rag(message):
            et = ev.get("type")
            if et == "source":
                idx = ev.get("index", 0)
                src = ev.get("source") or "Unknown"
                sources_by_idx[idx] = {"source": src, "title": ev.get("title")}
                line = f"Source {idx}: {src}"
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "source",
                            "message": line,
                            "index": idx,
                            "source": src,
                            "title": ev.get("title"),
                        }
                    )
                    + "\n\n"
                )
            elif et == "text":
                delta = ev.get("delta") or ""
                accumulated += delta
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "text",
                            "delta": delta,
                            "message": accumulated,
                        }
                    )
                    + "\n\n"
                )
            elif et == "done":
                raw_cites = extract_inline_citations(accumulated)
                citations = []
                for c in raw_cites:
                    i = c["source_index"]
                    meta = sources_by_idx.get(i, {})
                    citations.append(
                        {
                            "source_index": i,
                            "quote": c["quote"],
                            "url": meta.get("source"),
                            "title": meta.get("title"),
                        }
                    )
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "done",
                            "message": accumulated,
                            "citations": citations,
                        }
                    )
                    + "\n\n"
                )
            elif et == "error":
                yield (
                    "data: "
                    + json.dumps({"type": "error", "message": ev.get("message", "Error")})
                    + "\n\n"
                )
    except Exception as e:
        logger.exception("SSE chat failed: %s", e)
        yield (
            "data: "
            + json.dumps({"type": "error", "message": "Error processing request"})
            + "\n\n"
        )


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Server-Sent Events: sources first (after retrieval), then streamed answer tokens.

    Retrieval uses Postgres/pgvector via `DATABASE_URL` (`SessionLocal` inside search).
    """
    return StreamingResponse(
        stream_rag_response(request.message),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
