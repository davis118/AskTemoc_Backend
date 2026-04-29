from __future__ import annotations

import re
from typing import TypedDict

from langchain_core.prompts import PromptTemplate


class ParsedCitation(TypedDict):
    """One inline citation marker parsed from model output."""

    source_index: int
    quote: str


# Context is built by format_numbered_retrieval_context() — each chunk is ### Source n with URL/title/body.
template = """
You are an expert at answering questions about The University of Texas at Dallas (UTD).
Your audience is students, faculty, and staff. Be clear and accurate.

If you do not know the answer based on the sources below, say clearly that you do not know—do not guess or invent facts.
Do not make up an answer.

You are given NUMBERED SOURCES below. Each block is independent: Source 1, Source 2, …

CITATION PROTOCOL (required for factual claims):
- For every sentence or clause that rests on the sources, add an INLINE citation using this exact pattern (machine-parseable):
  [n, "EXACT_TEXT"]
  where:
    • n is the source number (integer) matching ### Source n in the context.
    • EXACT_TEXT is a CONTIGUOUS substring copied character-for-character from that source’s body text (between the --- lines). You may use a shorter snippet, but it must be an exact copy—no paraphrase inside the quotes.
- Place the citation immediately after the claim it supports (same line or end of sentence).
- You may repeat the same n with different EXACT_TEXT snippets when needed.
- If the context does not contain the answer, say you do not know; do not fabricate citations or supported claims.
- Do not cite a source index that does not appear in the context.

Example shape (illustration only):
To earn a minor in Computer Science at UTD, students complete 21 semester credit hours as described in the program requirements [1, "the exact sentence or phrase copied from Source 1"].

---

NUMBERED SOURCES:
{context}

---

Question:
{question}

Answer (use the citation pattern [n, "EXACT_TEXT"] for supported facts):
"""

rag_prompt_template = PromptTemplate.from_template(template)


def format_numbered_retrieval_context(
    chunks: list[tuple[str, str | None, str | None]],
) -> str:
    """
    Format retrieved chunks for the citation protocol.

    Each tuple is (body_text, url, title). Indices are 1-based ### Source n blocks.
    """
    parts: list[str] = []
    for i, (body, url, title) in enumerate(chunks, start=1):
        url_s = (url or "Unknown").strip()
        title_s = (title or "").strip()
        body_s = (body or "").strip()
        title_line = f"Title: {title_s}\n" if title_s else ""
        parts.append(
            f"### Source {i}\n"
            f"URL: {url_s}\n"
            f"{title_line}"
            f"---\n"
            f"{body_s}\n"
            f"---"
        )
    if not parts:
        return "(No retrieval context.)"
    return "\n\n".join(parts)


# Inline citations: [1, "verbatim quote"] possibly with escapes inside quotes
_INLINE_CITE_RE = re.compile(
    r"\[\s*(\d+)\s*,\s*\"((?:[^\"\\]|\\.)*)\"\s*\]"
)


def extract_inline_citations(answer: str) -> list[ParsedCitation]:
    """
    Parse citation markers from the assistant answer for the frontend.

    Returns ordered list of {source_index, quote}. Duplicates preserved if repeated.
    """
    out: list[ParsedCitation] = []
    for m in _INLINE_CITE_RE.finditer(answer or ""):
        idx = int(m.group(1))
        raw_q = m.group(2).replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
        out.append({"source_index": idx, "quote": raw_q})
    return out
