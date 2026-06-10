import json
from typing import AsyncIterator

from sse_starlette.sse import EventSourceResponse

from app.services.rag_pipeline import (
    RetrievedChunk,
    Citation,
)


def _sse_event(event: str, data: object) -> dict:
    """Build an SSE event dict."""
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


def _chunks_to_preview(chunks: list[RetrievedChunk]) -> list[dict]:
    """Convert RetrievedChunk list to SSE-suitable dicts."""
    return [
        {
            "doc_id": c.doc_id,
            "filename": c.filename,
            "content_preview": c.content_preview,
            "score": c.score,
        }
        for c in chunks
    ]


def _citations_to_dicts(citations: list[Citation]) -> list[dict]:
    """Convert Citation list to SSE-suitable dicts."""
    return [
        {
            "doc_id": c.doc_id,
            "filename": c.filename,
            "chunk_index": c.chunk_index,
            "content_snippet": c.content_snippet,
        }
        for c in citations
    ]


async def rag_ask_sse(
    retrieved: list[RetrievedChunk],
    answer_stream: AsyncIterator[str],
    citations: list[Citation],
) -> AsyncIterator[dict]:
    """Generate SSE events for the /qa/ask endpoint (RAG mode).

    Events: retrieval → answer* → citations → done
    """
    # 1. Retrieval results
    yield _sse_event("retrieval", {"chunks": _chunks_to_preview(retrieved)})

    # 2. Answer tokens
    async for token in answer_stream:
        yield _sse_event("answer", token)

    # 3. Citations
    yield _sse_event("citations", _citations_to_dicts(citations))

    # 4. Done
    yield _sse_event("done", {})


async def bare_ask_sse(
    answer_stream: AsyncIterator[str],
) -> AsyncIterator[dict]:
    """Generate SSE events for the /qa/ask endpoint (bare LLM mode — no retrieval).

    Events: answer* → done
    """
    async for token in answer_stream:
        yield _sse_event("answer", token)

    yield _sse_event("done", {})


async def rag_compare_sse(
    rag_answer_stream: AsyncIterator[str],
    bare_answer_stream: AsyncIterator[str],
    citations: list[Citation],
) -> AsyncIterator[dict]:
    """Generate SSE events for the /qa/compare endpoint.

    Events: rag_answer* / bare_answer* (interleaved) → rag_citations → done
    """
    # Interleave: take from each stream alternately
    rag_done = False
    bare_done = False
    rag_iter = rag_answer_stream.__aiter__()
    bare_iter = bare_answer_stream.__aiter__()

    while not rag_done or not bare_done:
        if not rag_done:
            try:
                token = await rag_iter.__anext__()
                yield _sse_event("rag_answer", token)
            except StopAsyncIteration:
                rag_done = True

        if not bare_done:
            try:
                token = await bare_iter.__anext__()
                yield _sse_event("bare_answer", token)
            except StopAsyncIteration:
                bare_done = True

    # Citations
    yield _sse_event("rag_citations", _citations_to_dicts(citations))

    # Done
    yield _sse_event("done", {})
