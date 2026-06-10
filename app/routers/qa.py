from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.schemas.qa import (
    QuestionRequest,
    AskResponse,
    CompareResponse,
    CitationInfo,
    RetrievedChunkPreview,
)
from app.services.rag_pipeline import (
    retrieve_chunks,
    generate_answer_stream,
    generate_bare_answer_stream,
    extract_citations,
)
from app.services.llm import get_client, get_model
from app.services.streaming import rag_ask_sse, rag_compare_sse, bare_ask_sse
from app.services.embedding import EmbeddingError

router = APIRouter(tags=["qa"])


@router.post("/ask")
async def ask_question(request: QuestionRequest):
    """QA with SSE streaming — supports both RAG and bare LLM modes."""
    mode = (request.mode or "rag").strip().lower()

    if mode == "bare":
        # Bare LLM: no retrieval, no citations
        answer_stream = generate_bare_answer_stream(request.question)
        return EventSourceResponse(bare_ask_sse(answer_stream))

    # RAG mode (default)
    try:
        retrieved = await retrieve_chunks(question=request.question, top_k=request.top_k)
        answer_stream = generate_answer_stream(request.question, retrieved)
        citations = extract_citations(retrieved)
    except EmbeddingError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return EventSourceResponse(
        rag_ask_sse(
            retrieved=retrieved,
            answer_stream=answer_stream,
            citations=citations,
        )
    )


@router.post("/ask/sync", response_model=AskResponse)
async def ask_question_sync(request: QuestionRequest):
    """RAG-enhanced QA with non-streaming response (for testing)."""
    retrieved = await retrieve_chunks(question=request.question, top_k=request.top_k)

    # Collect full answer
    answer_parts = []
    async for token in generate_answer_stream(request.question, retrieved):
        answer_parts.append(token)
    answer = "".join(answer_parts)

    citations = extract_citations(retrieved)

    return AskResponse(
        answer=answer,
        citations=[
            CitationInfo(
                doc_id=c.doc_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                content_snippet=c.content_snippet,
            )
            for c in citations
        ],
        retrieved_chunks=[
            RetrievedChunkPreview(
                doc_id=r.doc_id,
                filename=r.filename,
                content_preview=r.content_preview,
                score=r.score,
            )
            for r in retrieved
        ],
    )


@router.post("/compare")
async def compare_rag_vs_bare(request: QuestionRequest):
    """Compare RAG-enhanced vs bare LLM answers with SSE streaming."""
    try:
        # RAG side
        retrieved = await retrieve_chunks(question=request.question, top_k=request.top_k)
        rag_stream = generate_answer_stream(request.question, retrieved)
        citations = extract_citations(retrieved)
    except EmbeddingError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Bare LLM side
    bare_stream = generate_bare_answer_stream(request.question)

    return EventSourceResponse(
        rag_compare_sse(
            rag_answer_stream=rag_stream,
            bare_answer_stream=bare_stream,
            citations=citations,
        )
    )


@router.post("/compare/sync", response_model=CompareResponse)
async def compare_rag_vs_bare_sync(request: QuestionRequest):
    """Compare RAG vs bare LLM with non-streaming response (for testing)."""
    # RAG side
    retrieved = await retrieve_chunks(question=request.question, top_k=request.top_k)
    rag_parts = []
    async for token in generate_answer_stream(request.question, retrieved):
        rag_parts.append(token)
    rag_answer = "".join(rag_parts)

    # Bare LLM side
    bare_parts = []
    async for token in generate_bare_answer_stream(request.question):
        bare_parts.append(token)
    bare_answer = "".join(bare_parts)

    citations = extract_citations(retrieved)

    return CompareResponse(
        rag_answer=rag_answer,
        bare_answer=bare_answer,
        rag_citations=[
            CitationInfo(
                doc_id=c.doc_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                content_snippet=c.content_snippet,
            )
            for c in citations
        ],
    )
