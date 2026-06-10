from dataclasses import dataclass, field
from typing import AsyncIterator

from app.services.llm import get_client, get_model
from app.services.vector_store import get_vector_store, VectorStore
from app.utils.config import RAG_TOP_K


@dataclass
class RetrievedChunk:
    doc_id: str
    filename: str
    chunk_index: int
    content: str
    content_preview: str
    score: float


@dataclass
class Citation:
    doc_id: str
    filename: str
    chunk_index: int
    content_snippet: str


RAG_PROMPT_TEMPLATE = """你是一个知识库问答助手。基于以下参考资料回答用户问题。如果资料不足以回答，请如实说明你无法从资料中找到相关信息。

参考资料：
{context}

用户问题：{question}

要求：
1. 回答准确、简洁
2. 引用资料内容时标注来源编号，如[1]、[2]
3. 如果资料不足以回答，明确说明"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Build context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"[{i + 1}] (来源: {chunk.filename})\n{chunk.content}")
    return "\n\n".join(parts)


def _extract_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Extract citation metadata from retrieved chunks."""
    return [
        Citation(
            doc_id=c.doc_id,
            filename=c.filename,
            chunk_index=c.chunk_index,
            content_snippet=c.content[:300],
        )
        for c in chunks
    ]


async def retrieve_chunks(
    question: str,
    top_k: int = RAG_TOP_K,
    store: VectorStore | None = None,
) -> list[RetrievedChunk]:
    """Embed question and retrieve relevant chunks from vector store."""
    if store is None:
        store = get_vector_store()

    results = await store.search(query=question, top_k=top_k)

    return [
        RetrievedChunk(
            doc_id=r["doc_id"],
            filename=r["filename"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            content_preview=r["content_preview"],
            score=r["score"],
        )
        for r in results
    ]


async def generate_answer_stream(
    question: str,
    chunks: list[RetrievedChunk],
) -> AsyncIterator[str]:
    """Generate LLM answer with context, yielding tokens."""
    client = get_client()
    model = get_model()

    context = _build_context(chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


async def generate_bare_answer_stream(question: str) -> AsyncIterator[str]:
    """Generate pure LLM answer without any context (for comparison)."""
    client = get_client()
    model = get_model()

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        stream=True,
    )

    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


async def rag_pipeline(
    question: str,
    top_k: int = RAG_TOP_K,
    store: VectorStore | None = None,
) -> tuple[list[RetrievedChunk], AsyncIterator[str], list[Citation]]:
    """Full RAG pipeline: retrieve → generate, returning all three components.

    Returns:
        (retrieved_chunks, answer_stream, citations)
    """
    chunks = await retrieve_chunks(question, top_k=top_k, store=store)
    answer_stream = generate_answer_stream(question, chunks)
    citations = _extract_citations(chunks)
    return chunks, answer_stream, citations
