from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from httpx import AsyncClient


# Mock embedding: return a fixed 384-dim vector (matching ChromaDB DefaultEmbeddingFunction)
MOCK_EMBEDDING = [0.1] * 384


def _mock_embedding_response(texts):
    """Create a mock embeddings response."""
    mock = MagicMock()
    mock.data = [MagicMock(embedding=MOCK_EMBEDDING) for _ in texts]
    return mock


def _mock_chat_stream_response(content: str):
    """Create an async generator that yields chat completion chunks."""
    async def stream():
        for char in content:
            delta = MagicMock()
            delta.content = char
            choice = MagicMock()
            choice.delta = delta
            chunk = MagicMock()
            chunk.choices = [choice]
            yield chunk
    return stream()


@pytest.mark.asyncio
async def test_qa_ask_returns_answer_and_citations(
    test_app: AsyncClient, sample_txt_content: bytes
):
    """Upload a doc with known facts, ask about them, verify answer with citations."""
    # 1. Upload document containing "张三的生日是1990年5月1日"
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", sample_txt_content, "text/plain")},
    )
    assert resp.status_code == 201

    # 2. Mock the embedding and LLM calls for the QA request
    with patch(
        "app.services.embedding._get_embedding_client"
    ) as mock_emb_client:
        mock_emb = MagicMock()
        mock_emb.embeddings.create = AsyncMock(
            side_effect=lambda **kwargs: _mock_embedding_response(kwargs["input"])
        )
        mock_emb_client.return_value = mock_emb

        with patch(
            "app.services.rag_pipeline.get_client"
        ) as mock_llm_client:
            mock_llm = MagicMock()
            mock_llm.chat.completions.create = AsyncMock(
                return_value=_mock_chat_stream_response(
                    "根据资料，张三的生日是1990年5月1日[1]。"
                )
            )
            mock_llm_client.return_value = mock_llm

            # 3. Ask question via sync endpoint
            resp = await test_app.post(
                "/api/v1/qa/ask/sync",
                json={"question": "张三的生日是什么时候？", "top_k": 5},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert isinstance(data["citations"], list)
    assert isinstance(data["retrieved_chunks"], list)


@pytest.mark.asyncio
async def test_qa_compare_returns_both_answers(
    test_app: AsyncClient, sample_txt_content: bytes
):
    """Test the compare endpoint returns both RAG and bare answers."""
    # Upload a document first
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", sample_txt_content, "text/plain")},
    )
    assert resp.status_code == 201

    with patch(
        "app.services.embedding._get_embedding_client"
    ) as mock_emb_client:
        mock_emb = MagicMock()
        mock_emb.embeddings.create = AsyncMock(
            side_effect=lambda **kwargs: _mock_embedding_response(kwargs["input"])
        )
        mock_emb_client.return_value = mock_emb

        with patch(
            "app.services.rag_pipeline.get_client"
        ) as mock_llm_client:
            mock_llm = MagicMock()

            # Return RAG answer then bare answer
            call_count = [0]

            async def mock_create(**kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return _mock_chat_stream_response("RAG增强回答：Transformer是...")
                else:
                    return _mock_chat_stream_response("纯LLM回答：Transformer是一种深度学习模型...")

            mock_llm.chat.completions.create = AsyncMock(side_effect=mock_create)
            mock_llm_client.return_value = mock_llm

            resp = await test_app.post(
                "/api/v1/qa/compare/sync",
                json={"question": "什么是Transformer？", "top_k": 5},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["rag_answer"]) > 0
    assert len(data["bare_answer"]) > 0
    assert isinstance(data["rag_citations"], list)


@pytest.mark.asyncio
async def test_qa_empty_question_rejected(test_app: AsyncClient):
    """Empty question should be rejected with 422."""
    resp = await test_app.post(
        "/api/v1/qa/ask/sync",
        json={"question": "", "top_k": 5},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_qa_no_documents_returns_empty_retrieval(
    test_app: AsyncClient,
):
    """Asking when no documents are indexed should return empty retrieval."""
    # No documents uploaded -- just ask
    with patch(
        "app.services.embedding._get_embedding_client"
    ) as mock_emb_client:
        mock_emb = MagicMock()
        mock_emb.embeddings.create = AsyncMock(
            side_effect=lambda **kwargs: _mock_embedding_response(kwargs["input"])
        )
        mock_emb_client.return_value = mock_emb

        with patch(
            "app.services.rag_pipeline.get_client"
        ) as mock_llm_client:
            mock_llm = MagicMock()
            mock_llm.chat.completions.create = AsyncMock(
                return_value=_mock_chat_stream_response("我无法从知识库中找到相关信息。")
            )
            mock_llm_client.return_value = mock_llm

            resp = await test_app.post(
                "/api/v1/qa/ask/sync",
                json={"question": "什么是深度学习？", "top_k": 5},
            )

    assert resp.status_code == 200
    data = resp.json()
    # Retrieved chunks should be empty since no docs indexed
    assert data["retrieved_chunks"] == []
    # But LLM should still respond
    assert "answer" in data
