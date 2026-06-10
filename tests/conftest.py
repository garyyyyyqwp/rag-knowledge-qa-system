import os
import shutil
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


@pytest.fixture(scope="function", autouse=True)
def setup_test_env(monkeypatch):
    """Ensure tests use in-memory ChromaDB and test-specific configs."""
    # Use a temp directory for ChromaDB (isolated per test session)
    tmpdir = tempfile.mkdtemp(prefix="chroma_test_")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", tmpdir)
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_knowledge_base")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "512")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "50")
    monkeypatch.setenv("RAG_TOP_K", "5")

    # Unset real API keys to prevent accidental API calls in tests
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-placeholder")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-placeholder")

    # Reset VectorStore singleton and patch get_vector_store.
    # The default persist_dir on VectorStore.__init__ is baked at import time,
    # so we replace get_vector_store with a closure that passes tmpdir.
    # We MUST also patch every consumer module that imports get_vector_store
    # via `from app.services.vector_store import get_vector_store` because
    # monkeypatching the source module does not update those local bindings.
    import app.services.vector_store as vs

    vs._store = None

    def _get_vector_store():
        if vs._store is not None:
            return vs._store
        store = vs.VectorStore(persist_dir=tmpdir)
        vs._store = store
        return store

    monkeypatch.setattr(vs, "get_vector_store", _get_vector_store)

    # Lazy-import consumer modules and patch their local bindings too
    import app.routers.documents as documents_mod
    import app.services.rag_pipeline as rag_mod
    monkeypatch.setattr(documents_mod, "get_vector_store", _get_vector_store)
    monkeypatch.setattr(rag_mod, "get_vector_store", _get_vector_store)

    yield

    # Cleanup temp dir
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def sample_txt_content() -> bytes:
    """Sample TXT content with specific facts for testing."""
    text = """深度学习基础笔记

## 神经网络基础

神经网络由多个神经元层组成。每个神经元接收输入，通过激活函数产生输出。

## 反向传播算法

反向传播算法由Rumelhart等人在1986年提出。该算法利用链式法则计算损失函数相对于每个权重的梯度。

关键步骤：
1. 前向传播：计算网络输出
2. 计算损失：比较预测值与真实值
3. 反向传播误差：从输出层向输入层传播
4. 更新权重：使用梯度下降优化器

张三的生日是1990年5月1日。他在北京工作。

## Transformer架构

Transformer架构由Vaswani等人在2017年提出，核心创新是自注意力机制（Self-Attention）。
它抛弃了传统的RNN结构，完全基于注意力机制处理序列数据。
"""
    return text.encode("utf-8")


@pytest.fixture
def sample_md_content() -> bytes:
    """Sample Markdown content for testing."""
    text = """# 测试文档

## 第一节

这是第一节的内容，包含一些基本信息。

## 第二节

这是第二节的内容，包含更多详细信息。
特别是关于Python编程语言的内容。
Python是一种解释型、面向对象的高级编程语言。
"""
    return text.encode("utf-8")


@pytest.fixture(autouse=True)
def mock_embedding(monkeypatch):
    """Replace embedding with ChromaDB's on-device model to avoid real API calls."""
    from chromadb.utils import embedding_functions

    ef = embedding_functions.DefaultEmbeddingFunction()

    async def mock_embed_texts(texts: list[str]) -> list[list[float]]:
        return ef(texts)

    async def mock_embed_single(text: str) -> list[float]:
        result = ef([text])
        return result[0]

    # Patch in vector_store since it does `from app.services.embedding import embed_texts`
    monkeypatch.setattr("app.services.vector_store.embed_texts", mock_embed_texts)


@pytest_asyncio.fixture
async def test_app():
    """Create a TestClient for the FastAPI app."""
    from main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
