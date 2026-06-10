# 个人知识库智能问答系统 (RAG) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI-based RAG knowledge QA system with document upload, ChromaDB vector storage, and SSE-streaming LLM answers with citations.

**Architecture:** Layered FastAPI app (routers → schemas → services). Documents are parsed to text, split with hybrid chunking (semantic boundaries + token-aware), embedded via configurable OpenAI/DeepSeek API, stored in ChromaDB PersistentClient. Queries flow Embed → Retrieve → Generate with structured SSE events.

**Tech Stack:** FastAPI + Pydantic v2, OpenAI SDK (AsyncOpenAI), ChromaDB (persistent), pdfplumber, sse-starlette, tiktoken, pytest + httpx

---

## File Structure Map

```
week2task/
├── .env.example                              # NEW: environment template
├── .gitignore                                # NEW: python/chroma patterns
├── requirements.txt                          # NEW: all deps
├── main.py                                   # NEW: FastAPI app + router mounting
├── app/
│   ├── __init__.py                           # NEW: empty
│   ├── routers/
│   │   ├── __init__.py                       # NEW: empty
│   │   ├── documents.py                      # NEW: upload/list/delete endpoints
│   │   └── qa.py                             # NEW: ask/compare SSE endpoints
│   ├── schemas/
│   │   ├── __init__.py                       # NEW: empty
│   │   ├── document.py                       # NEW: request/response models
│   │   └── qa.py                             # NEW: question/citation models
│   ├── services/
│   │   ├── __init__.py                       # NEW: empty
│   │   ├── llm.py                            # NEW: AsyncOpenAI client wrapper
│   │   ├── embedding.py                      # NEW: configurable embedding provider
│   │   ├── chunker.py                        # NEW: hybrid chunking engine
│   │   ├── parser.py                         # NEW: PDF/MD/TXT text extraction
│   │   ├── vector_store.py                   # NEW: ChromaDB CRUD + doc listing
│   │   ├── rag_pipeline.py                   # NEW: orchestrate retrieve→generate
│   │   └── streaming.py                      # NEW: SSE event formatters
│   └── utils/
│       ├── __init__.py                       # NEW: empty
│       └── config.py                         # NEW: env var loader + validation
├── tests/
│   ├── __init__.py                           # NEW: empty
│   ├── conftest.py                           # NEW: fixtures (test client, temp chroma)
│   ├── test_chunker.py                       # NEW: chunking unit tests
│   ├── test_documents.py                     # NEW: upload/list/delete integration
│   └── test_qa.py                            # NEW: ask/compare integration
```

---

### Task 0: Project Scaffolding

**Files:**
- Create: `.env.example`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `app/routers/__init__.py`
- Create: `app/schemas/__init__.py`
- Create: `app/services/__init__.py`
- Create: `app/utils/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `.env.example`**

```env
# LLM
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

# Embedding (defaults to LLM config if unset)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-your-api-key-here
EMBEDDING_BASE_URL=https://api.openai.com/v1

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_data
CHROMA_COLLECTION_NAME=knowledge_base

# Chunker
CHUNK_MAX_TOKENS=512
CHUNK_OVERLAP_TOKENS=50

# RAG
RAG_TOP_K=5
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.py[cod]
*$py.class
*.so
.env
chroma_data/
.venv/
venv/
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
```

- [ ] **Step 3: Write `requirements.txt`**

```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0
python-multipart>=0.0.9
openai>=1.30.0
sse-starlette>=2.1.0
chromadb>=0.5.0
pdfplumber>=0.11.0
python-dotenv>=1.0.0
tiktoken>=0.7.0
pytest>=8.0.0
httpx>=0.27.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 4: Create empty `__init__.py` files**

Run:
```bash
touch app/__init__.py app/routers/__init__.py app/schemas/__init__.py app/services/__init__.py app/utils/__init__.py tests/__init__.py
```

- [ ] **Step 5: Install dependencies**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && pip install -r requirements.txt
```
Expected: all packages install without error.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: project scaffolding - env, gitignore, deps, init files"
```

---

### Task 1: Config Module (utils/config.py)

**Files:**
- Create: `app/utils/config.py`

- [ ] **Step 1: Write config.py with all env variables**

```python
import os
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(
            f"Environment variable '{key}' is not set. "
            f"Please set it in .env file or in the environment."
        )
    return value


# --- LLM ---
OPENAI_API_KEY = get_env("OPENAI_API_KEY", required=True)
OPENAI_MODEL = get_env("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = get_env("OPENAI_BASE_URL", "https://api.openai.com/v1")

# --- Embedding ---
EMBEDDING_PROVIDER = get_env("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_API_KEY = get_env("EMBEDDING_API_KEY", OPENAI_API_KEY)
EMBEDDING_BASE_URL = get_env("EMBEDDING_BASE_URL", OPENAI_BASE_URL)

# --- ChromaDB ---
CHROMA_PERSIST_DIR = get_env("CHROMA_PERSIST_DIR", "./chroma_data")
CHROMA_COLLECTION_NAME = get_env("CHROMA_COLLECTION_NAME", "knowledge_base")

# --- Chunker ---
CHUNK_MAX_TOKENS = int(get_env("CHUNK_MAX_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(get_env("CHUNK_OVERLAP_TOKENS", "50"))

# --- RAG ---
RAG_TOP_K = int(get_env("RAG_TOP_K", "5"))
```

- [ ] **Step 2: Verify config loads**

Run (from project root):
```bash
cd C:/Users/35452/Desktop/week2task && python -c "from app.utils.config import OPENAI_MODEL, CHUNK_MAX_TOKENS; print(f'Model: {OPENAI_MODEL}, Chunk: {CHUNK_MAX_TOKENS}')"
```
Expected: prints default values (no .env needed for defaults).

- [ ] **Step 3: Commit**

```bash
git add app/utils/config.py && git commit -m "feat: add config module with all env variables"
```

---

### Task 2: LLM Client Service (services/llm.py)

**Files:**
- Create: `app/services/llm.py`

- [ ] **Step 1: Write llm.py**

```python
from openai import AsyncOpenAI

from app.utils.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


def get_model() -> str:
    return OPENAI_MODEL
```

- [ ] **Step 2: Verify import**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -c "from app.services.llm import get_client, get_model; print(f'Model: {get_model()}', type(get_client()))"
```
Expected: prints model name and `<class 'openai.AsyncOpenAI'>`.

- [ ] **Step 3: Commit**

```bash
git add app/services/llm.py && git commit -m "feat: add LLM client service (AsyncOpenAI wrapper)"
```

---

### Task 3: Embedding Service (services/embedding.py)

**Files:**
- Create: `app/services/embedding.py`

- [ ] **Step 1: Write embedding.py**

```python
from openai import AsyncOpenAI

from app.utils.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
)

_client: AsyncOpenAI | None = None


def _get_embedding_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text strings.

    Returns a list of embedding vectors (list of floats).
    """
    client = _get_embedding_client()
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def embed_single(text: str) -> list[float]:
    """Generate embedding for a single text string."""
    results = await embed_texts([text])
    return results[0]
```

- [ ] **Step 2: Verify import (no API call needed)**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -c "from app.services.embedding import embed_texts; print('embedding service imported OK')"
```
Expected: prints "embedding service imported OK".

- [ ] **Step 3: Commit**

```bash
git add app/services/embedding.py && git commit -m "feat: add embedding service with configurable provider"
```

---

### Task 4: Document Parser Service (services/parser.py)

**Files:**
- Create: `app/services/parser.py`

- [ ] **Step 1: Write parser.py**

```python
import io
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


class ParseError(Exception):
    """Raised when document parsing fails."""
    pass


class UnsupportedFileTypeError(ParseError):
    """Raised when the file type is not supported."""
    pass


class FileTooLargeError(ParseError):
    """Raised when the file exceeds the size limit."""
    pass


def validate_file(filename: str, content: bytes) -> Path:
    """Validate file type and size. Returns the file extension."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"不支持的文件类型: {ext}")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"文件大小超过限制 ({MAX_FILE_SIZE_BYTES // (1024*1024)}MB)"
        )
    return Path(filename)


def parse_pdf(content: bytes) -> str:
    """Extract text from a PDF file using pdfplumber."""
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def parse_markdown(content: bytes) -> str:
    """Return markdown text as-is (it's already plain text)."""
    return content.decode("utf-8", errors="replace")


def parse_txt(content: bytes) -> str:
    """Return plain text as-is."""
    return content.decode("utf-8", errors="replace")


async def parse_document(filename: str, content: bytes) -> str:
    """Parse a document by filename and return extracted text.

    Raises ParseError for unsupported types or parse failures.
    """
    file_path = validate_file(filename, content)
    ext = file_path.suffix.lower()

    parsers = {
        ".pdf": parse_pdf,
        ".md": parse_markdown,
        ".txt": parse_txt,
    }
    parser = parsers.get(ext)
    if parser is None:
        raise UnsupportedFileTypeError(f"不支持的文件类型: {ext}")

    try:
        text = parser(content)
    except Exception as e:
        raise ParseError(f"文档解析失败: {str(e)}") from e

    if not text or not text.strip():
        raise ParseError("文档内容为空，无法提取文本")

    return text
```

- [ ] **Step 2: Quick smoke test**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -c "
import asyncio
from app.services.parser import parse_document
text = asyncio.run(parse_document('test.md', '# Hello\\n\\nWorld'.encode()))
print(repr(text))
"
```
Expected: prints `'# Hello\n\nWorld'`.

- [ ] **Step 3: Commit**

```bash
git add app/services/parser.py && git commit -m "feat: add document parser service (PDF/MD/TXT)"
```

---

### Task 5: Hybrid Chunker Service (services/chunker.py)

**Files:**
- Create: `app/services/chunker.py`
- Test: `tests/test_chunker.py` (Task 5a)

- [ ] **Step 1: Write the failing test (tests/test_chunker.py)**

```python
import pytest
from app.services.chunker import chunk_text, Chunk

# Use small token limits for testing
TEST_MAX_TOKENS = 50
TEST_OVERLAP = 10


class TestChunkText:
    """Tests for hybrid chunking strategy."""

    def test_short_text_single_chunk(self):
        """Short text under token limit returns one chunk."""
        text = "This is a short sentence."
        chunks = chunk_text(text, max_tokens=TEST_MAX_TOKENS, overlap_tokens=TEST_OVERLAP)
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].content == text
        assert chunks[0].index == 0

    def test_markdown_heading_split(self):
        """Markdown text splits on ## headings."""
        text = "## Section One\nContent for section one.\n\n## Section Two\nContent for section two."
        chunks = chunk_text(text, max_tokens=500, overlap_tokens=50)
        assert len(chunks) >= 2
        assert any("Section One" in c.content for c in chunks)
        assert any("Section Two" in c.content for c in chunks)

    def test_paragraph_split(self):
        """Plain text splits on double newlines."""
        text = "First paragraph here.\n\nSecond paragraph is separate.\n\nThird one too."
        chunks = chunk_text(text, max_tokens=500, overlap_tokens=50)
        assert len(chunks) >= 3

    def test_long_paragraph_splits_by_sentence(self):
        """A paragraph exceeding max_tokens splits on sentence boundaries."""
        # Build a long paragraph of repeated sentences
        sentence = "This is sentence number {} with enough words to fill space. "
        text = "".join(sentence.format(i) for i in range(30))
        chunks = chunk_text(text, max_tokens=TEST_MAX_TOKENS, overlap_tokens=5)
        assert len(chunks) > 1
        for c in chunks:
            assert c.index >= 0

    def test_chunk_overlap(self):
        """Adjacent chunks share overlap content."""
        text = ("A" * 20 + " ") * 30  # long text that forces multiple chunks
        chunks = chunk_text(text, max_tokens=50, overlap_tokens=20)
        if len(chunks) >= 2:
            # last N chars of chunk 0 should appear at start of chunk 1
            overlap_window = chunks[0].content[-30:]
            assert any(overlap_window[:20] in chunks[1].content for _ in [1])

    def test_empty_text(self):
        """Empty text returns empty list."""
        chunks = chunk_text("", max_tokens=500, overlap_tokens=50)
        assert chunks == []

    def test_chunk_metadata_present(self):
        """Each chunk has required fields."""
        text = "Hello world content."
        chunks = chunk_text(text, max_tokens=500, overlap_tokens=50)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.content == text
        assert c.index == 0
        assert c.token_count > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -m pytest tests/test_chunker.py -v
```
Expected: FAIL with ModuleNotFoundError (chunker.py not written yet).

- [ ] **Step 3: Write chunker.py**

```python
import re
from dataclasses import dataclass, field
from typing import List

import tiktoken

from app.utils.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS

# Use cl100k_base encoding (compatible with text-embedding-3-small)
_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    content: str
    index: int
    token_count: int = 0

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = len(_encoding.encode(self.content))


def _count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _split_by_sentences(text: str, max_tokens: int) -> list[str]:
    """Split text on sentence boundaries, respecting max_tokens per chunk."""
    # Split on Chinese/English sentence endings
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append("".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        chunks.append("".join(current))

    return chunks


def _split_by_headings(text: str) -> list[str]:
    """Split markdown text by ## / ### headings."""
    # Match lines starting with ## or ### (but not # which is doc title)
    parts = re.split(r'\n(?=#{2,3}\s)', text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_paragraphs(text: str) -> list[str]:
    """Split text by double newlines (paragraph boundaries)."""
    parts = re.split(r'\n\s*\n', text)
    return [p.strip() for p in parts if p.strip()]


def _is_markdown(text: str) -> bool:
    """Heuristic: detect if text looks like markdown (has ## headings)."""
    return bool(re.search(r'^#{2,3}\s', text, re.MULTILINE))


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[Chunk]:
    """
    Hybrid chunking strategy:
    1. Semantic boundary split: MD by headings, TXT by paragraphs
    2. If a segment exceeds max_tokens, split by sentences
    3. Add overlap between adjacent chunks
    """
    if not text or not text.strip():
        return []

    # Phase 1: Semantic boundary splitting
    if _is_markdown(text):
        segments = _split_by_headings(text)
    else:
        segments = _split_by_paragraphs(text)

    # Phase 2: Token-aware secondary splitting
    raw_chunks: list[str] = []
    for seg in segments:
        if _count_tokens(seg) <= max_tokens:
            raw_chunks.append(seg)
        else:
            raw_chunks.extend(_split_by_sentences(seg, max_tokens))

    # Phase 3: Add overlap
    final_chunks: list[str] = []
    for i, raw in enumerate(raw_chunks):
        if i > 0 and overlap_tokens > 0:
            # Prepend overlap from previous chunk's end
            prev = raw_chunks[i - 1]
            prev_tokens = _encoding.encode(prev)
            overlap_slice = prev_tokens[-overlap_tokens:]
            overlap_text = _encoding.decode(overlap_slice)
            raw = overlap_text + raw
        final_chunks.append(raw)

    return [
        Chunk(content=c, index=i)
        for i, c in enumerate(final_chunks)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -m pytest tests/test_chunker.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/chunker.py tests/test_chunker.py && git commit -m "feat: add hybrid chunker with sentence/heading/paragraph splitting + overlap"
```

---

### Task 6: Vector Store Service (services/vector_store.py)

**Files:**
- Create: `app/services/vector_store.py`

- [ ] **Step 1: Write vector_store.py**

```python
import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.api.types import IncludeEnum

from app.services.embedding import embed_texts
from app.utils.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


class VectorStoreError(Exception):
    """Raised when a ChromaDB operation fails."""
    pass


class VectorStore:
    """Encapsulates ChromaDB operations for the knowledge base."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def add_document(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        chunks: list[dict],  # [{"content": str, "index": int}, ...]
    ) -> int:
        """Add chunks of a document to ChromaDB. Returns chunk count."""
        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        embeddings = await embed_texts(texts)

        ids = [f"{doc_id}_{c['index']}" for c in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "chunk_index": c["index"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for c in chunks
        ]

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        except Exception as e:
            raise VectorStoreError(f"向量存储失败: {str(e)}") from e

        return len(chunks)

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search for chunks relevant to the query."""
        from app.services.embedding import embed_single

        if self.collection.count() == 0:
            return []

        query_embedding = await embed_single(query)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=[IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances],
            )
        except Exception as e:
            raise VectorStoreError(f"向量检索失败: {str(e)}") from e

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Convert cosine distance to similarity score (1 - distance)
                score = 1.0 - distance
                hits.append({
                    "chunk_id": chunk_id,
                    "doc_id": metadata.get("doc_id", ""),
                    "filename": metadata.get("filename", ""),
                    "file_type": metadata.get("file_type", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "content": document,
                    "content_preview": document[:200] if document else "",
                    "score": round(score, 4),
                })

        return hits

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns number of chunks removed."""
        try:
            # Find all chunks for this doc
            existing = self.collection.get(
                where={"doc_id": doc_id},
                include=[IncludeEnum.metadatas],
            )
            count = len(existing["ids"]) if existing["ids"] else 0

            if count > 0:
                self.collection.delete(where={"doc_id": doc_id})

            return count
        except Exception as e:
            raise VectorStoreError(f"文档删除失败: {str(e)}") from e

    def list_documents(self) -> list[dict]:
        """List all unique documents with chunk counts."""
        try:
            all_data = self.collection.get(include=[IncludeEnum.metadatas])
        except Exception as e:
            raise VectorStoreError(f"获取文档列表失败: {str(e)}") from e

        if not all_data["metadatas"]:
            return []

        # Aggregate by doc_id
        doc_map: dict[str, dict] = {}
        for meta in all_data["metadatas"]:
            did = meta["doc_id"]
            if did not in doc_map:
                doc_map[did] = {
                    "doc_id": did,
                    "filename": meta.get("filename", ""),
                    "file_type": meta.get("file_type", ""),
                    "chunk_count": 0,
                    "created_at": meta.get("created_at", ""),
                }
            doc_map[did]["chunk_count"] += 1

        return list(doc_map.values())

    def doc_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the store."""
        existing = self.collection.get(
            where={"doc_id": doc_id},
            limit=1,
        )
        return len(existing["ids"]) > 0 if existing["ids"] else False

    def count(self) -> int:
        """Return total number of chunks."""
        return self.collection.count()


# Singleton instance
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
```

- [ ] **Step 2: Verify import**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -c "from app.services.vector_store import get_vector_store; print('vector_store imported OK')"
```
Expected: prints "vector_store imported OK".

- [ ] **Step 3: Commit**

```bash
git add app/services/vector_store.py && git commit -m "feat: add ChromaDB vector store service with CRUD + search"
```

---

### Task 7: RAG Pipeline Service (services/rag_pipeline.py)

**Files:**
- Create: `app/services/rag_pipeline.py`

- [ ] **Step 1: Write rag_pipeline.py**

```python
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
```

- [ ] **Step 2: Verify import**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -c "from app.services.rag_pipeline import rag_pipeline, retrieve_chunks; print('rag_pipeline imported OK')"
```
Expected: prints "rag_pipeline imported OK".

- [ ] **Step 3: Commit**

```bash
git add app/services/rag_pipeline.py && git commit -m "feat: add RAG pipeline service (retrieve → generate with citations)"
```

---

### Task 8: SSE Streaming Service (services/streaming.py)

**Files:**
- Create: `app/services/streaming.py`

- [ ] **Step 1: Write streaming.py**

```python
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
    """Generate SSE events for the /qa/ask endpoint.

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
```

- [ ] **Step 2: Verify import**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -c "from app.services.streaming import rag_ask_sse, rag_compare_sse; print('streaming imported OK')"
```
Expected: prints "streaming imported OK".

- [ ] **Step 3: Commit**

```bash
git add app/services/streaming.py && git commit -m "feat: add SSE streaming service with structured event types"
```

---

### Task 9: Schemas (schemas/document.py, schemas/qa.py)

**Files:**
- Create: `app/schemas/document.py`
- Create: `app/schemas/qa.py`

- [ ] **Step 1: Write schemas/document.py**

```python
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    status: str = "indexed"


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    size_bytes: int = 0
    created_at: str = ""


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DocumentDeleteResponse(BaseModel):
    doc_id: str
    deleted: bool
    chunks_removed: int


class ErrorResponse(BaseModel):
    detail: str
```

- [ ] **Step 2: Write schemas/qa.py**

```python
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000, description="用户自然语言问题")
    top_k: int = Field(default=5, ge=1, le=50, description="检索的chunk数量")


class RetrievedChunkPreview(BaseModel):
    doc_id: str
    filename: str
    content_preview: str
    score: float


class CitationInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    content_snippet: str


class AskResponse(BaseModel):
    """Non-streaming response for testing purposes."""
    answer: str
    citations: list[CitationInfo]
    retrieved_chunks: list[RetrievedChunkPreview]


class CompareResponse(BaseModel):
    """Non-streaming comparison response for testing."""
    rag_answer: str
    bare_answer: str
    rag_citations: list[CitationInfo]
```

- [ ] **Step 3: Verify imports**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -c "
from app.schemas.document import DocumentUploadResponse, DocumentInfo, DocumentDeleteResponse
from app.schemas.qa import QuestionRequest, AskResponse, CompareResponse
print('All schemas imported OK')
"
```
Expected: prints "All schemas imported OK".

- [ ] **Step 4: Commit**

```bash
git add app/schemas/document.py app/schemas/qa.py && git commit -m "feat: add Pydantic v2 schemas for documents and QA"
```

---

### Task 10: Documents Router (routers/documents.py)

**Files:**
- Create: `app/routers/documents.py`

- [ ] **Step 1: Write routers/documents.py**

```python
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.document import (
    DocumentUploadResponse,
    DocumentInfo,
    DocumentDeleteResponse,
    ErrorResponse,
)
from app.services.parser import parse_document, UnsupportedFileTypeError, FileTooLargeError, ParseError
from app.services.chunker import chunk_text
from app.services.vector_store import get_vector_store, VectorStoreError

router = APIRouter(tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse, "description": "不支持的文件类型或解析失败"},
        413: {"model": ErrorResponse, "description": "文件超过大小限制"},
    },
)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF/MD/TXT), parse, chunk, embed, and store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # Read file content
    content = await file.read()

    # Parse document
    try:
        text = await parse_document(file.filename, content)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except ParseError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Chunk
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="文档解析后无有效内容")

    # Generate doc_id and determine file_type
    doc_id = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "txt"
    file_type = ext if ext in ("pdf", "md", "txt") else "txt"

    # Store in ChromaDB
    chunk_dicts = [{"content": c.content, "index": c.index} for c in chunks]
    store = get_vector_store()

    try:
        await store.add_document(
            doc_id=doc_id,
            filename=file.filename,
            file_type=file_type,
            chunks=chunk_dicts,
        )
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        file_type=file_type,
        chunk_count=len(chunks),
        status="indexed",
    )


@router.get("", response_model=list[DocumentInfo])
async def list_documents():
    """List all indexed documents with chunk counts."""
    store = get_vector_store()
    try:
        docs = store.list_documents()
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [
        DocumentInfo(
            doc_id=d["doc_id"],
            filename=d["filename"],
            file_type=d["file_type"],
            chunk_count=d["chunk_count"],
            created_at=d.get("created_at", ""),
        )
        for d in docs
    ]


@router.delete(
    "/{doc_id}",
    response_model=DocumentDeleteResponse,
    responses={404: {"model": ErrorResponse, "description": "文档不存在"}},
)
async def delete_document(doc_id: str):
    """Delete a document and all its chunks."""
    store = get_vector_store()

    if not store.doc_exists(doc_id):
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")

    try:
        chunks_removed = store.delete_document(doc_id)
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DocumentDeleteResponse(
        doc_id=doc_id,
        deleted=True,
        chunks_removed=chunks_removed,
    )
```

- [ ] **Step 2: Verify router loads**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -c "from app.routers.documents import router; print('documents router loaded OK, routes:', [r.path for r in router.routes])"
```
Expected: prints route paths including `/upload`, `/`, `/{doc_id}`.

- [ ] **Step 3: Commit**

```bash
git add app/routers/documents.py && git commit -m "feat: add documents router (upload/list/delete)"
```

---

### Task 11: QA Router (routers/qa.py)

**Files:**
- Create: `app/routers/qa.py`

- [ ] **Step 1: Write routers/qa.py**

```python
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
    _extract_citations,
    _build_context,
)
from app.services.llm import get_client, get_model
from app.services.streaming import rag_ask_sse, rag_compare_sse

router = APIRouter(tags=["qa"])


@router.post("/ask")
async def ask_question(request: QuestionRequest):
    """RAG-enhanced QA with SSE streaming output."""
    # Retrieve relevant chunks
    retrieved = await retrieve_chunks(question=request.question, top_k=request.top_k)

    # Build answer stream
    answer_stream = generate_answer_stream(request.question, retrieved)

    # Build citations
    citations = _extract_citations(retrieved)

    # Return SSE stream
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

    citations = _extract_citations(retrieved)

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
    # RAG side
    retrieved = await retrieve_chunks(question=request.question, top_k=request.top_k)
    rag_stream = generate_answer_stream(request.question, retrieved)
    citations = _extract_citations(retrieved)

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

    citations = _extract_citations(retrieved)

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
```

- [ ] **Step 2: Verify router loads**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -c "from app.routers.qa import router; print('qa router loaded OK, routes:', [r.path for r in router.routes])"
```
Expected: prints route paths including `/ask`, `/ask/sync`, `/compare`, `/compare/sync`.

- [ ] **Step 3: Commit**

```bash
git add app/routers/qa.py && git commit -m "feat: add QA router with SSE streaming (ask, compare) and sync test variants"
```

---

### Task 12: Main Application (main.py)

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write main.py**

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import documents, qa

app = FastAPI(
    title="个人知识库智能问答系统",
    description="RAG-based Personal Knowledge Base QA System — upload documents, ask questions, get cited answers",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(documents.router, prefix="/api/v1/documents")
app.include_router(qa.router, prefix="/api/v1/qa")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "个人知识库智能问答系统",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "ok", "service": "RAG Knowledge Base QA", "version": "0.1.0"}


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Not Found",
            "message": "请求的资源不存在。可用接口请查看 /docs",
            "available_endpoints": {
                "swagger_docs": "/docs",
                "openapi_json": "/openapi.json",
                "health_check": "/health",
                "upload_doc": "POST /api/v1/documents/upload",
                "list_docs": "GET /api/v1/documents",
                "delete_doc": "DELETE /api/v1/documents/{doc_id}",
                "ask_qa": "POST /api/v1/qa/ask",
                "compare_qa": "POST /api/v1/qa/compare",
            },
        },
    )
```

- [ ] **Step 2: Verify app starts**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -c "from main import app; print(f'App: {app.title}, routes: {len(app.routes)}')"
```
Expected: prints "App: 个人知识库智能问答系统, routes: N" (N > 5).

- [ ] **Step 3: Commit**

```bash
git add main.py && git commit -m "feat: add FastAPI main application with router mounting and health check"
```

---

### Task 13: Test Fixtures (tests/conftest.py)

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write conftest.py**

```python
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture(autouse=True)
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


@pytest.fixture
async def test_app():
    """Create a TestClient for the FastAPI app."""
    from main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
```

- [ ] **Step 2: Verify fixtures load**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -m pytest tests/conftest.py --collect-only
```
Expected: no collection errors.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py && git commit -m "test: add pytest fixtures with isolated ChromaDB and sample data"
```

---

### Task 14: Document API Integration Tests (tests/test_documents.py)

**Files:**
- Create: `tests/test_documents.py`

- [ ] **Step 1: Write test_documents.py**

```python
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_and_list(test_app: AsyncClient, sample_txt_content: bytes):
    """Upload a TXT file, verify response, then list documents."""
    # Upload
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("test_notes.txt", sample_txt_content, "text/plain")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "indexed"
    assert data["filename"] == "test_notes.txt"
    assert data["file_type"] == "txt"
    assert data["chunk_count"] > 0
    doc_id = data["doc_id"]
    assert len(doc_id) == 32  # uuid hex

    # List
    resp = await test_app.get("/api/v1/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list)
    assert any(d["doc_id"] == doc_id for d in docs)
    found = next(d for d in docs if d["doc_id"] == doc_id)
    assert found["filename"] == "test_notes.txt"
    assert found["chunk_count"] == data["chunk_count"]


@pytest.mark.asyncio
async def test_delete_document(test_app: AsyncClient, sample_txt_content: bytes):
    """Upload a document, delete it, verify it's gone."""
    # Upload
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("to_delete.txt", sample_txt_content, "text/plain")},
    )
    doc_id = resp.json()["doc_id"]

    # Delete
    resp = await test_app.delete(f"/api/v1/documents/{doc_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True
    assert data["chunks_removed"] > 0

    # Verify not in list
    resp = await test_app.get("/api/v1/documents")
    docs = resp.json()
    assert not any(d["doc_id"] == doc_id for d in docs)


@pytest.mark.asyncio
async def test_unsupported_file_type(test_app: AsyncClient):
    """Upload an unsupported file type should return 400."""
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("test.exe", b"binary content", "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "不支持的文件类型" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_delete_nonexistent_document(test_app: AsyncClient):
    """Delete a non-existent document should return 404."""
    resp = await test_app.delete("/api/v1/documents/nonexistent123")
    assert resp.status_code == 404
    assert "不存在" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_upload_markdown(test_app: AsyncClient, sample_md_content: bytes):
    """Upload a Markdown file and verify it chunks by headings."""
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("test_doc.md", sample_md_content, "text/markdown")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["file_type"] == "md"
    assert data["chunk_count"] >= 2  # Should split on ## headings
```

- [ ] **Step 2: Run tests**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -m pytest tests/test_documents.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_documents.py && git commit -m "test: add document API integration tests (upload, list, delete, errors)"
```

---

### Task 15: QA API Integration Tests (tests/test_qa.py)

**Files:**
- Create: `tests/test_qa.py`

**Note:** These tests mock the LLM/Embedding calls since we don't have real API keys in test. We use `unittest.mock` to patch OpenAI calls.

- [ ] **Step 1: Write test_qa.py**

```python
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from httpx import AsyncClient


# Mock embedding: return a fixed 1024-dim vector
MOCK_EMBEDDING = [0.1] * 1024


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
    # Should contain the answer or at least some response
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
    # No documents uploaded — just ask
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
```

- [ ] **Step 2: Run tests**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -m pytest tests/test_qa.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_qa.py && git commit -m "test: add QA API integration tests with mocked LLM/embedding"
```

---

### Task 16: Final Verification

**Files:**
- None (verification only)

- [ ] **Step 1: Run all tests**

```bash
cd C:/Users/35452/Desktop/week2task && python -m pytest tests/ -v
```
Expected: ALL tests PASS (7 chunker + 5 documents + 4 qa = 16 tests).

- [ ] **Step 2: Verify app starts clean**

```bash
cd C:/Users/35452/Desktop/week2task && python -c "
from main import app
routes = [(r.path, list(r.methods)) for r in app.routes if hasattr(r, 'methods')]
for path, methods in sorted(routes):
    print(f'{path} {methods}')
"
```
Expected: prints all endpoints:
```
/api/v1/documents {GET}
/api/v1/documents/{doc_id} {DELETE}
/api/v1/documents/upload {POST}
/api/v1/qa/ask {POST}
/api/v1/qa/ask/sync {POST}
/api/v1/qa/compare {POST}
/api/v1/qa/compare/sync {POST}
/ {GET}
/health {GET}
```

- [ ] **Step 3: Verify OpenAPI schema validity**

Run:
```bash
cd C:/Users/35452/Desktop/week2task && python -c "
from main import app
schema = app.openapi()
print(f'OpenAPI version: {schema[\"openapi\"]}')
print(f'Paths: {list(schema[\"paths\"].keys())}')
"
```
Expected: prints OpenAPI paths.

- [ ] **Step 4: Commit any final changes**

```bash
git status
git add -A
git diff --cached --stat
```

If clean:
```bash
git commit --allow-empty -m "chore: final verification - all tests pass, API routes confirmed"
```

---

## Spec Coverage Checklist

| Spec Section | Covered By |
|---|---|
| 1.1 文档上传 | Task 10 (router) + Task 4 (parser) + Task 5 (chunker) + Task 6 (vector_store) |
| 1.1 知识库管理 | Task 10 (list/delete endpoints) |
| 1.1 智能问答 SSE | Task 11 (qa router) + Task 7 (pipeline) + Task 8 (streaming) |
| 1.1 RAG vs 无RAG对比 | Task 11 (/compare endpoints) + Task 7 (bare answer) |
| 2 目录结构 | Task 0 (scaffolding) |
| 3 API 端点 | Task 10, 11, 12 (main.py routing) |
| 4 服务层 | Task 1-8 (all services) |
| 5 配置管理 | Task 1 (config.py) |
| 6 错误处理 | Task 10, 11 (HTTPException in routers) |
| 7 测试 | Task 5a, 13, 14, 15 |
| 9 依赖 | Task 0 (requirements.txt) |
