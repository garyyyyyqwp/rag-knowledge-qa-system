# RAG Knowledge QA System

A Retrieval-Augmented Generation (RAG) powered Q&A system that answers questions based on uploaded documents (PDF, DOCX, TXT). Supports both RAG (with knowledge base retrieval) and bare LLM modes, with real-time streaming output and a compare mode to evaluate responses side by side.

## Features

- **Document Management** — Upload, list, and delete PDF/DOCX/TXT documents
- **Dual QA Modes**:
  - **RAG Mode** — Retrieves relevant chunks from your knowledge base before answering
  - **LLM Only Mode** — Pure model response without retrieval
  - **Compare Mode** — Side-by-side comparison of RAG vs bare LLM answers
- **SSE Streaming** — Real-time token-by-token streaming for responsive output
- **Smart Citations** — Sources are automatically filtered to show only chunks referenced in the answer
- **Per-Mode History** — Each mode maintains its own independent conversation history
- **Dark Theme** — Clean, modern dark UI

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| LLM | Zhipu AI (GLM-4-flash) |
| Embeddings | Zhipu AI (embedding-2) |
| Vector Store | ChromaDB |
| Document Parsing | PyMuPDF (PDF), python-docx (DOCX) |
| Frontend | Vanilla JS, CSS (dark theme) |
| Chunking | Recursive character text splitter (tiktoken) |

## Getting Started

### Prerequisites

- Python 3.11+
- A [Zhipu AI API key](https://open.bigmodel.cn/)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/garyyyyyqwp/rag-knowledge-qa-system.git
cd rag-knowledge-qa-system
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

Create a `.env` file (or copy `.env.example`):

```env
# LLM (Zhipu GLM)
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
OPENAI_API_KEY=your_zhipu_api_key_here
OPENAI_MODEL=glm-4-flash

# Embedding
EMBEDDING_MODEL=embedding-2

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_data_v2
CHROMA_COLLECTION_NAME=knowledge_base_v2
```

4. Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

5. Open `http://localhost:8000` in your browser.

## API Endpoints

### Documents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/documents/upload` | Upload a document (PDF/DOCX/TXT) |
| GET | `/api/v1/documents` | List all documents |
| DELETE | `/api/v1/documents/{id}` | Delete a document |

### Q&A

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/qa/ask` | Ask a question (SSE streaming) |
| POST | `/api/v1/qa/compare` | Compare RAG vs bare LLM (SSE streaming) |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |

## Deployment

### Render (Blueprint)

The project includes a `render.yaml` for one-click deployment on Render:

1. Push the repo to GitHub
2. In Render Dashboard, click **New → Blueprint**
3. Connect your GitHub repository
4. Set the `ZHIPUAI_API_KEY` environment variable in Render dashboard
5. Deploy

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | Zhipu AI API key |
| `OPENAI_API_BASE` | No | `https://open.bigmodel.cn/api/paas/v4` | API base URL |
| `OPENAI_MODEL` | No | `glm-4-flash` | LLM model |
| `EMBEDDING_MODEL` | No | `embedding-2` | Embedding model |
| `CHROMA_PERSIST_DIR` | No | `./chroma_data_v2` | ChromaDB persistence path |

## Project Structure

```
rag-knowledge-qa-system/
├── app/
│   ├── routers/          # FastAPI route handlers
│   │   ├── documents.py
│   │   └── qa.py
│   ├── schemas/          # Pydantic models
│   │   ├── document.py
│   │   └── qa.py
│   ├── services/         # Business logic
│   │   ├── chunker.py    # Text chunking
│   │   ├── embedding.py  # Vector embedding
│   │   ├── llm.py        # LLM interaction
│   │   ├── parser.py     # Document parsing
│   │   ├── rag_pipeline.py
│   │   ├── streaming.py  # SSE event formatting
│   │   └── vector_store.py
│   └── utils/
│       └── config.py
├── static/               # Frontend
│   ├── css/style.css
│   ├── js/app.js
│   └── index.html
├── tests/
├── main.py               # App entry point
├── render.yaml            # Render deploy config
└── requirements.txt
```

## License

MIT
