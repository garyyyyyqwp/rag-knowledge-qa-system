# 个人知识库智能问答系统 — 设计规范

> 版本: v1.0 | 日期: 2026-06-09 | 状态: Ready for Review

## 1. 概述

构建一个基于 RAG (Retrieval-Augmented Generation) 的个人知识库智能问答系统。用户上传学习笔记、技术文档、论文等，系统将其向量化存储于 ChromaDB，之后可用自然语言提问，系统检索最相关内容片段，交由 LLM 生成带引用来源的准确回答。

### 1.1 核心功能

1. **文档上传与处理**: POST 接口，支持 PDF / Markdown / TXT，自动解析 → 混合分块 → Embedding → 存入 ChromaDB
2. **知识库管理**: GET 列出已入库文档及 Chunk 统计；DELETE 删除指定文档
3. **智能问答**: POST 接口，Embed → Retrieve → Generate 管道，SSE 流式输出 LLM 答案 + 引用来源
4. **RAG vs 无 RAG 对比**: 独立端点，对比"纯 LLM 回答"和"RAG 增强回答"

### 1.2 关键设计决策

| 决策项 | 选择 | 理由 |
|---|---|---|
| 分块策略 | 混合策略（C） | 语义边界优先 + token 二次切分 + 重叠，适配多文档格式 |
| LLM 模型 | 可配置（C） | 通过 .env 切换，默认 GPT-4o-mini，兼容任意 OpenAI 接口 |
| Embedding | 可配置（C） | 默认 text-embedding-3-small，支持 DeepSeek 切换 |
| ChromaDB 模式 | 本地持久化（B） | PersistentClient，路径可配置，重启保留数据 |
| SSE 格式 | 结构化事件（B） | retrieval / answer / citations / done 四种事件类型 |

---

## 2. 项目目录结构

```
week2task/
├── .env.example
├── .gitignore
├── requirements.txt
├── main.py                          # FastAPI 应用入口
├── app/
│   ├── __init__.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── documents.py             # 文档上传/管理路由
│   │   └── qa.py                    # 智能问答 RAG 路由
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── document.py              # 文档相关 Pydantic 模型
│   │   └── qa.py                    # 问答相关 Pydantic 模型
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm.py                   # 复用 Week1 OpenAI 封装 (AsyncOpenAI)
│   │   ├── embedding.py             # Embedding 服务（可配置 provider）
│   │   ├── chunker.py               # 混合分块策略
│   │   ├── parser.py                # 文档解析（PDF/MD/TXT → 纯文本）
│   │   ├── vector_store.py          # ChromaDB 操作封装
│   │   ├── rag_pipeline.py          # Embed → Retrieve → Generate 管道
│   │   └── streaming.py             # SSE 流式事件生成
│   └── utils/
│       ├── __init__.py
│       └── config.py                # 环境变量配置管理
├── tests/
│   ├── __init__.py
│   ├── test_documents.py
│   ├── test_qa.py
│   └── test_chunker.py
└── chroma_data/                     # ChromaDB 持久化目录（自动生成，gitignore）
```

---

## 3. API 端点设计

### 3.1 文档上传

```
POST /api/v1/documents/upload
Content-Type: multipart/form-data
```

**输入:** `file` (PDF/MD/TXT file)

**处理流程:** 解析文本 → 混合分块 → 生成 Embedding → 存入 ChromaDB

**返回 (201):**
```json
{
  "doc_id": "uuid",
  "filename": "深度学习笔记.md",
  "file_type": "md",
  "chunk_count": 12,
  "status": "indexed"
}
```

**错误 (400):** 不支持的文件类型

### 3.2 知识库文档列表

```
GET /api/v1/documents
```

**返回 (200):**
```json
[
  {
    "doc_id": "uuid",
    "filename": "深度学习笔记.md",
    "file_type": "md",
    "chunk_count": 12,
    "size_bytes": 45000,
    "created_at": "2026-06-09T10:30:00Z"
  }
]
```

### 3.3 删除文档

```
DELETE /api/v1/documents/{doc_id}
```

**返回 (200):**
```json
{
  "doc_id": "uuid",
  "deleted": true,
  "chunks_removed": 12
}
```

**错误 (404):** 文档不存在

### 3.4 RAG 问答（SSE 流式）

```
POST /api/v1/qa/ask
Content-Type: application/json
Accept: text/event-stream
```

**输入:**
```json
{
  "question": "反向传播算法的原理是什么？",
  "top_k": 5
}
```

**SSE 事件序列:**

```
event: retrieval
data: {"chunks":[{"doc_id":"u1","filename":"DL笔记.md","content_preview":"反向传播算法...","score":0.92},{"doc_id":"u2","filename":"论文.pdf","content_preview":"...","score":0.87}]}

event: answer
data: 反向传播

event: answer
data: 算法通过链式法则...

event: citations
data: [{"doc_id":"u1","filename":"DL笔记.md","chunk_index":3,"content_snippet":"反向传播利用链式法则..."}]

event: done
data: {}
```

### 3.5 RAG vs 无 RAG 对比（SSE 流式）

```
POST /api/v1/qa/compare
Content-Type: application/json
Accept: text/event-stream
```

**输入:**
```json
{
  "question": "什么是Transformer？",
  "top_k": 5
}
```

**SSE 事件序列:**

```
event: rag_answer
data: Transformer 架构基于自注意力...

event: bare_answer
data: Transformer 是一种深度学习架构...

event: rag_citations
data: [{"doc_id":"u1","filename":"Transformer论文.pdf"}]

event: done
data: {}
```

> rag_answer 和 bare_answer 事件交错发送（分别来自两个并发 LLM 调用），前端可按需展示。

---

## 4. 服务层设计

### 4.1 服务职责表

| 服务 | 文件 | 职责 | 依赖 |
|---|---|---|---|
| 文档解析 | `parser.py` | PDF(pdfplumber)/MD/TXT → 纯文本 | 无 |
| 混合分块 | `chunker.py` | 文本 → List[Chunk] | 无 |
| LLM 客户端 | `llm.py` | AsyncOpenAI 封装 | `config.py` |
| Embedding | `embedding.py` | 文本 → 向量 (OpenAI/DeepSeek) | `llm.py` |
| 向量存储 | `vector_store.py` | ChromaDB CRUD 操作 | `embedding.py` |
| RAG 管道 | `rag_pipeline.py` | 编排检索→生成完整流程 | `embedding.py`, `vector_store.py`, `llm.py` |
| 流式输出 | `streaming.py` | SSE 事件格式化 | `rag_pipeline.py` |

### 4.2 混合分块策略

1. **语义边界切分:**
   - MD: 按 `##` / `###` 标题边界切
   - TXT: 按双换行 `\n\n`（段落边界）切
   - PDF: 提取文本后按段落边界切
2. **Token 二次切分:** 每个语义块若超过 `CHUNK_MAX_TOKENS`(512)，按句子边界二次切分
3. **重叠:** 相邻 chunk 保留 `CHUNK_OVERLAP_TOKENS`(50) token 重叠

### 4.3 Prompt 模板

```
你是一个知识库问答助手。基于以下参考资料回答用户问题。如果资料不足以回答，请如实说明你无法从资料中找到相关信息。

参考资料：
[1] (来源: {filename})
{chunk_content}

[2] (来源: {filename})
{chunk_content}

...

用户问题：{question}

要求：
1. 回答准确、简洁
2. 引用资料内容时标注来源编号，如[1]、[2]
3. 如果资料不足以回答，明确说明
```

### 4.4 ChromaDB 数据模型

**Collection:** `knowledge_base` (或 .env 可配置)

**Metadata per chunk:**
```json
{
  "doc_id": "uuid",
  "filename": "DL笔记.md",
  "file_type": "md",
  "chunk_index": 3,
  "content_hash": "sha256...",
  "created_at": "2026-06-09T10:30:00Z"
}
```

### 4.5 文档元数据管理

由于 ChromaDB 不直接支持按 doc_id 列文档，需要维护一个轻量级元数据层：

- `vector_store.py` 内部使用 ChromaDB 的 metadata filter (`where={"doc_id": doc_id}`) 实现按文档删除和统计
- 文档列表通过聚合 ChromaDB metadata 实现

---

## 5. 配置管理

### 5.1 环境变量 (.env)

```env
# LLM
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

# Embedding
EMBEDDING_PROVIDER=openai              # openai | deepseek
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-xxx               # 默认与 OPENAI_API_KEY 相同
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

### 5.2 config.py 设计

在 Week 1 `config.py` 基础上扩展，每个变量通过 `get_env()` 读取，有合理的默认值。Embedding 相关变量可选（未设时 fallback 到 LLM 对应变量）。

---

## 6. 错误处理

| 场景 | HTTP 状态码 | 响应 |
|---|---|---|
| 不支持的文件类型 | 400 | `{ "detail": "不支持的文件类型: .exe" }` |
| 文件超出大小限制 | 413 | `{ "detail": "文件大小超过限制 (10MB)" }` |
| 文档不存在 | 404 | `{ "detail": "文档不存在: xxx" }` |
| LLM/Embedding API 错误 | 502 | `{ "detail": "LLM服务暂时不可用，请稍后重试" }` |
| ChromaDB 操作失败 | 500 | `{ "detail": "向量数据库操作失败" }` |
| 空问题 | 422 | Pydantic validation 自动处理 |

---

## 7. 测试计划

| # | 测试用例 | 覆盖范围 |
|---|---|---|
| 1 | `test_upload_and_list` | 上传 TXT 文件 → 验证 201 + chunk_count → GET /documents 确认存在 |
| 2 | `test_qa_ask` | 上传包含特定事实的文档 → /qa/ask 提问 → 非流式验证答案含正确事实+引用 |
| 3 | `test_delete_document` | 上传 → DELETE → 确认列表不包含 → QA 搜不到已删内容 |
| 4 | `test_unsupported_file_type` | 上传 .exe → 验证 400 |
| 5 | `test_rag_vs_bare_compare` | /qa/compare 提问 → 验证两种回答均非空且 rag_citations 存在 |

使用 `pytest + httpx.AsyncClient` + `TestClient`。ChromaDB 测试使用内存模式（`chromadb.Client()`），测试间独立。

---

## 8. Skills 调用方案

| 阶段 | Skill | 用途 |
|---|---|---|
| 设计完成后 | `writing-plans` | 将本 spec 转化为详细实现计划 |
| 实现各模块 | `test-driven-development` | 每个 service/endpoint 先写测试 |
| 并行实现 | `subagent-driven-development` | 独立模块并行开发 |
| 完成验证 | `verification-before-completion` | 运行全量测试确认通过 |
| 代码审查 | `requesting-code-review` | 合并前质量审查 |

---

## 9. 依赖清单

```
# Web 框架
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0
python-multipart>=0.0.9

# LLM / Embedding
openai>=1.30.0

# 流式输出
sse-starlette>=2.1.0

# 向量数据库
chromadb>=0.5.0

# 文档处理
pdfplumber>=0.11.0

# 工具
python-dotenv>=1.0.0
tiktoken>=0.7.0          # token 计数用于分块

# 测试
pytest>=8.0.0
httpx>=0.27.0
pytest-asyncio>=0.23.0
```

---

## 10. 范围边界

**在范围内:**
- PDF / MD / TXT 上传与解析
- 混合分块 + Embedding + ChromaDB 存储
- SSE 流式 RAG 问答 + 引用来源
- RAG vs 纯 LLM 对比端点
- 基本 CRUD 文档管理
- pytest 测试覆盖

**不在范围内 (不做):**
- 前端 UI（仅 API 服务）
- 用户认证/多租户
- OCR 图片文字识别
- 文档更新/编辑
- 多轮对话上下文记忆
- Docker 容器化（可选，非必须）
