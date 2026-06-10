# RAG 检索增强生成 — 前置学习计划与讲解文件设计方案

> 基于已完成的 RAG 微服务项目（FastAPI + ChromaDB + OpenAI），设计全套前置学习路径。

---

## 一、现状分析

### 1.1 已有项目技术栈

- **Web 框架**: FastAPI + Pydantic v2
- **LLM 客户端**: AsyncOpenAI (支持任意 OpenAI 兼容 API)
- **Embedding**: text-embedding-3-small (可切换 DeepSeek)
- **向量数据库**: ChromaDB PersistentClient (cosine 相似度)
- **文档解析**: pdfplumber (PDF) + 原生 (MD/TXT)
- **分块策略**: 混合策略 (语义边界 → token 二次切分 → overlap)
- **流式输出**: sse-starlette (4 种结构化 SSE 事件)
- **Token 计数**: tiktoken (cl100k_base)
- **测试**: pytest + httpx + pytest-asyncio

### 1.2 项目架构 (已有)

```
FastAPI App (main.py)
├── /api/v1/documents/       # 文档 CRUD
│   ├── POST   /upload       # 上传 → 解析 → 分块 → Embedding → ChromaDB
│   ├── GET    /             # 列出已入库文档
│   └── DELETE /{doc_id}     # 删除文档及向量
└── /api/v1/qa/              # 智能问答
    ├── POST  /ask           # RAG 增强问答 (SSE流式)
    ├── POST  /ask/sync      # RAG 增强问答 (非流式)
    ├── POST  /compare       # RAG vs 纯LLM 对比 (SSE流式)
    └── POST  /compare/sync  # RAG vs 纯LLM 对比 (非流式)
```

---

## 二、学习文件清单 (12 个 HTML 讲解文件)

学习路径由浅入深，从 RAG 基础概念到完整系统架构：

| 编号 | 文件名 | 主题 | 对应学习日 |
|------|--------|------|-----------|
| 01 | `01-rag-overview.html` | RAG 技术概述：为什么大模型需要外部知识 | Day 1-2 |
| 02 | `02-embeddings.html` | Embedding 与向量化：文本到向量的魔法 | Day 1-2 |
| 03 | `03-vector-similarity.html` | 向量相似度检索：余弦相似度与 Top-K | Day 1-2 |
| 04 | `04-minimal-rag.html` | 最简 RAG 管道：Embed → Retrieve → Generate | Day 1-2 |
| 05 | `05-text-chunking.html` | 文本切分策略：从固定长度到语义切分 | Day 3-4 |
| 06 | `06-document-parsing.html` | 文档解析：PDF / Markdown / TXT 处理 | Day 3-4 |
| 07 | `07-chromadb.html` | ChromaDB 向量数据库实战 | Day 3-4 |
| 08 | `08-fastapi-microservice.html` | FastAPI 微服务架构设计 | Day 5-7 |
| 09 | `09-sse-streaming.html` | SSE 流式输出原理与实践 | Day 5-7 |
| 10 | `10-rag-pipeline.html` | RAG 完整管道：从文档到答案 | Day 5-7 |
| 11 | `11-rag-vs-bare.html` | RAG vs 纯 LLM 对比分析 | Day 5-7 |
| 12 | `12-rag-architecture.html` | 个人知识库系统架构全景 | Day 5-7 |

---

## 三、每个文件的设计要点

### 文件 01: `01-rag-overview.html` — RAG 技术概述
- **核心概念**: 什么是 RAG、为什么 LLM 需要外部知识（幻觉问题、知识截止日期、领域知识缺失）
- **RAG 工作流程**: 索引(Indexing) → 检索(Retrieval) → 增强(Augmentation) → 生成(Generation)
- **实际场景**: 客服系统、法律咨询、医疗问答、个人知识库
- **代码示例**: 伪代码展示 RAG 基本流程
- **图示**: RAG 架构流程图

### 文件 02: `02-embeddings.html` — Embedding 与向量化
- **核心概念**: 什么是 Embedding、词嵌入 vs 句嵌入、向量空间
- **Embedding API 调用**: OpenAI text-embedding-3-small 实战代码
- **项目对应代码**: `app/services/embedding.py` 的 `embed_texts()` 和 `embed_single()`
- **可视化**: 2D/3D 降维可视化示意 (t-SNE)
- **动手实验**: 多组句子语义相似度对比

### 文件 03: `03-vector-similarity.html` — 向量相似度检索
- **核心概念**: 余弦相似度、欧氏距离、点积 — 公式与直观理解
- **NumPy 实现**: 余弦相似度计算代码
- **Top-K 检索**: 什么是 Top-K、如何选择 K 值
- **项目对应配置**: `RAG_TOP_K=5` 的含义
- **可视化**: 热力图展示句子间相似度矩阵

### 文件 04: `04-minimal-rag.html` — 最简 RAG 管道
- **核心概念**: 不依赖任何向量数据库，纯 Python 实现完整 RAG
- **完整代码**: 准备知识片段 → Chunk → Embed → 内存向量库 → 检索 → LLM 生成
- **对比实验**: 有 RAG vs 无 RAG 的回答质量差异
- **项目对应**: `app/services/rag_pipeline.py` 的设计思路源头

### 文件 05: `05-text-chunking.html` — 文本切分策略
- **核心概念**: 为什么需要 Chunk、Chunk Size 和 Overlap 的影响
- **三种策略**: 固定长度切分 vs 语义切分 vs 混合切分 (本项目的选择)
- **项目对应代码**: `app/services/chunker.py` 的混合切分引擎
  - 语义边界: Markdown `##` 标题、`\n\n` 段落
  - Token 二次切分: 按句子边界 (`。！？.!?`)
  - Overlap: 相邻 chunk 重叠 `CHUNK_OVERLAP_TOKENS=50`
- **A/B 对比**: 不同参数对检索效果的实验对比
- **配置**: `CHUNK_MAX_TOKENS=512`, `CHUNK_OVERLAP_TOKENS=50`

### 文件 06: `06-document-parsing.html` — 文档解析
- **核心概念**: 为什么需要文档解析、支持的文件格式
- **PDF 解析**: pdfplumber 逐页提取文本
- **Markdown/TXT 解析**: UTF-8 解码
- **项目对应代码**: `app/services/parser.py`
  - `parse_pdf()` / `parse_markdown()` / `parse_txt()`
  - 异常体系: `ParseError` → `UnsupportedFileTypeError`, `FileTooLargeError`
  - 10MB 文件大小限制
- **实践指导**: 如何处理复杂 PDF (表格、图片等限制)

### 文件 07: `07-chromadb.html` — ChromaDB 向量数据库
- **核心概念**: 什么是向量数据库、为什么不用传统数据库
- **ChromaDB 基本操作**: Collection 创建、向量插入、相似度查询、元数据过滤、持久化
- **项目对应代码**: `app/services/vector_store.py` 的完整 CRUD
  - `add_document()` — 批量插入向量 + 元数据
  - `search()` — 余弦相似度 Top-K 查询
  - `delete_document()` — 按 doc_id 过滤删除
  - `list_documents()` — 聚合元数据列出文档
  - Singleton 模式: `get_vector_store()`
- **数据模型**: metadata 包含 `doc_id`, `filename`, `file_type`, `chunk_index`, `created_at`
- **配置**: `hnsw:space: cosine` 余弦距离

### 文件 08: `08-fastapi-microservice.html` — FastAPI 微服务架构
- **核心概念**: FastAPI 基础、路由、中间件、依赖注入
- **三层架构**: Routers → Schemas → Services
- **项目对应结构**:
  - `main.py`: 应用入口、CORS、路由挂载、异常处理
  - `app/routers/documents.py`: 文档 CRUD 端点
  - `app/routers/qa.py`: 问答端点
  - `app/schemas/`: Pydantic v2 请求/响应模型
  - `app/services/`: 业务逻辑层
- **API 设计**: RESTful 规范、状态码、错误处理

### 文件 09: `09-sse-streaming.html` — SSE 流式输出
- **核心概念**: 什么是 SSE、SSE vs WebSocket、EventSource API
- **项目对应代码**: `app/services/streaming.py`
  - 4 种结构化事件: `retrieval` → `answer` → `citations` → `done`
  - `sse-starlette` 的 `EventSourceResponse`
  - 对比端点: `rag_answer` / `bare_answer` 交错发送
- **前端消费**: JavaScript EventSource 示例代码

### 文件 10: `10-rag-pipeline.html` — RAG 完整管道
- **核心概念**: 从用户提问到最终答案的完整数据流
- **项目对应代码**: `app/services/rag_pipeline.py`
  - `retrieve_chunks()`: Embed 问题 → ChromaDB 检索
  - `generate_answer_stream()`: 构建 Prompt → LLM 流式生成
  - `generate_bare_answer_stream()`: 纯 LLM (无 RAG 对比)
  - Prompt 模板: 系统角色 + 参考资料 + 用户问题 + 引用要求
  - `_build_context()`: 拼接检索到的 Chunk
  - `_extract_citations()`: 提取引用元数据
- **数据流图**: 完整管道流程图

### 文件 11: `11-rag-vs-bare.html` — RAG vs 纯 LLM 对比
- **核心概念**: 为什么需要对比、对比维度
- **项目对应端点**: `POST /api/v1/qa/compare` (SSE) 和 `/compare/sync`
- **对比维度**: 准确性、幻觉率、引用完整性、回答详细度
- **实验案例**: 已知事实 vs 未知事实的对比
- **结果分析**: 何时 RAG 优势明显、何时差距不大

### 文件 12: `12-rag-architecture.html` — 系统架构全景
- **核心概念**: 完整系统架构回顾
- **项目全景**: 所有模块关系图、数据流图、部署拓扑
- **技术选型总结**: 为什么选这些技术
- **扩展方向**: 多轮对话、图片 OCR、多租户、Docker 部署
- **学习路线图**: 从 Week 1 到 Week 2 的技能成长

---

## 四、HTML 设计规范

每个 HTML 文件需满足以下标准：

1. **视觉设计**: 
   - 现代简约风格，使用 CSS Grid/Flexbox 布局
   - 柔和配色方案 (深蓝主色调 #1a237e，配合浅蓝 #e8eaf6 背景)
   - 代码块使用深色主题 (类似 VS Code Dark)
   - 响应式设计，适配不同屏幕尺寸

2. **内容结构**:
   - 顶部导航: 面包屑 + 学习路径进度条
   - 标题区: 主题标题 + 一句话摘要
   - 概念讲解区: 图文结合，使用 SVG 图表
   - 代码示例区: 实际可运行的 Python 代码
   - 实践练习区: 动手实验指导
   - 要点总结区: 关键知识点回顾
   - 底部导航: 上一篇/下一篇链接

3. **交互元素**:
   - 可折叠的详细说明区域
   - 代码复制按钮
   - 术语提示 (tooltip)

4. **内容语言**: 中文为主，代码和术语保留英文

---

## 五、输出目录结构

```
c:\Users\35452\Desktop\week2task\学习任务\
├── .trae\
│   └── documents\
│       └── rag-learning-plan.md        # 本计划文件
├── css\
│   └── learning-style.css              # 共享样式表
├── js\
│   └── learning-script.js              # 共享脚本 (导航、代码复制等)
├── 01-rag-overview.html
├── 02-embeddings.html
├── 03-vector-similarity.html
├── 04-minimal-rag.html
├── 05-text-chunking.html
├── 06-document-parsing.html
├── 07-chromadb.html
├── 08-fastapi-microservice.html
├── 09-sse-streaming.html
├── 10-rag-pipeline.html
├── 11-rag-vs-bare.html
└── 12-rag-architecture.html
```

---

## 六、实施步骤

1. 创建 `css/learning-style.css` — 共享样式表 (现代简约设计)
2. 创建 `js/learning-script.js` — 共享交互脚本
3. 按顺序创建 12 个 HTML 文件，每个文件包含:
   - 完整的概念讲解 (对应项目代码)
   - 代码示例 (可运行)
   - SVG 图表 / ASCII 流程图
   - 实践练习指导
4. 每个文件之间通过导航链接串联，形成完整学习路径

---

## 七、验证方式

- 每个 HTML 文件可独立在浏览器中打开并正常渲染
- 代码示例与项目实际代码保持一致
- 学习路径从基础到进阶，逻辑连贯
- 样式统一美观，导航流畅
