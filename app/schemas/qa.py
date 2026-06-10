from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000, description="用户自然语言问题")
    top_k: int = Field(default=5, ge=1, le=50, description="检索的chunk数量")
    mode: str = Field(default="rag", description="对话模式: rag | bare")


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
