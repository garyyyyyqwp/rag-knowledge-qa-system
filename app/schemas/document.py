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
