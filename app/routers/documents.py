import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.document import (
    DocumentUploadResponse,
    DocumentInfo,
    DocumentDeleteResponse,
    ErrorResponse,
)
from app.services.parser import (
    parse_document,
    UnsupportedFileTypeError,
    FileTooLargeError,
    FileContentInvalidError,
    ParseError,
)
from app.services.chunker import chunk_text
from app.services.vector_store import get_vector_store, VectorStoreError
from app.services.embedding import EmbeddingError

router = APIRouter(tags=["documents"])

# Supported file types and their display names
_FILE_TYPE_MAP = {"pdf": "pdf", "md": "md", "txt": "txt", "docx": "docx"}


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse, "description": "不支持的文件类型、文件损坏或解析失败"},
        413: {"model": ErrorResponse, "description": "文件超过大小限制 (10MB)"},
        500: {"model": ErrorResponse, "description": "向量存储或 Embedding 服务错误"},
    },
)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF/DOCX/MD/TXT), parse, chunk, embed, and store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件读取失败: {str(e)}")

    # Parse document
    try:
        text = await parse_document(file.filename, content)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except FileContentInvalidError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ParseError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Chunk
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="文档解析后无有效文本内容，无法分块")

    # Generate doc_id and determine file_type
    doc_id = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "txt"
    file_type = _FILE_TYPE_MAP.get(ext, "txt")

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
    except EmbeddingError as e:
        # Embedding errors should be returned as 500 with actionable info
        raise HTTPException(status_code=500, detail=str(e))
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档存储失败: {str(e)}")

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
