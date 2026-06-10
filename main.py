from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

# Static files directory
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)

# Routers
app.include_router(documents.router, prefix="/api/v1/documents")
app.include_router(qa.router, prefix="/api/v1/qa")

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the SPA frontend."""
    return FileResponse(static_dir / "index.html")


@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "ok", "service": "RAG Knowledge Base QA", "version": "0.1.0"}


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    # If detail is set by a route handler (not the default Starlette 404),
    # pass it through unchanged
    detail = getattr(exc, "detail", "")
    if detail and detail != "Not Found":
        return JSONResponse(status_code=404, content={"detail": detail})

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


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误", "message": "请联系管理员"},
    )


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """SPA fallback — serve index.html for non-API client-side routes."""
    # Explicitly reject known non-SPA paths
    if full_path.startswith(("api/", "health", "docs", "openapi.json", "redoc")):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")
    # Reject Vite HMR / dev tool requests that pollute the log
    if full_path.startswith("@vite/") or full_path.startswith("__vite"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")
    # Reject static asset requests — let the StaticFiles mount handle them
    if full_path.startswith("static/"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")
    # Serve existing static files directly
    target = static_dir / full_path
    if target.exists() and target.is_file():
        return FileResponse(target)
    return FileResponse(static_dir / "index.html")
