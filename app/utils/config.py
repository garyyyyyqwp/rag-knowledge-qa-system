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
