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
