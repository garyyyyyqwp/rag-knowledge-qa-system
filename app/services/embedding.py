from openai import AsyncOpenAI, APIError, AuthenticationError, RateLimitError

from app.utils.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
)

_client: AsyncOpenAI | None = None


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


def _get_embedding_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text strings.

    Returns a list of embedding vectors (list of floats).

    Raises:
        EmbeddingError: when the API call fails with a clear message
    """
    client = _get_embedding_client()
    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
    except AuthenticationError as e:
        raise EmbeddingError(
            f"Embedding API 认证失败，请检查 EMBEDDING_API_KEY 是否正确。"
            f"当前 Base URL: {EMBEDDING_BASE_URL}"
        ) from e
    except RateLimitError as e:
        # Extract detail from 智谱 error body (e.g. "余额不足或无可用资源包,请充值")
        detail = ""
        if hasattr(e, "body") and isinstance(e.body, dict):
            detail = e.body.get("error", {}).get("message", "")
        msg = detail or str(e)
        if "余额" in msg or "资源包" in msg:
            raise EmbeddingError(
                f"智谱 API 余额不足（{msg}）。"
                f"请到 https://open.bigmodel.cn 充值后再试。"
            ) from e
        raise EmbeddingError(
            f"Embedding API 请求频率过高（{msg}），请稍后重试。"
        ) from e
    except APIError as e:
        status_code = getattr(e, "status_code", None)
        if status_code == 404:
            raise EmbeddingError(
                f"Embedding 模型 '{EMBEDDING_MODEL}' 在当前 API ({EMBEDDING_BASE_URL}) 上不可用。"
                f"请确认模型名称是否正确，或更换 EMBEDDING_BASE_URL。"
            ) from e
        else:
            raise EmbeddingError(
                f"Embedding API 调用失败 (HTTP {status_code}): {str(e)}"
            ) from e
    except Exception as e:
        raise EmbeddingError(
            f"Embedding 生成失败: {str(e)}。请检查 EMBEDDING_BASE_URL ({EMBEDDING_BASE_URL}) 和网络连接。"
        ) from e

    return [item.embedding for item in response.data]


async def embed_single(text: str) -> list[float]:
    """Generate embedding for a single text string."""
    results = await embed_texts([text])
    return results[0]
