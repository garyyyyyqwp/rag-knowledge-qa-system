import re
from dataclasses import dataclass, field

import tiktoken

from app.utils.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS

# Use cl100k_base for token counting
_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    content: str
    index: int
    token_count: int = 0

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = len(_encoding.encode(self.content))


def _count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _split_by_sentences(text: str, max_tokens: int) -> list[str]:
    """Split text on sentence boundaries, respecting max_tokens per chunk."""
    # Split on Chinese/English sentence endings
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append("".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        chunks.append("".join(current))

    return chunks


def _split_by_headings(text: str) -> list[str]:
    """Split markdown text by ## / ### headings."""
    # Match lines starting with ## or ### (but not # which is doc title)
    parts = re.split(r'\n(?=#{2,3}\s)', text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_paragraphs(text: str) -> list[str]:
    """Split text by double newlines (paragraph boundaries)."""
    parts = re.split(r'\n\s*\n', text)
    return [p.strip() for p in parts if p.strip()]


def _is_markdown(text: str) -> bool:
    """Heuristic: detect if text looks like markdown (has ## headings)."""
    return bool(re.search(r'^#{2,3}\s', text, re.MULTILINE))


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Hybrid chunking strategy:
    1. Semantic boundary split: MD by headings, TXT by paragraphs
    2. If a segment exceeds max_tokens, split by sentences
    3. Add overlap between adjacent chunks
    """
    if not text or not text.strip():
        return []

    # Phase 1: Semantic boundary splitting
    if _is_markdown(text):
        segments = _split_by_headings(text)
    else:
        segments = _split_by_paragraphs(text)

    # Phase 2: Token-aware secondary splitting
    raw_chunks: list[str] = []
    for seg in segments:
        if _count_tokens(seg) <= max_tokens:
            raw_chunks.append(seg)
        else:
            raw_chunks.extend(_split_by_sentences(seg, max_tokens))

    # Phase 3: Add overlap
    final_chunks: list[str] = []
    for i, raw in enumerate(raw_chunks):
        if i > 0 and overlap_tokens > 0:
            # Prepend overlap from previous chunk's end
            prev = raw_chunks[i - 1]
            prev_tokens = _encoding.encode(prev)
            overlap_slice = prev_tokens[-overlap_tokens:]
            overlap_text = _encoding.decode(overlap_slice)
            raw = overlap_text + raw
        final_chunks.append(raw)

    return [
        Chunk(content=c, index=i)
        for i, c in enumerate(final_chunks)
    ]
