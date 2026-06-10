import pytest
from app.services.chunker import chunk_text, Chunk

# Use small token limits for testing
TEST_MAX_TOKENS = 50
TEST_OVERLAP = 10


class TestChunkText:
    """Tests for hybrid chunking strategy."""

    def test_short_text_single_chunk(self):
        """Short text under token limit returns one chunk."""
        text = "This is a short sentence."
        chunks = chunk_text(text, max_tokens=TEST_MAX_TOKENS, overlap_tokens=TEST_OVERLAP)
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].content == text
        assert chunks[0].index == 0

    def test_markdown_heading_split(self):
        """Markdown text splits on ## headings."""
        text = "## Section One\nContent for section one.\n\n## Section Two\nContent for section two."
        chunks = chunk_text(text, max_tokens=500, overlap_tokens=50)
        assert len(chunks) >= 2
        assert any("Section One" in c.content for c in chunks)
        assert any("Section Two" in c.content for c in chunks)

    def test_paragraph_split(self):
        """Plain text splits on double newlines."""
        text = "First paragraph here.\n\nSecond paragraph is separate.\n\nThird one too."
        chunks = chunk_text(text, max_tokens=500, overlap_tokens=50)
        assert len(chunks) >= 3

    def test_long_paragraph_splits_by_sentence(self):
        """A paragraph exceeding max_tokens splits on sentence boundaries."""
        sentence = "This is sentence number {} with enough words to fill space. "
        text = "".join(sentence.format(i) for i in range(30))
        chunks = chunk_text(text, max_tokens=TEST_MAX_TOKENS, overlap_tokens=5)
        assert len(chunks) > 1
        for c in chunks:
            assert c.index >= 0

    def test_chunk_overlap(self):
        """Adjacent chunks share overlap content."""
        text = ("A" * 20 + " ") * 30  # long text that forces multiple chunks
        chunks = chunk_text(text, max_tokens=50, overlap_tokens=20)
        if len(chunks) >= 2:
            overlap_window = chunks[0].content[-30:]
            assert any(overlap_window[:20] in chunks[1].content for _ in [1])

    def test_empty_text(self):
        """Empty text returns empty list."""
        chunks = chunk_text("", max_tokens=500, overlap_tokens=50)
        assert chunks == []

    def test_chunk_metadata_present(self):
        """Each chunk has required fields."""
        text = "Hello world content."
        chunks = chunk_text(text, max_tokens=500, overlap_tokens=50)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.content == text
        assert c.index == 0
        assert c.token_count > 0
