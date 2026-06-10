import uuid
from datetime import datetime, timezone

import chromadb

from app.services.embedding import embed_texts
from app.utils.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


class VectorStoreError(Exception):
    """Raised when a ChromaDB operation fails."""
    pass


class VectorStore:
    """Encapsulates ChromaDB operations for the knowledge base."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def add_document(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        chunks: list[dict],  # [{"content": str, "index": int}, ...]
    ) -> int:
        """Add chunks of a document to ChromaDB. Returns chunk count."""
        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        embeddings = await embed_texts(texts)

        ids = [f"{doc_id}_{c['index']}" for c in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "chunk_index": c["index"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for c in chunks
        ]

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        except Exception as e:
            raise VectorStoreError(f"向量存储失败: {str(e)}") from e

        return len(chunks)

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search for chunks relevant to the query."""
        from app.services.embedding import embed_single

        if self.collection.count() == 0:
            return []

        query_embedding = await embed_single(query)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise VectorStoreError(f"向量检索失败: {str(e)}") from e

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Convert cosine distance to similarity score (1 - distance)
                score = 1.0 - distance
                hits.append({
                    "chunk_id": chunk_id,
                    "doc_id": metadata.get("doc_id", ""),
                    "filename": metadata.get("filename", ""),
                    "file_type": metadata.get("file_type", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "content": document,
                    "content_preview": document[:200] if document else "",
                    "score": round(score, 4),
                })

        return hits

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns number of chunks removed."""
        try:
            existing = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"],
            )
            count = len(existing["ids"]) if existing["ids"] else 0

            if count > 0:
                self.collection.delete(where={"doc_id": doc_id})

            return count
        except Exception as e:
            raise VectorStoreError(f"文档删除失败: {str(e)}") from e

    def list_documents(self) -> list[dict]:
        """List all unique documents with chunk counts."""
        try:
            all_data = self.collection.get(include=["metadatas"])
        except Exception as e:
            raise VectorStoreError(f"获取文档列表失败: {str(e)}") from e

        if not all_data["metadatas"]:
            return []

        # Aggregate by doc_id
        doc_map: dict[str, dict] = {}
        for meta in all_data["metadatas"]:
            did = meta["doc_id"]
            if did not in doc_map:
                doc_map[did] = {
                    "doc_id": did,
                    "filename": meta.get("filename", ""),
                    "file_type": meta.get("file_type", ""),
                    "chunk_count": 0,
                    "created_at": meta.get("created_at", ""),
                }
            doc_map[did]["chunk_count"] += 1

        return list(doc_map.values())

    def doc_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the store."""
        existing = self.collection.get(
            where={"doc_id": doc_id},
            limit=1,
        )
        return len(existing["ids"]) > 0 if existing["ids"] else False

    def count(self) -> int:
        """Return total number of chunks."""
        return self.collection.count()


# Singleton instance
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
