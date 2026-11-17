"""Retrieval storage backends: In-memory BM25 and pgvector cosine search."""

import json
import logging
import re

import psycopg
from rank_bm25 import BM25Okapi

from src.config import settings
from src.retrieval.indexer import DocumentChunk, load_chunks

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25Store:
    """In-memory BM25 index built from the static document collection."""

    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else load_chunks()
        self.chunk_by_id = {c.id: c for c in self.chunks}
        self.corpus = [tokenize(f"{c.title} {c.content}") for c in self.chunks]
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 8) -> list[tuple[DocumentChunk, float]]:
        """Returns top_k (chunk, bm25_score) tuples."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_pairs = list(zip(self.chunks, scores, strict=False))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in scored_pairs[:top_k] if score > 0.0]


class VectorStore:
    """pgvector cosine distance retrieval over rag.documents."""

    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self._fallback_chunks = chunks or load_chunks()
        self._fallback_map = {c.id: c for c in self._fallback_chunks}

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
    ) -> list[tuple[DocumentChunk, float]]:
        """Queries pgvector using cosine distance (<=>). Returns (chunk, similarity)."""
        if not query_embedding:
            return []

        try:
            conn = psycopg.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                dbname=settings.POSTGRES_DB,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                autocommit=True,
            )
            with conn.cursor() as cur:
                emb_str = f"[{','.join(str(x) for x in query_embedding)}]"
                cur.execute(
                    """
                    SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS similarity
                    FROM rag.documents
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (emb_str, emb_str, top_k),
                )
                results = []
                for row in cur.fetchall():
                    cid, content, meta, sim = row
                    meta_dict = meta if isinstance(meta, dict) else json.loads(meta)
                    chunk = DocumentChunk(
                        id=cid,
                        doc_type=meta_dict.get("doc_type", "unknown"),
                        title=meta_dict.get("title", cid),
                        content=content,
                        metadata=meta_dict,
                        content_hash="",
                    )
                    results.append((chunk, float(sim)))
                conn.close()
                return results
        except Exception as exc:
            logger.warning(
                "pgvector search failed or DB is offline (%s). Using fallback chunks.", exc
            )
            # Return first top_k fallback chunks
            return [(c, 0.5) for c in self._fallback_chunks[:top_k]]
