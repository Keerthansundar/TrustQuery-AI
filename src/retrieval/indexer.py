"""Document indexer for RAG retrieval over schema docs, business glossary, and examples."""

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import yaml

from src.config import settings
from src.llm.llm_client import openai_client

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    id: str
    doc_type: str  # 'table', 'glossary', 'example'
    title: str
    content: str
    metadata: dict[str, Any]
    content_hash: str

    @classmethod
    def create(
        cls,
        id_: str,
        doc_type: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> "DocumentChunk":
        meta = metadata or {}
        meta["doc_type"] = doc_type
        meta["title"] = title
        text_for_hashing = f"{id_}|{title}|{content.strip()}"
        c_hash = hashlib.sha256(text_for_hashing.encode("utf-8")).hexdigest()
        return cls(
            id=id_,
            doc_type=doc_type,
            title=title,
            content=content.strip(),
            metadata=meta,
            content_hash=c_hash,
        )


def load_chunks() -> list[DocumentChunk]:
    """Loads and formats all document chunks from schema_docs.yaml and examples.yaml."""
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    schema_path = data_dir / "schema_docs.yaml"
    examples_path = data_dir / "examples.yaml"

    chunks: list[DocumentChunk] = []

    # 1. Parse tables and glossary from schema_docs.yaml
    if schema_path.exists():
        with open(schema_path, encoding="utf-8") as f:
            schema_data = yaml.safe_load(f) or {}

        # Table documents
        tables = schema_data.get("tables", {})
        for table_name, table_info in tables.items():
            desc = table_info.get("description", "")
            cols = table_info.get("columns", {})
            col_lines = "\n".join(f"  - {col}: {cdesc}" for col, cdesc in cols.items())
            content = f"Table: public.{table_name}\nDescription: {desc}\nColumns:\n{col_lines}"
            chunks.append(
                DocumentChunk.create(
                    id_=f"table:{table_name}",
                    doc_type="table",
                    title=f"Table public.{table_name}",
                    content=content,
                    metadata={"table_name": table_name, "columns": list(cols.keys())},
                )
            )

        # Glossary documents
        glossary = schema_data.get("glossary", {})
        for term_key, term_info in glossary.items():
            term = term_info.get("term", term_key)
            definition = term_info.get("definition", "")
            content = f"Business Glossary Term: {term}\nDefinition: {definition}"
            chunks.append(
                DocumentChunk.create(
                    id_=f"glossary:{term_key}",
                    doc_type="glossary",
                    title=f"Glossary: {term}",
                    content=content,
                    metadata={"term_key": term_key, "term": term},
                )
            )

    # 2. Parse few-shot examples from examples.yaml
    if examples_path.exists():
        with open(examples_path, encoding="utf-8") as f:
            examples_data = yaml.safe_load(f) or {}

        examples = examples_data.get("examples", [])
        for ex in examples:
            ex_id = ex.get("id", "")
            q = ex.get("question", "")
            sql = ex.get("sql", "").strip()
            content = f"Example Question: {q}\nPostgreSQL Query:\n{sql}"
            chunks.append(
                DocumentChunk.create(
                    id_=f"example:{ex_id}",
                    doc_type="example",
                    title=f"Example: {q}",
                    content=content,
                    metadata={"example_id": ex_id, "question": q, "sql": sql},
                )
            )

    return chunks


def ensure_vector_dimension(cur, target_dim: int = 1536) -> None:
    """Checks and updates rag.documents embedding vector dimensionality if needed."""
    cur.execute(
        """
        SELECT atttypmod
        FROM pg_attribute
        JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
        JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
        WHERE pg_namespace.nspname = 'rag'
          AND pg_class.relname = 'documents'
          AND pg_attribute.attname = 'embedding';
        """
    )
    res = cur.fetchone()
    current_dim = res[0] if res else None

    # If column exists with different dimension, migrate it
    if current_dim is not None and current_dim != target_dim:
        logger.info(
            "Migrating rag.documents embedding dimension from %s to %d for OpenAI embeddings...",
            current_dim,
            target_dim,
        )
        cur.execute("UPDATE rag.documents SET embedding = NULL;")
        cur.execute(f"ALTER TABLE rag.documents ALTER COLUMN embedding TYPE vector({target_dim});")
        logger.info(
            "Successfully altered rag.documents embedding column to vector(%d).", target_dim
        )


def index_documents(force_reembed: bool = False, rebuild: bool = False) -> int:
    """Embeds and upserts chunks into rag.documents using OpenAI embeddings and pgvector.

    - Re-embeds only changed or newly added documents unless force_reembed or rebuild is True.
    - If rebuild is True, wipes existing embeddings to re-index all chunks cleanly.
    """
    chunks = load_chunks()
    if not chunks:
        logger.warning("No document chunks found to index.")
        return 0

    print(f"Loaded {len(chunks)} document chunks. Connecting to PostgreSQL...")

    conn = psycopg.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        autocommit=True,
    )

    with conn.cursor() as cur:
        # 1. Ensure table exists with configured dimensions (1536 for text-embedding-3-small)
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS rag.documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({settings.EMBEDDING_DIMENSIONS}),
                content_hash TEXT UNIQUE NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        ensure_vector_dimension(cur, settings.EMBEDDING_DIMENSIONS)

        if rebuild:
            print("Rebuild requested: clearing existing embeddings in rag.documents...")
            cur.execute("TRUNCATE rag.documents;")

        # 2. Check existing hashes
        cur.execute("SELECT id, content_hash FROM rag.documents WHERE embedding IS NOT NULL;")
        existing_hashes = dict(cur.fetchall())

        to_embed: list[DocumentChunk] = []
        for ch in chunks:
            if (
                force_reembed
                or rebuild
                or ch.id not in existing_hashes
                or existing_hashes[ch.id] != ch.content_hash
            ):
                to_embed.append(ch)

        if to_embed:
            print(
                f"Embedding {len(to_embed)} new/modified documents via OpenAI {settings.EMBEDDING_MODEL} (dim={settings.EMBEDDING_DIMENSIONS})..."
            )
            texts = [c.content for c in to_embed]
            embeddings = openai_client.embed(texts)

            for chunk, emb in zip(to_embed, embeddings, strict=False):
                meta_json = json.dumps(chunk.metadata)
                cur.execute(
                    """
                    INSERT INTO rag.documents (id, content, metadata, embedding, content_hash)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        content_hash = EXCLUDED.content_hash;
                    """,
                    (chunk.id, chunk.content, meta_json, emb, chunk.content_hash),
                )
            print(f"Successfully upserted {len(to_embed)} embedded documents.")
        else:
            print("All document chunks are already up to date in rag.documents.")

    conn.close()
    return len(chunks)


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Index documentation chunks into pgvector.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe existing embeddings and re-index everything with OpenAI embeddings",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-embedding of all chunks even if hash matches",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        count = index_documents(force_reembed=args.force, rebuild=args.rebuild)
        print(f"[SUCCESS] Indexing complete. Total active chunks: {count}")
    except Exception as e:
        print(f"[ERROR] Indexing failed: {e}", file=sys.stderr)
        sys.exit(1)
