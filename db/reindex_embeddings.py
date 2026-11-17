"""Re-indexes all schema, glossary, and example chunks with OpenAI embeddings into pgvector.

Run this script after adding OPENAI_API_KEY to rebuild the vector store
and embed all documents using text-embedding-3-small (1536 dimensions).
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.indexer import index_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reindex_embeddings")

if __name__ == "__main__":
    print("=" * 65)
    print("Migrating Vector Store Embeddings to OpenAI (text-embedding-3-small, dim=1536)")
    print("=" * 65)
    try:
        total = index_documents(force_reembed=True, rebuild=True)
        print(f"\n[SUCCESS] Successfully rebuilt vector index with {total} chunks!")
    except Exception as exc:
        print(f"\n[ERROR] Migration and re-indexing failed: {exc}", file=sys.stderr)
        sys.exit(1)
