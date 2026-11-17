"""Unit tests for retrieval engine: BM25, RRF fusion, FK neighbors, and glossary expansion."""

from src.retrieval.indexer import DocumentChunk, load_chunks
from src.retrieval.retriever import HybridRetriever, rrf_merge
from src.retrieval.store import BM25Store


def test_load_chunks_integrity():
    chunks = load_chunks()
    assert len(chunks) > 0

    doc_types = {c.doc_type for c in chunks}
    assert "table" in doc_types
    assert "glossary" in doc_types
    assert "example" in doc_types

    # Ensure all 7 tables exist as documents
    table_ids = {c.id for c in chunks if c.doc_type == "table"}
    expected_tables = {
        "table:categories",
        "table:customers",
        "table:products",
        "table:orders",
        "table:order_items",
        "table:payments",
        "table:shipments",
    }
    assert expected_tables.issubset(table_ids)

    # Verify content hash presence
    for ch in chunks:
        assert ch.content_hash, f"Missing content hash for {ch.id}"


def test_bm25_store_search():
    store = BM25Store()
    results = store.search("revenue products", top_k=5)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert score > 0
    assert any(term in top_chunk.content.lower() for term in ["revenue", "product"])


def test_rrf_merge_ranking():
    c1 = DocumentChunk.create("1", "table", "T1", "content 1")
    c2 = DocumentChunk.create("2", "table", "T2", "content 2")
    c3 = DocumentChunk.create("3", "table", "T3", "content 3")

    vec_results = [(c1, 0.9), (c2, 0.8), (c3, 0.7)]
    bm25_results = [(c2, 10.0), (c1, 8.0), (c3, 5.0)]

    merged = rrf_merge(vec_results, bm25_results, k=60, top_n=3)
    assert len(merged) == 3
    # c1 and c2 should be at the top
    assert merged[0].id in ("1", "2")
    assert merged[1].id in ("1", "2")
    assert merged[2].id == "3"


def test_deterministic_expansions_fk_and_glossary():
    retriever = HybridRetriever()

    # Query with order_items table chunk should expand to orders and products
    order_items_chunk = retriever.chunk_by_id["table:order_items"]
    expansions = retriever._deterministic_expansions(
        "What is our total revenue?", [order_items_chunk]
    )
    expanded_ids = {c.id for c in expansions}

    # Should include orders and products neighbor tables
    assert "table:orders" in expanded_ids
    assert "table:products" in expanded_ids

    # Should include glossary:net_revenue because "revenue" is in question
    assert "glossary:net_revenue" in expanded_ids


def test_schema_mode_full():
    from src.config import settings

    retriever = HybridRetriever()
    original_mode = settings.SCHEMA_MODE
    try:
        settings.SCHEMA_MODE = "full"
        results = retriever.retrieve("Any question")
        assert len(results) >= 7  # At least all 7 tables
        doc_types = {c.doc_type for c in results}
        assert "table" in doc_types
    finally:
        settings.SCHEMA_MODE = original_mode
