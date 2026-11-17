"""Two-stage hybrid retriever with RRF fusion, LLM reranking, and coverage grading."""

import logging

from pydantic import BaseModel, Field

from src.config import settings
from src.llm.llm_client import openai_client
from src.retrieval.indexer import DocumentChunk, load_chunks
from src.retrieval.store import BM25Store, VectorStore

logger = logging.getLogger(__name__)

# Foreign key neighbor table mapping for deterministic expansion
TABLE_NEIGHBORS: dict[str, list[str]] = {
    "order_items": ["orders", "products"],
    "products": ["categories", "order_items"],
    "orders": ["customers", "order_items", "payments", "shipments"],
    "payments": ["orders"],
    "shipments": ["orders"],
    "customers": ["orders"],
    "categories": ["products"],
}

# Glossary term keywords
GLOSSARY_KEYWORDS: dict[str, list[str]] = {
    "revenue": ["net_revenue", "gross_revenue", "line_revenue"],
    "sales": ["net_revenue", "gross_revenue", "line_revenue"],
    "profit": ["profit"],
    "aov": ["aov"],
    "average order": ["aov"],
    "active": ["active_customer"],
    "return": ["return_rate"],
    "delivery": ["delivery_time"],
    "fulfilment": ["fulfilment_time"],
    "quarter": ["date_relative_windows"],
    "month": ["date_relative_windows"],
    "last 30 days": ["date_relative_windows"],
    "name": ["customer_full_name"],
    "rollup": ["category_rollups"],
    "cash": ["cash_collected"],
}


class GradingResult(BaseModel):
    is_sufficient: bool = Field(description="True if context covers the question, False otherwise")
    missing_info: str = Field(
        description="Explanation of what schema or glossary information is missing"
    )


class RerankResult(BaseModel):
    ordered_ids: list[str] = Field(
        description="Ordered list of top chunk IDs most relevant to the question"
    )


def rrf_merge(
    vector_results: list[tuple[DocumentChunk, float]],
    bm25_results: list[tuple[DocumentChunk, float]],
    k: int = 60,
    top_n: int = 8,
) -> list[DocumentChunk]:
    """Combines vector and BM25 rankings using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, DocumentChunk] = {}

    for rank, (chunk, _) in enumerate(vector_results):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.id] = chunk

    for rank, (chunk, _) in enumerate(bm25_results):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.id] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids[:top_n]]


class HybridRetriever:
    """Two-stage hybrid retriever implementing RRF fusion, LLM reranking, and grading."""

    def __init__(self) -> None:
        self.all_chunks = load_chunks()
        self.chunk_by_id = {c.id: c for c in self.all_chunks}
        self.bm25_store = BM25Store(self.all_chunks)
        self.vector_store = VectorStore(self.all_chunks)

    def _deterministic_expansions(
        self, question: str, chosen_chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Automatically includes FK neighbor tables and keyword-matched glossary entries."""
        extra_chunks: list[DocumentChunk] = []
        already_present_ids = {c.id for c in chosen_chunks}

        # 1. Expand FK neighbors from selected table chunks only.
        # Do not expand tables based on unrelated few-shot examples.
        referenced_tables = set()

        for ch in chosen_chunks:
            if ch.doc_type == "table":
                tname = ch.metadata.get("table_name")
                if tname:
                    referenced_tables.add(tname)

        for tname in list(referenced_tables):
            for neighbor in TABLE_NEIGHBORS.get(tname, []):
                nid = f"table:{neighbor}"
                if nid not in already_present_ids and nid in self.chunk_by_id:
                    extra_chunks.append(self.chunk_by_id[nid])
                    already_present_ids.add(nid)

        # 2. Match glossary terms in question text
        q_lower = question.lower()
        for keyword, term_keys in GLOSSARY_KEYWORDS.items():
            if keyword in q_lower:
                for tkey in term_keys:
                    gid = f"glossary:{tkey}"
                    if gid not in already_present_ids and gid in self.chunk_by_id:
                        extra_chunks.append(self.chunk_by_id[gid])
                        already_present_ids.add(gid)

        return extra_chunks

    def _rerank(
        self, question: str, candidate_chunks: list[DocumentChunk], top_k: int = 5
    ) -> list[DocumentChunk]:
        """Stage 2: LLM reranking down to top 5 using FAST_MODEL."""
        if not settings.RERANK_ENABLED or len(candidate_chunks) <= top_k:
            return candidate_chunks[:top_k]

        candidates_text = "\n\n".join(
            f"ID: {c.id}\nTitle: {c.title}\nContent:\n{c.content[:1000]}" 
            for c in candidate_chunks
        )

        prompt = (
            f"You are ranking database context documents for Text-to-SQL generation.\n\n"
            f"User Question: {question}\n\n"
            f"Candidate Documents:\n{candidates_text}\n\n"
            f"Select the top {top_k} documents that are most useful for writing "
            "the correct SQL query.\n\n"
            "Ranking rules:\n"
            "1. Prioritize table schemas containing the required columns.\n"
            "2. Prioritize glossary definitions only when the question requires "
            "a defined business metric.\n"
            "3. Prioritize examples with similar query intent, aggregation, "
            "filters, and joins.\n"
            "4. Penalize examples that are topically or semantically unrelated.\n"
            "5. Do not select a document merely because it mentions a related table.\n"
            "6. Do not assume joins are necessary when one table is sufficient.\n\n"
            "Return JSON with 'ordered_ids', ranked from most to least relevant."
        )

        try:
            parsed, _ = openai_client.generate_json(
                prompt=prompt,
                schema=RerankResult,
                model=settings.FAST_MODEL,
                temperature=0.0,
            )
            ordered_ids = (
                parsed.get("ordered_ids", [])
                if isinstance(parsed, dict)
                else getattr(parsed, "ordered_ids", [])
            )
            reranked = [self.chunk_by_id[cid] for cid in ordered_ids if cid in self.chunk_by_id]
            if reranked:
                # Add any remaining candidates if less than top_k
                for c in candidate_chunks:
                    if len(reranked) >= top_k:
                        break
                    if c not in reranked:
                        reranked.append(c)
                return reranked[:top_k]
        except Exception as e:
            logger.warning("LLM reranking failed (%s). Falling back to RRF ordering.", e)

        return candidate_chunks[:top_k]

    def _grade_retrieval(
        self, question: str, retrieved_chunks: list[DocumentChunk]
    ) -> tuple[bool, str]:
        """Checks if retrieved context covers the question."""
        if not settings.GRADING_ENABLED:
            return True, ""

        context_summary = "\n\n".join(
            f"- {c.title} ({c.doc_type}):\n{c.content}"
            for c in retrieved_chunks
        )

        prompt = (
            "You are a database retrieval grader.\n"
            "Determine whether the retrieved context contains enough information "
            "to construct an accurate PostgreSQL query.\n\n"
            "A context is sufficient when it provides the necessary:\n"
            "1. Tables and columns.\n"
            "2. Foreign-key relationships, if joins are needed.\n"
            "3. Business metric definitions, if the question requires them.\n\n"
            "The context does NOT need to contain actual database result values.\n"
            "Do not request unrelated tables or glossary definitions.\n"
            "missing_info must always be a string.\n"
            "If no information is missing, return an empty string \n"
            "Never return a list or array for missing_info.\n"
            "Do not mark context insufficient merely because query results "
            "are not present.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved Context:\n{context_summary}\n\n"
            "Return JSON with is_sufficient and missing_info."
        )

        try:
            parsed, _ = openai_client.generate_json(
                prompt=prompt,
                schema=GradingResult,
                model=settings.FAST_MODEL,
                temperature=0.0,
            )
            is_suff = (
                parsed.get("is_sufficient", True)
                if isinstance(parsed, dict)
                else getattr(parsed, "is_sufficient", True)
            )
            missing = (
                parsed.get("missing_info", "")
                if isinstance(parsed, dict)
                else getattr(parsed, "missing_info", "")
            )

            if isinstance(missing, list):
                missing = ", ".join(str(item) for item in missing)
            elif not isinstance(missing, str):
                missing = str(missing) if missing is not None else ""

            return is_suff, missing
        except Exception as e:
            logger.warning("Retrieval grading failed (%s). Assuming sufficient.", e)
            return True, ""

    def retrieve(self, question: str) -> list[DocumentChunk]:
        """Main retrieval pipeline: Full Schema mode check, Hybrid RRF, Rerank, Grading."""
        # 1. Check SCHEMA_MODE=full
        if settings.SCHEMA_MODE.lower() == "full":
            logger.info("SCHEMA_MODE=full: Returning all schema and glossary documents.")
            return [c for c in self.all_chunks if c.doc_type in ("table", "glossary")]

        # Stage 1: Hybrid retrieval (BM25 + pgvector)
        bm25_top = self.bm25_store.search(question, top_k=8)

        # Get question embedding for pgvector
        try:
            q_emb = openai_client.embed([question])
            query_vec = q_emb[0] if q_emb else []
        except Exception as e:
            logger.warning("Could not generate query embedding (%s). Relying on BM25.", e)
            query_vec = []

        vec_top = self.vector_store.search(query_vec, top_k=8) if query_vec else []
        rrf_candidates = rrf_merge(vec_top, bm25_top, top_n=8)

        # Stage 2: LLM reranking to top 5
        top_chunks = self._rerank(question, rrf_candidates, top_k=5)

        # Deterministic expansions (FK neighbors + glossary keywords)
        expansions = self._deterministic_expansions(question, top_chunks)
        combined_chunks = list(top_chunks)
        seen_ids = {c.id for c in combined_chunks}
        for ex in expansions:
            if ex.id not in seen_ids:
                combined_chunks.append(ex)
                seen_ids.add(ex.id)

        # Retrieval grading + optional single re-retrieval
        is_sufficient, missing = self._grade_retrieval(question, combined_chunks)
        if not is_sufficient and missing:
            logger.info(
                "Context insufficient according to grader. Re-retrieving with missing context: %s",
                missing,
            )
            expanded_query = f"{question} {missing}"
            re_bm25 = self.bm25_store.search(expanded_query, top_k=5)
            for chunk, _ in re_bm25:
                if chunk.id not in seen_ids:
                    combined_chunks.append(chunk)
                    seen_ids.add(chunk.id)

        logger.info("========== RETRIEVAL DEBUG ==========")

        for chunk in combined_chunks:
            logger.info(
                "ID=%s | TYPE=%s | TITLE=%s | CONTENT=%s",
                chunk.id,
                chunk.doc_type,
                chunk.title,
                chunk.content,
            )

        logger.info("========== END RETRIEVAL DEBUG ==========")

        return combined_chunks


# Global retriever instance
retriever = HybridRetriever()
