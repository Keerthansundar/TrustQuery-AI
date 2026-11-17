"""Minimal offline/online smoke test script for OpenAI API credentials and models.

Runs:
1. One embedding call (text-embedding-3-small, dim=1536, ~4 tokens)
2. One text-to-SQL completion call (gpt-4o-mini, minimal tokens)

Usage:
    python scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.llm.llm_client import LLMUnavailableError, get_token_usage_stats, openai_client


def main() -> None:
    print("=" * 65)
    print("Text-to-SQL Analyst: OpenAI Migration Smoke Test")
    print("=" * 65)

    key = settings.OPENAI_API_KEY
    if not key or key.strip() == "" or key.startswith("change_") or key.startswith("your-key"):
        print("\n[BLOCKED] OPENAI_API_KEY is not configured or still contains placeholder.")
        print("Please add your valid OpenAI API key in .env:")
        print("    OPENAI_API_KEY=sk-proj-...")
        print("Then run this smoke test again.")
        sys.exit(1)

    masked_key = key[:7] + "..." + key[-4:] if len(key) > 12 else "***"
    print(f"\n1. Loaded API Key: {masked_key}")
    print(
        f"2. Embedding Model: {settings.EMBEDDING_MODEL} (Expected Dim: {settings.EMBEDDING_DIMENSIONS})"
    )
    print(f"3. Chat/SQL Model:  {settings.GEN_MODEL}")

    # Step 1: Embedding call
    print("\n--- Testing Embedding API ---")
    test_text = "Total sales by customer segment"
    try:
        t0 = time.perf_counter()
        embeddings = openai_client.embed([test_text])
        emb_lat = (time.perf_counter() - t0) * 1000.0

        assert len(embeddings) == 1, "Expected 1 embedding vector"
        vec = embeddings[0]
        dim = len(vec)
        print(f"✅ Embedding succeeded in {emb_lat:.1f}ms")
        print(f"   Input text: '{test_text}'")
        print(
            f"   Vector dimensionality: {dim} (matches expected: {dim == settings.EMBEDDING_DIMENSIONS})"
        )
        print(f"   Vector sample: [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}, ...]")
    except LLMUnavailableError as e:
        print(f"❌ Embedding failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected embedding error: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Chat completion text-to-SQL call
    print("\n--- Testing Text-to-SQL Chat Completion API ---")
    prompt = "Write a PostgreSQL 16 SELECT query to return the total count of customers in table public.customers."
    system = (
        "You are a PostgreSQL expert. Return only the SQL query without markdown or explanations."
    )
    try:
        t0 = time.perf_counter()
        resp = openai_client.generate(
            prompt=prompt, system=system, model=settings.GEN_MODEL, temperature=0.0
        )
        chat_lat = (time.perf_counter() - t0) * 1000.0

        print(f"✅ Text-to-SQL completion succeeded in {chat_lat:.1f}ms (model: {resp.model})")
        print(f"   Generated SQL: {resp.text.strip()}")
        print(
            f"   Tokens used: {resp.prompt_tokens} prompt + {resp.completion_tokens} completion = {resp.total_tokens} total"
        )
    except LLMUnavailableError as e:
        print(f"❌ Chat completion failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected completion error: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Spend accounting
    stats = get_token_usage_stats()
    print("\n" + "=" * 65)
    print("Smoke Test Summary:")
    print(f"- Total API Calls:     {stats.total_calls}")
    print(
        f"- Total Tokens:        {stats.total_tokens} ({stats.embedding_tokens} embedding, {stats.prompt_tokens} prompt, {stats.completion_tokens} completion)"
    )
    print(f"- Estimated Test Cost: ~${stats.estimated_cost_usd:.6f} USD (< $0.0001)")
    print("=" * 65)
    print("\n[SUCCESS] OpenAI configuration verified! Next recommended steps:")
    print("  1. Rebuild vector database:  python db/reindex_embeddings.py")
    print("  2. Launch web app:           streamlit run app.py")
    print("  3. Run evaluation (subset):  python -m evaluation.run_eval --limit 5")


if __name__ == "__main__":
    main()
