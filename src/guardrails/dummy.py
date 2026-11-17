
"""Dummy test for the hybrid Text-to-SQL retriever."""

import logging

from src.retrieval.retriever import retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

QUESTION = "Which carriers have delivered more than 500 packages?"

REQUIRED_COLUMNS = {
    "carrier",
    "shipment_status",
    "shipment_id",
}

def main():
    print("\n" + "=" * 80)
    print("HYBRID RETRIEVER DUMMY TEST")
    print("=" * 80)

    print(f"\nQuestion: {QUESTION}")

    # Run retrieval
    chunks = retriever.retrieve(QUESTION)

    print(f"\nTotal retrieved chunks: {len(chunks)}")

    print("\n" + "-" * 80)
    print("RETRIEVED DOCUMENTS")
    print("-" * 80)

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n[{index}] ID: {chunk.id}")
        print(f"    Type: {chunk.doc_type}")
        print(f"    Title: {chunk.title}")
        print(f"    Content:\n{chunk.content}")

    # Check whether shipments schema was retrieved
    shipments_chunks = [
        chunk for chunk in chunks
        if chunk.doc_type == "table"
        and chunk.metadata.get("table_name") == "shipments"
    ]

    print("\n" + "-" * 80)
    print("SCHEMA COVERAGE CHECK")
    print("-" * 80)

    if not shipments_chunks:
        print("FAIL: shipments table schema was not retrieved.")
        return

    shipments_text = " ".join(
        chunk.content.lower() for chunk in shipments_chunks
    )

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in shipments_text
    ]

    if missing_columns:
        print(f"FAIL: Missing columns: {missing_columns}")
    else:
        print("PASS: All required shipments columns were retrieved.")

    # Check for relevant examples
    relevant_examples = [
        chunk for chunk in chunks
        if chunk.doc_type == "example"
        and "shipment" in chunk.content.lower()
    ]

    print(
        f"\nRelevant shipment examples retrieved: "
        f"{len(relevant_examples)}"
    )

    # Summary
    print("\n" + "=" * 80)
    print("RETRIEVAL TEST SUMMARY")
    print("=" * 80)

    print(f"Total chunks: {len(chunks)}")
    print(f"Shipments schema: {'PASS' if shipments_chunks else 'FAIL'}")
    print(
        "Required columns: "
        f"{'PASS' if not missing_columns else 'FAIL'}"
    )
    print(
        "Relevant examples: "
        f"{'PASS' if relevant_examples else 'WARNING - None found'}"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
