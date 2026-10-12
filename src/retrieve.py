import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = PROJECT_ROOT / "indexes" / "financial.faiss"
METADATA_PATH = PROJECT_ROOT / "indexes" / "chunk_metadata.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_metadata():
    """Load chunk records in FAISS vector order."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Chunk metadata not found. Run embed.py first."
        )

    records = []

    with METADATA_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_retriever():
    """Load the persisted index, metadata, and embedding model."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run embed.py first."
        )

    index = faiss.read_index(str(INDEX_PATH))
    metadata = load_metadata()

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index has {index.ntotal} vectors but "
            f"{len(metadata)} metadata records."
        )

    model = SentenceTransformer(
    MODEL_NAME,
    local_files_only=True,
)

    return model, index, metadata


def retrieve(
    query,
    model,
    index,
    metadata,
    top_k=5,
    company=None,
    period=None,
):
    """Retrieve relevant chunks with optional metadata filters."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if index.ntotal == 0:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.ascontiguousarray(
        query_embedding, dtype=np.float32
    )

    # Exact search over the complete index.
    # Filtering afterwards ensures we don't miss matching
    # documents just because other companies ranked higher.
    scores, indices = index.search(
        query_embedding, index.ntotal
    )

    results = []

    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue

        chunk = metadata[int(idx)]

        if company and chunk["company"].casefold() != company.casefold():
            continue

        if period and chunk["period"].casefold() != period.casefold():
            continue

        results.append({
            "rank": len(results) + 1,
            "similarity": float(score),
            "chunk_id": chunk["chunk_id"],
            "company": chunk["company"],
            "period": chunk["period"],
            "section": chunk.get("section", "Unknown"),
            "page_number": chunk["page_number"],
            "document_name": chunk["document_name"],
            "source_url": chunk["source_url"],
            "text": chunk["text"],
        })

        if len(results) >= top_k:
            break

    return results



if __name__ == "__main__":
    model, index, metadata = load_retriever()

    print(f"Loaded vectors: {index.ntotal}")

    query = (
    "Wipro consolidated revenue from operations "
    "year ended March 31 2026"
)

    results = retrieve(
    query=query,
    model=model,
    index=index,
    metadata=metadata,
    top_k=10,
    company="Wipro",
    period="FY2025-26",
)

    print(f"\nQuery: {query}\n")

    for result in results:
        print(
            f"Rank: {result['rank']} | "
            f"Similarity: {result['similarity']:.4f}"
        )
        print(
            f"Company: {result['company']} | "
            f"Period: {result['period']}"
        )
        print(
            f"Section: {result['section']} | "
            f"Page: {result['page_number']}"
        )
        print(f"Chunk ID: {result['chunk_id']}")
        print(result["text"])
        print("-" * 70)
