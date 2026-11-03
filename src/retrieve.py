import json
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from logging_setup import setup_logging

logger = logging.getLogger(__name__)


def load_metadata():
    """Load chunk records in FAISS vector order."""
    if not settings.metadata_path.exists():
        raise FileNotFoundError(
            "Chunk metadata not found. Run embed.py first."
        )

    records = []

    with settings.metadata_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def check_index_matches_config():
    """Fail fast if the index was built with a different embedding model."""
    if not settings.index_info_path.exists():
        logger.warning(
            "No %s found; can't verify which model built this index. "
            "Re-run embed.py to create it.",
            settings.index_info_path.name,
        )
        return

    info = json.loads(settings.index_info_path.read_text(encoding="utf-8"))

    if info["embedding_model"] != settings.embedding_model:
        raise ValueError(
            f"Index was built with '{info['embedding_model']}' but config "
            f"says '{settings.embedding_model}'. Re-run embed.py."
        )

    if (
        info["chunk_size"] != settings.chunk_size
        or info["chunk_overlap"] != settings.chunk_overlap
    ):
        logger.warning(
            "Index was built with chunk_size=%s/overlap=%s but config says "
            "%s/%s. Re-run chunk.py and embed.py to apply the new values.",
            info["chunk_size"], info["chunk_overlap"],
            settings.chunk_size, settings.chunk_overlap,
        )


def load_retriever():
    """Load the persisted index, metadata, and embedding model."""
    if not settings.index_path.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run embed.py first."
        )

    check_index_matches_config()

    index = faiss.read_index(str(settings.index_path))
    metadata = load_metadata()

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index has {index.ntotal} vectors but "
            f"{len(metadata)} metadata records."
        )

    model = SentenceTransformer(
        settings.embedding_model,
        local_files_only=True,
    )

    logger.info("Retriever loaded: %d vectors", index.ntotal)

    return model, index, metadata


def retrieve(
    query,
    model,
    index,
    metadata,
    top_k=None,
    company=None,
    period=None,
):
    """Retrieve relevant chunks with optional metadata filters."""
    top_k = top_k or settings.top_k

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if index.ntotal == 0:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

    # Exact search over the complete index.
    # Filtering afterwards ensures we don't miss matching
    # documents just because other companies ranked higher.
    scores, indices = index.search(query_embedding, index.ntotal)

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
    setup_logging()

    model, index, metadata = load_retriever()

    query = (
        "Wipro consolidated revenue from operations "
        "year ended March 31 2026"
    )

    results = retrieve(
        query=query,
        model=model,
        index=index,
        metadata=metadata,
        company="Wipro",
        period="FY2025-26",
    )

    # print() is right here: this is the program's OUTPUT for a human,
    # not a diagnostic message.
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
