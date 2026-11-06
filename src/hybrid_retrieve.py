import logging
import re

from rank_bm25 import BM25Okapi

from config import settings
from embeddings import embed_query
from logging_setup import setup_logging
from retrieve import load_retriever

logger = logging.getLogger(__name__)


def tokenize(text):
    return re.findall(r"[a-zA-Z]+|\d[\d,]*", text.lower())


def load_data():
    """Load the dense retriever and build a BM25 index over the same chunks.

    BM25 is built from the FAISS metadata (not chunks.jsonl), so position i
    always refers to the same chunk in both indexes. Reading two separate
    files risks them drifting apart, e.g. when embed.py skips empty chunks.
    """
    model, index, metadata = load_retriever()

    # Keyword index over the same text we embed (context line included),
    # so "standalone" or "crore" in a query can match the context too.
    bm25 = BM25Okapi([
        tokenize(chunk.get("embed_text") or chunk.get("text", ""))
        for chunk in metadata
    ])

    logger.info("BM25 index built over %d chunks", len(metadata))

    return metadata, index, model, bm25


def hybrid_retrieve(
    query,
    metadata,
    index,
    model,
    bm25,
    top_k=None,
    company=None,
    period=None,
    candidate_k=None,
    rrf_k=None,
):
    top_k = top_k or settings.top_k
    candidate_k = candidate_k or settings.candidate_k
    rrf_k = rrf_k or settings.rrf_k

    query_tokens = tokenize(query)

    # 1. Keyword retrieval
    bm25_scores = bm25.get_scores(query_tokens)
    keyword_ranking = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:candidate_k]

    # 2. Semantic retrieval
    query_vector = embed_query(model, query)

    _, dense_indices = index.search(
        query_vector,
        min(candidate_k, index.ntotal),
    )

    dense_ranking = [int(i) for i in dense_indices[0] if i >= 0]

    # 3. Reciprocal Rank Fusion
    fused_scores = {}

    for ranking in (keyword_ranking, dense_ranking):
        for rank, idx in enumerate(ranking, start=1):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + (
                1.0 / (rrf_k + rank)
            )

    ranked_indices = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    # 4. Apply metadata filters
    results = []

    for idx in ranked_indices:
        meta = metadata[idx]

        if company and meta.get("company", "").lower() != company.lower():
            continue

        if period and meta.get("period", "").lower() != period.lower():
            continue

        results.append({
            "rrf_score": fused_scores[idx],
            "chunk_id": meta.get("chunk_id"),
            "company": meta.get("company"),
            "period": meta.get("period"),
            "section": meta.get("section"),
            "context": meta.get("context", ""),
            "basis": meta.get("basis"),
            "unit": meta.get("unit"),
            "currency": meta.get("currency"),
            "chunk_type": meta.get("chunk_type"),
            "page_number": meta.get("page_number"),
            "printed_page": meta.get("printed_page"),
            "document_name": meta.get("document_name"),
            "text": meta.get("text", ""),
        })

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    setup_logging()

    metadata, index, model, bm25 = load_data()

    query = (
        "Wipro consolidated revenue from operations "
        "year ended March 31 2026"
    )

    results = hybrid_retrieve(
        query=query,
        metadata=metadata,
        index=index,
        model=model,
        bm25=bm25,
        company="Wipro",
        period="FY2025-26",
    )

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank} | RRF score: {result['rrf_score']:.5f}")
        print("Context:", result["context"])
        print("PDF page:", result["page_number"], "| Printed page:", result["printed_page"])
        print("Chunk ID:", result["chunk_id"])
        print(result["text"])
        print("-" * 70)
