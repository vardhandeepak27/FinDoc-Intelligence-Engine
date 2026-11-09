import logging
import re

import numpy as np

from rank_bm25 import BM25Okapi

from config import settings
from embeddings import embed_query
from logging_setup import setup_logging
from query_understanding import understand
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


def allowed_mask(metadata, company=None, basis=None) -> np.ndarray:
    """Which chunks may be returned.

    company: hard filter (must match).
    basis:   keep chunks with the same basis OR with no known basis. Many
             relevant pages (e.g. management commentary) never say
             "standalone" or "consolidated", so excluding unknowns would
             throw away good answers. We only exclude contradictions.
    """
    keep = np.ones(len(metadata), dtype=bool)
    for i, meta in enumerate(metadata):
        if company and meta.get("company", "").lower() != company.lower():
            keep[i] = False
        elif basis and meta.get("basis") not in (None, basis):
            keep[i] = False
    return keep


def hybrid_retrieve(
    query,
    metadata,
    index,
    model,
    bm25,
    top_k=None,
    company=None,
    basis=None,
    keyword_query=None,
    candidate_k=None,
    rrf_k=None,
):
    """BM25 + dense retrieval merged with weighted Reciprocal Rank Fusion.

    Filters are applied BEFORE choosing each retriever's candidates
    (pre-filtering). Post-filtering a global top-50 can leave almost
    nothing when the requested company is rare in that top-50.
    """
    top_k = top_k or settings.top_k
    candidate_k = candidate_k or settings.candidate_k
    rrf_k = rrf_k or settings.rrf_k
    keep = allowed_mask(metadata, company, basis)

    # 1. Keyword retrieval (on the alias-expanded query, if given)
    bm25_scores = bm25.get_scores(tokenize(keyword_query or query))
    bm25_scores = np.where(keep, bm25_scores, -np.inf)
    keyword_ranking = [
        int(i) for i in np.argsort(-bm25_scores)[:candidate_k] if keep[i]
    ]

    # 2. Semantic retrieval: search everything (exact, ~6k vectors), then
    #    keep the allowed chunks in rank order
    query_vector = embed_query(model, query)
    _, dense_indices = index.search(query_vector, index.ntotal)
    dense_ranking = [
        int(i) for i in dense_indices[0] if i >= 0 and keep[i]
    ][:candidate_k]

    # 3. Weighted Reciprocal Rank Fusion
    fused_scores = {}
    for ranking, weight in (
        (keyword_ranking, settings.bm25_weight),
        (dense_ranking, settings.dense_weight),
    ):
        for rank, idx in enumerate(ranking, start=1):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + weight / (rrf_k + rank)

    ranked = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

    return [
        {
            "rrf_score": fused_scores[idx],
            "chunk_index": idx,
            "chunk_id": metadata[idx].get("chunk_id"),
            "company": metadata[idx].get("company"),
            "period": metadata[idx].get("period"),
            "section": metadata[idx].get("section"),
            "context": metadata[idx].get("context", ""),
            "basis": metadata[idx].get("basis"),
            "unit": metadata[idx].get("unit"),
            "currency": metadata[idx].get("currency"),
            "chunk_type": metadata[idx].get("chunk_type"),
            "page_number": metadata[idx].get("page_number"),
            "printed_page": metadata[idx].get("printed_page"),
            "document_name": metadata[idx].get("document_name"),
            "text": metadata[idx].get("text", ""),
        }
        for idx in ranked
    ]


def search(query, metadata, index, model, bm25, top_k=None):
    """Full retrieval for a raw user question: understand, filter, retrieve."""
    intent = understand(query)
    logger.info(
        "Query intent: company=%s basis=%s", intent.company, intent.basis
    )
    return intent, hybrid_retrieve(
        query, metadata, index, model, bm25,
        top_k=top_k,
        company=intent.company,
        basis=intent.basis,
        keyword_query=intent.keyword_query,
    )


if __name__ == "__main__":
    import sys

    setup_logging()
    metadata, index, model, bm25 = load_data()

    query = " ".join(sys.argv[1:]) or (
        "What was Wipro's consolidated revenue from operations for FY2025-26?"
    )
    intent, results = search(query, metadata, index, model, bm25)

    print(f"\nQuery: {query}")
    print(f"Understood: company={intent.company} basis={intent.basis}\n")

    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank} | RRF score: {result['rrf_score']:.5f}")
        print("Context:", result["context"])
        print("PDF page:", result["page_number"], "| Printed page:", result["printed_page"])
        print(result["text"][:600])
        print("-" * 70)
