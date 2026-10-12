
import json
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_PATH = ROOT / "indexes" / "financial.faiss"
METADATA_PATH = ROOT / "indexes" / "chunk_metadata.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"


def tokenize(text):
    return re.findall(r"[a-zA-Z]+|\d[\d,]*", text.lower())


def load_data():
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]

    index = faiss.read_index(str(INDEX_PATH))

    if len(chunks) != len(metadata) or index.ntotal != len(metadata):
        raise ValueError(
            "Chunks, metadata, and FAISS index are misaligned."
        )

    model = SentenceTransformer(
        MODEL_NAME,
        local_files_only=True,
    )

    bm25 = BM25Okapi([
        tokenize(chunk.get("text", ""))
        for chunk in chunks
    ])

    return chunks, metadata, index, model, bm25


def hybrid_retrieve(
    query,
    chunks,
    metadata,
    index,
    model,
    bm25,
    top_k=10,
    company=None,
    period=None,
    candidate_k=50,
    rrf_k=60,
):
    query_tokens = tokenize(query)

    # 1. Keyword retrieval
    bm25_scores = bm25.get_scores(query_tokens)
    keyword_ranking = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:candidate_k]

    # 2. Semantic retrieval
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    _, dense_indices = index.search(
        query_vector,
        min(candidate_k, index.ntotal),
    )

    dense_ranking = [
        int(i) for i in dense_indices[0] if i >= 0
    ]

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
            "page_number": meta.get("page_number"),
            "document_name": meta.get("document_name"),
            "text": chunks[idx].get("text", ""),
        })

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    chunks, metadata, index, model, bm25 = load_data()

    query = (
        "Wipro consolidated revenue from operations "
        "year ended March 31 2026"
    )

    results = hybrid_retrieve(
        query=query,
        chunks=chunks,
        metadata=metadata,
        index=index,
        model=model,
        bm25=bm25,
        top_k=10,
        company="Wipro",
        period="FY2025-26",
    )

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank: {rank} | "
            f"RRF score: {result['rrf_score']:.5f}"
        )
        print("Page:", result["page_number"])
        print("Section:", result["section"])
        print("Chunk ID:", result["chunk_id"])
        print(result["text"])
        print("-" * 70)
