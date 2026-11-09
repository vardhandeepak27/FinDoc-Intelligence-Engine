import hashlib
import json
import logging
from datetime import datetime, timezone

import faiss
import numpy as np

from config import settings
from embeddings import count_tokens, embed_documents, load_model
from logging_setup import setup_logging

logger = logging.getLogger(__name__)


def load_chunks():
    """Load chunk records from JSONL, skipping empty ones."""
    if not settings.chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {settings.chunks_path}. "
            "Run chunk.py first."
        )

    chunks = []
    skipped = 0

    with settings.chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if not record.get("text", "").strip():
                skipped += 1
                continue

            chunks.append(record)

    if skipped:
        logger.warning("Skipped %d empty chunks", skipped)

    if not chunks:
        raise ValueError("No valid chunks found.")

    return chunks


# ------------------------------------------------------------------ cache
# A vector depends only on (model, doc prefix, token limit, text). If the
# text of a chunk hasn't changed, its vector hasn't either, so we store
# vectors by a hash of their text and only embed what's new or changed.

def cache_key(text: str, max_tokens: int) -> str:
    raw = f"{settings.embedding_doc_prefix}|{max_tokens}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(max_tokens: int) -> dict[str, np.ndarray]:
    path = settings.embedding_cache_path
    if path.exists():
        data = np.load(path, allow_pickle=False)
        return dict(zip(data["keys"].tolist(), data["vectors"]))

    # First run with caching: seed the cache from the existing index, if it
    # was built with the same model, prefix and token limit.
    if not (settings.index_info_path.exists() and settings.index_path.exists()):
        return {}
    info = json.loads(settings.index_info_path.read_text(encoding="utf-8"))
    if (
        info.get("embedding_model") != settings.embedding_model
        or info.get("doc_prefix", "") != settings.embedding_doc_prefix
        or info.get("max_seq_length") != max_tokens
    ):
        return {}
    index = faiss.read_index(str(settings.index_path))
    with settings.metadata_path.open("r", encoding="utf-8") as file:
        old_chunks = [json.loads(line) for line in file if line.strip()]
    if index.ntotal != len(old_chunks):
        return {}
    vectors = index.reconstruct_n(0, index.ntotal)
    logger.info("Seeding embedding cache from the existing index (%d vectors)", index.ntotal)
    return {
        cache_key(c.get("embed_text") or c["text"], max_tokens): vectors[i]
        for i, c in enumerate(old_chunks)
    }


def save_cache(cache: dict[str, np.ndarray]) -> None:
    keys = np.array(list(cache.keys()))
    vectors = np.stack(list(cache.values())).astype(np.float32)
    np.savez(settings.embedding_cache_path, keys=keys, vectors=vectors)


def build_index():
    """Embed chunks and persist the FAISS index, metadata and build info."""
    settings.model_index_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    # Embed the chunk WITH its context line, so a table of numbers carries
    # "Wipro | Standalone | Statement of Profit and Loss | ₹ million".
    texts = [chunk.get("embed_text") or chunk["text"] for chunk in chunks]

    logger.info("Chunks loaded: %d", len(chunks))
    logger.info("Loading embedding model: %s", settings.embedding_model)

    model = load_model()

    # The model silently truncates anything longer than max_seq_length
    # tokens: the end of the chunk is never embedded. Measure it.
    max_tokens = model.max_seq_length
    token_counts = count_tokens(model, texts)
    truncated = sum(n > max_tokens for n in token_counts)
    sorted_counts = sorted(token_counts)
    logger.info(
        "Tokens per chunk: median %d, p95 %d, max %d (model limit %d)",
        sorted_counts[len(sorted_counts) // 2],
        sorted_counts[int(len(sorted_counts) * 0.95)],
        sorted_counts[-1], max_tokens,
    )
    if truncated:
        logger.warning(
            "%d of %d chunks (%.0f%%) exceed the model's %d-token limit and "
            "will be truncated when embedded.",
            truncated, len(texts), 100 * truncated / len(texts), max_tokens,
        )

    cache = load_cache(max_tokens)
    keys = [cache_key(t, max_tokens) for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    logger.info(
        "Embedding cache: %d reused, %d to embed", len(texts) - len(todo), len(todo)
    )
    if todo:
        new_vectors = embed_documents(model, [texts[i] for i in todo])
        for i, vector in zip(todo, new_vectors):
            cache[keys[i]] = vector

    embeddings = np.ascontiguousarray(np.stack([cache[k] for k in keys]))
    # Keep only vectors still in use, so the cache doesn't grow forever
    save_cache({k: cache[k] for k in keys})

    dimension = embeddings.shape[1]

    # Inner product of normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(settings.index_path))

    # Preserve the exact ordering used by the FAISS vectors
    with settings.metadata_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Record HOW this index was built, so retrieval can refuse to use
    # an index that doesn't match the current configuration.
    build_info = {
        "embedding_model": settings.embedding_model,
        "query_prefix": settings.embedding_query_prefix,
        "doc_prefix": settings.embedding_doc_prefix,
        "dimension": dimension,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "num_vectors": index.ntotal,
        "chunker_version": chunks[0].get("chunker_version"),
        "max_seq_length": max_tokens,
        "truncated_chunks": truncated,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    settings.index_info_path.write_text(
        json.dumps(build_info, indent=2), encoding="utf-8"
    )

    logger.info(
        "Indexing complete: %d vectors, %d dimensions",
        index.ntotal, dimension,
    )
    logger.info("FAISS index: %s", settings.index_path)
    logger.info("Build info: %s", settings.index_info_path)


if __name__ == "__main__":
    setup_logging()
    build_index()
