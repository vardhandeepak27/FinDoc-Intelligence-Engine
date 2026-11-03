import json
import logging
from datetime import datetime, timezone

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
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


def build_index():
    """Embed chunks and persist the FAISS index, metadata and build info."""
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]

    logger.info("Chunks loaded: %d", len(chunks))
    logger.info("Loading embedding model: %s", settings.embedding_model)

    model = SentenceTransformer(settings.embedding_model)

    embeddings = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
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
        "dimension": dimension,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "num_vectors": index.ntotal,
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
