import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "indexes"
INDEX_PATH = INDEX_DIR / "financial.faiss"
METADATA_PATH = INDEX_DIR / "chunk_metadata.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks():
    """Load chunk records from JSONL."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_PATH}. "
            "Run chunk.py first."
        )

    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)

                if not record.get("text", "").strip():
                    continue

                chunks.append(record)

    if not chunks:
        raise ValueError("No valid chunks found.")

    return chunks


def build_index():
    """Embed chunks and persist the FAISS index and metadata."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]

    print(f"Chunks loaded: {len(chunks)}")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.ascontiguousarray(
        embeddings, dtype=np.float32
    )

    dimension = embeddings.shape[1]

    # Inner product of normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    # Preserve the exact ordering used by the FAISS vectors
    with METADATA_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print("\nIndexing complete")
    print(f"Chunks indexed: {index.ntotal}")
    print(f"Embedding dimensions: {dimension}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"FAISS index: {INDEX_PATH}")
    print(f"Chunk metadata: {METADATA_PATH}")


if __name__ == "__main__":
    build_index()
