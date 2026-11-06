"""The one place that knows how to load the embedding model and apply
its query/document prefixes.

Every caller (indexing, retrieval, evaluation) goes through these
functions, so documents and queries are always embedded the same way.
"""
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


def load_model(local_files_only: bool = False) -> SentenceTransformer:
    model = SentenceTransformer(
        settings.embedding_model,
        local_files_only=local_files_only,
    )
    if settings.embedding_max_tokens:
        model.max_seq_length = min(
            settings.embedding_max_tokens,
            model.tokenizer.model_max_length,
        )
    return model


def embed_documents(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    prefixed = [settings.embedding_doc_prefix + t for t in texts]
    vectors = model.encode(
        prefixed,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.ascontiguousarray(vectors, dtype=np.float32)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    vector = model.encode(
        [settings.embedding_query_prefix + query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.ascontiguousarray(vector, dtype=np.float32)


def count_tokens(model: SentenceTransformer, texts: list[str]) -> list[int]:
    prefixed = [settings.embedding_doc_prefix + t for t in texts]
    return [len(ids) for ids in model.tokenizer(prefixed)["input_ids"]]
