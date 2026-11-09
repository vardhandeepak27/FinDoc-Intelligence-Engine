"""Second-stage ranking with a cross-encoder.

Bi-encoder (our embedding model): question and chunk are embedded
SEPARATELY, then compared. Fast enough to search every chunk, but the
model never sees the two texts together.

Cross-encoder (this file): question and chunk are read TOGETHER as one
input, and the model outputs a relevance score. Much more accurate,
much slower, so we only run it on the top candidates from hybrid search:

    hybrid search (6,264 chunks) -> top N candidates -> cross-encoder -> top k
"""
import logging
import time
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def load_reranker(model_name: str | None = None, max_tokens: int | None = None) -> CrossEncoder:
    name = model_name or settings.reranker_model
    logger.info("Loading reranker: %s", name)
    return CrossEncoder(
        name,
        max_length=max_tokens or settings.rerank_max_tokens,
        # Some models ship in float16 (half precision). That's fast on a
        # GPU but very slow on most CPUs, so on CPU we force float32.
        model_kwargs={"dtype": torch.float32},
    )


def rerank(
    query: str,
    results: list[dict],
    top_k: int | None = None,
    model_name: str | None = None,
    max_tokens: int | None = None,
) -> list[dict]:
    """Re-order retrieval results by cross-encoder relevance."""
    if not results:
        return results

    model = load_reranker(model_name, max_tokens)
    # The reranker sees the context line too, so it knows the company,
    # basis and unit of a table that is only numbers.
    pairs = [(query, f"{r.get('context', '')}\n\n{r['text']}") for r in results]

    started = time.perf_counter()
    scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
    logger.debug("Reranked %d candidates in %.2fs", len(pairs), time.perf_counter() - started)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    ranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)
    return ranked[: top_k or settings.top_k]
