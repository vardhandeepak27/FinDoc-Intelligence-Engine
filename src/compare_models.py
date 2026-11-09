"""Build an index for each candidate embedding model and evaluate it.

Each model runs in a separate Python process with its settings passed as
environment variables. This is the same mechanism a cloud server uses
(12-factor config): no code changes, only a different environment.

Indexes are stored per model (indexes/<model>/), and a model whose index
already matches its settings is not rebuilt, so the script can be stopped
and restarted.

Usage:
  python src\\compare_models.py                 # all candidates
  python src\\compare_models.py bge-small       # only names containing "bge-small"
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from config import settings

SRC = Path(__file__).resolve().parent

CANDIDATES = [
    {   # baseline: small, 256-token limit
        "FINDOC_EMBEDDING_MODEL": "all-MiniLM-L6-v2",
    },
    {   # same size as MiniLM, 512 tokens, trained for retrieval
        "FINDOC_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "FINDOC_EMBEDDING_QUERY_PREFIX":
            "Represent this sentence for searching relevant passages: ",
    },
    {   # bigger BGE: 768-dim vectors, ~3x slower
        "FINDOC_EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
        "FINDOC_EMBEDDING_QUERY_PREFIX":
            "Represent this sentence for searching relevant passages: ",
    },
    {   # long context (8k tokens): no truncation at all; capped for CPU speed
        "FINDOC_EMBEDDING_MODEL": "nomic-ai/modernbert-embed-base",
        "FINDOC_EMBEDDING_QUERY_PREFIX": "search_query: ",
        "FINDOC_EMBEDDING_DOC_PREFIX": "search_document: ",
        "FINDOC_EMBEDDING_MAX_TOKENS": "1536",
    },
]


def run(script: str, env_overrides: dict, *args: str) -> None:
    env = {**os.environ, **env_overrides}
    subprocess.run(
        [sys.executable, str(SRC / script), *args], env=env, check=True
    )


def index_is_current(candidate: dict) -> bool:
    """Skip rebuilding if this model's index already matches its settings."""
    slug = candidate["FINDOC_EMBEDDING_MODEL"].split("/")[-1]
    info_path = settings.index_dir / slug / "index_info.json"
    if not info_path.exists():
        return False
    info = json.loads(info_path.read_text(encoding="utf-8"))
    chunks_mtime = settings.chunks_path.stat().st_mtime
    return (
        info.get("query_prefix", "") == candidate.get("FINDOC_EMBEDDING_QUERY_PREFIX", "")
        and info.get("doc_prefix", "") == candidate.get("FINDOC_EMBEDDING_DOC_PREFIX", "")
        and info_path.stat().st_mtime > chunks_mtime  # built after latest chunking
    )


def latest_result(model_name: str) -> dict | None:
    slug = model_name.split("/")[-1]
    runs = sorted(settings.eval_results_dir.glob(f"*_{slug}_auto.json"))
    return json.loads(runs[-1].read_text(encoding="utf-8")) if runs else None


def main(name_filter: str | None) -> None:
    selected = [
        c for c in CANDIDATES
        if not name_filter or name_filter.lower() in c["FINDOC_EMBEDDING_MODEL"].lower()
    ]

    for candidate in selected:
        name = candidate["FINDOC_EMBEDDING_MODEL"]
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

        if index_is_current(candidate):
            print("Index is up to date - skipping embedding.")
        else:
            started = time.perf_counter()
            run("embed.py", candidate)
            print(f"Embedding took {(time.perf_counter() - started) / 60:.1f} min")

        run("evaluate_retrieval.py", candidate, "--filters", "auto")

    # ---- comparison table (hybrid retriever, filters worked out from the question)
    print(f"\n{'model':<34}{'trunc%':>7}{'dense@10':>10}{'hyb@1':>7}"
          f"{'hyb@10':>8}{'tbl@10':>8}{'MRR':>6}")
    print("-" * 80)
    for candidate in CANDIDATES:
        name = candidate["FINDOC_EMBEDDING_MODEL"]
        result = latest_result(name)
        if not result:
            continue
        info = result["index_info"]
        trunc = 100 * info.get("truncated_chunks", 0) / max(info.get("num_vectors", 1), 1)
        table_ranks = [
            r["rank"] for r in result["per_question"]["hybrid"] if r["type"] == "table"
        ]
        table_hit10 = sum(r is not None for r in table_ranks) / len(table_ranks)
        overall = result["overall"]
        print(
            f"{name.split('/')[-1]:<34}{trunc:>6.0f}%"
            f"{overall['dense']['hit@10']:>10.2f}"
            f"{overall['hybrid']['hit@1']:>7.2f}"
            f"{overall['hybrid']['hit@10']:>8.2f}"
            f"{table_hit10:>8.2f}"
            f"{overall['hybrid']['mrr@10']:>6.2f}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
