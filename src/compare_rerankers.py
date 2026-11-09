"""Compare cross-encoder rerankers on the evaluation set.

For each question we run hybrid retrieval ONCE to get the top-N
candidates, then let every reranker re-order those same candidates.
Same candidates for everyone = a fair comparison.

Reported per reranker:
  hit@k / MRR   - answer-level quality of the final top 10
  sec/query     - added latency (paid on EVERY user question)

Also reported:
  "no rerank"   - hybrid order, the baseline
  "ceiling"     - share of questions whose answer is anywhere in the N
                  candidates. No reranker can beat this: it can only
                  re-order what retrieval found.

Usage:
  python src\\compare_rerankers.py
  python src\\compare_rerankers.py --candidates 50
"""
import argparse
import json
import logging
import time
from datetime import datetime

from config import settings
from evaluate_retrieval import (
    build_filters, check_labels, first_hit_rank, load_questions, summarize,
)
from hybrid_retrieve import hybrid_retrieve, load_data
from logging_setup import setup_logging
from rerank import rerank

logger = logging.getLogger(__name__)

TOP_K = 10

CANDIDATES = [
    # (model, max tokens, note)
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", 512, "22M params, very fast"),
    ("mixedbread-ai/mxbai-rerank-xsmall-v1", 512, "70M params"),
    ("BAAI/bge-reranker-base", 512, "278M params, strong, slower"),
    ("Alibaba-NLP/gte-reranker-modernbert-base", 1024, "149M, long context"),
]


def main(n_candidates: int) -> None:
    questions = load_questions()
    metadata, index, model, bm25 = load_data()
    check_labels(questions, metadata)

    # ---- stage 1: the same candidates for every reranker
    candidates = {}
    for q in questions:
        f = build_filters(q, "auto")
        candidates[q["id"]] = (f["query"], hybrid_retrieve(
            f["query"], metadata, index, model, bm25,
            top_k=n_candidates, company=f["company"], basis=f["basis"],
            keyword_query=f["keyword_query"],
        ))

    def ranks_for(order_fn) -> list[int | None]:
        out = []
        for q in questions:
            query, results = candidates[q["id"]]
            ordered = order_fn(query, results)
            out.append(first_hit_rank(
                [r["chunk_index"] for r in ordered[:TOP_K]], metadata, q
            ))
        return out

    table = {}
    types = [q["type"] for q in questions]

    # ---- baseline and ceiling
    table["no rerank (hybrid)"] = (ranks_for(lambda q, r: r), 0.0)
    ceiling = [
        first_hit_rank([r["chunk_index"] for r in candidates[q["id"]][1]], metadata, q)
        for q in questions
    ]

    # ---- each reranker
    for name, max_tokens, note in CANDIDATES:
        print(f"\nRunning {name} ({note})...")
        try:
            rerank("warm up", [{"text": "x", "context": ""}], model_name=name,
                   max_tokens=max_tokens)  # load + first-call overhead
        except Exception as error:  # e.g. model not downloadable
            print(f"  skipped: {error}")
            continue
        started = time.perf_counter()
        ranks = ranks_for(lambda q, r: rerank(
            q, [dict(x) for x in r], top_k=TOP_K, model_name=name,
            max_tokens=max_tokens,
        ))
        per_query = (time.perf_counter() - started) / len(questions)
        table[name.split("/")[-1]] = (ranks, per_query)
        s = summarize(ranks)
        print(f"  done: hit@1 {s['hit@1']:.2f}, hit@10 {s['hit@10']:.2f}, "
              f"{per_query:.2f} s/query")

    # ---- report
    found = sum(r is not None for r in ceiling) / len(ceiling)
    print(f"\nCeiling: answer is somewhere in the top {n_candidates} "
          f"candidates for {found:.2f} of questions\n")
    header = (f"{'reranker':<36}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}"
              f"{'hit@10':>8}{'MRR':>6}{'tbl@1':>7}{'sec/q':>7}")
    print(header)
    print("-" * len(header))
    for name, (ranks, latency) in table.items():
        s = summarize(ranks)
        table_ranks = [r for r, t in zip(ranks, types) if t == "table"]
        tbl1 = sum(r == 1 for r in table_ranks) / len(table_ranks)
        print(f"{name:<36}{s['hit@1']:>7.2f}{s['hit@3']:>7.2f}{s['hit@5']:>7.2f}"
              f"{s['hit@10']:>8.2f}{s['mrr@10']:>6.2f}{tbl1:>7.2f}{latency:>7.2f}")

    # ---- save
    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    out = settings.eval_results_dir / f"{datetime.now():%Y%m%d-%H%M%S}_rerankers.json"
    out.write_text(json.dumps({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": settings.embedding_model,
        "candidates": n_candidates,
        "default_basis": settings.default_basis,
        "ceiling_hit": found,
        "results": {
            name: {"summary": summarize(ranks), "sec_per_query": latency,
                   "ranks": dict(zip([q["id"] for q in questions], ranks))}
            for name, (ranks, latency) in table.items()
        },
    }, indent=2), encoding="utf-8")
    logger.info("Saved: %s", out)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=settings.rerank_candidates)
    main(parser.parse_args().candidates)
