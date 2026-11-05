"""Measure retrieval quality against a hand-labelled question set.

For each question in data/eval/retrieval_eval.jsonl we know which PDF
page(s) contain the answer. We run three retrievers (keyword only,
semantic only, hybrid) and record where the first correct page appears.

Metrics:
  Hit@k  - share of questions with a correct page in the top k results
  MRR@10 - mean of 1/rank of the first correct result (0 if not in top 10)

Each run is saved with the index's build info, so runs made with
different models or chunkers can be compared later.

Usage:
  python src\\evaluate_retrieval.py              # no company filter
  python src\\evaluate_retrieval.py --filter     # filter by the question's company
"""
import argparse
import json
import logging
from datetime import datetime

import numpy as np

from config import settings
from hybrid_retrieve import hybrid_retrieve, load_data, tokenize
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

TOP_K = 10
K_VALUES = (1, 3, 5, 10)


def load_questions() -> list[dict]:
    with settings.eval_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# --------------------------------------------------------------- rankers
# Each ranker returns chunk indices, best first, after the company filter.

def rank_bm25(question, metadata, index, model, bm25, company):
    scores = bm25.get_scores(tokenize(question))
    order = np.argsort(-scores)
    return [int(i) for i in order if _keep(metadata[i], company)][:TOP_K]


def rank_dense(question, metadata, index, model, bm25, company):
    vector = model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    # Search everything, then filter, so filtering never hides a match
    _, indices = index.search(vector, index.ntotal)
    return [int(i) for i in indices[0] if i >= 0 and _keep(metadata[i], company)][:TOP_K]


def rank_hybrid(question, metadata, index, model, bm25, company):
    chunk_position = {m["chunk_id"]: i for i, m in enumerate(metadata)}
    results = hybrid_retrieve(
        question, metadata, index, model, bm25, top_k=TOP_K, company=company
    )
    return [chunk_position[r["chunk_id"]] for r in results]


def _keep(meta: dict, company: str | None) -> bool:
    return company is None or meta["company"].lower() == company.lower()


RANKERS = {"bm25": rank_bm25, "dense": rank_dense, "hybrid": rank_hybrid}


# --------------------------------------------------------------- scoring

def first_hit_rank(ranked: list[int], metadata: list[dict], question: dict) -> int | None:
    """1-based rank of the first chunk from a correct page, else None."""
    expected = set(question["expected_pages"])
    for rank, i in enumerate(ranked, start=1):
        meta = metadata[i]
        if (
            meta["document_name"] == question["document_name"]
            and meta["page_number"] in expected
        ):
            return rank
    return None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    summary = {
        f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n
        for k in K_VALUES
    }
    summary["mrr@10"] = sum(1 / r for r in ranks if r) / n
    return summary


def print_table(title: str, rows: dict[str, dict]) -> None:
    print(f"\n{title}")
    header = f"{'retriever':<10}" + "".join(f"{m:>9}" for m in rows[next(iter(rows))])
    print(header)
    print("-" * len(header))
    for name, summary in rows.items():
        print(f"{name:<10}" + "".join(f"{v:>9.2f}" for v in summary.values()))


# ------------------------------------------------------------------ main

def evaluate(use_filter: bool) -> None:
    questions = load_questions()
    metadata, index, model, bm25 = load_data()

    per_question = {name: [] for name in RANKERS}

    for question in questions:
        company = question["company"] if use_filter else None
        for name, ranker in RANKERS.items():
            ranked = ranker(question["question"], metadata, index, model, bm25, company)
            rank = first_hit_rank(ranked, metadata, question)
            top = metadata[ranked[0]] if ranked else {}
            per_question[name].append({
                "id": question["id"],
                "type": question["type"],
                "rank": rank,
                "top_result": f"{top.get('company', '')} p{top.get('page_number', '')}",
            })

    # ---- overall and by question type
    overall = {
        name: summarize([r["rank"] for r in rows])
        for name, rows in per_question.items()
    }
    print_table(
        f"Retrieval quality on {len(questions)} questions "
        f"({'with' if use_filter else 'without'} company filter)",
        overall,
    )
    for qtype in sorted({q["type"] for q in questions}):
        print_table(
            f"Only '{qtype}' questions",
            {
                name: summarize([r["rank"] for r in rows if r["type"] == qtype])
                for name, rows in per_question.items()
            },
        )

    # ---- what hybrid got wrong: these are the cases to study
    print("\nHybrid misses (correct page not in top 10):")
    by_id = {q["id"]: q for q in questions}
    for row in per_question["hybrid"]:
        if row["rank"] is None:
            q = by_id[row["id"]]
            print(
                f"  {row['id']}  {q['question']}\n"
                f"        expected p{q['expected_pages']} | top result: {row['top_result']}"
            )

    # ---- save the run
    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    index_info = (
        json.loads(settings.index_info_path.read_text(encoding="utf-8"))
        if settings.index_info_path.exists() else {}
    )
    run = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "company_filter": use_filter,
        "index_info": index_info,
        "overall": overall,
        "per_question": per_question,
    }
    model_tag = index_info.get("embedding_model", "unknown").split("/")[-1]
    out_path = settings.eval_results_dir / (
        f"{datetime.now():%Y%m%d-%H%M%S}_{model_tag}"
        f"{'_filtered' if use_filter else ''}.json"
    )
    out_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    logger.info("Saved run: %s", out_path)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", action="store_true",
        help="restrict results to the question's company",
    )
    evaluate(use_filter=parser.parse_args().filter)
