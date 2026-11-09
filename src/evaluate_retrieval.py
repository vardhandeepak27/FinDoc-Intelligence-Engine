"""Measure retrieval quality against a hand-labelled question set.

For each question in data/eval/retrieval_eval.jsonl we know which PDF
page(s) contain the answer. We run three retrievers (keyword only,
semantic only, hybrid) and record where the first correct page appears.

A result is correct when it comes from a labelled page AND its text
contains one of the labelled answer strings (e.g. "926,240"). Questions
without answer strings fall back to page-level matching.

Metrics:
  Hit@k  - share of questions with a correct result in the top k
  MRR@10 - mean of 1/rank of the first correct result (0 if not in top 10)

Each run is saved with the index's build info, so runs made with
different models or chunkers can be compared later.

Filter modes:
  none    no filters at all
  oracle  the question's labelled company (an upper bound: perfect
          query understanding, which real users don't give us)
  auto    company + basis worked out from the question text
          (what the real system does)

Usage:
  python src\\evaluate_retrieval.py --filters auto
  python src\\evaluate_retrieval.py --filters oracle
"""
import argparse
import json
import logging
import re
from datetime import datetime

import numpy as np

from config import settings
from embeddings import embed_query
from hybrid_retrieve import allowed_mask, hybrid_retrieve, load_data, tokenize
from logging_setup import setup_logging
from query_understanding import understand

logger = logging.getLogger(__name__)

TOP_K = 10
K_VALUES = (1, 3, 5, 10)


def load_questions() -> list[dict]:
    with settings.eval_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# --------------------------------------------------------------- rankers
# Each ranker returns chunk indices, best first, after filtering.

def rank_bm25(f, metadata, index, model, bm25):
    keep = allowed_mask(metadata, f["company"], f["basis"])
    scores = bm25.get_scores(tokenize(f["keyword_query"]))
    return [int(i) for i in np.argsort(-scores) if keep[i]][:TOP_K]


def rank_dense(f, metadata, index, model, bm25):
    keep = allowed_mask(metadata, f["company"], f["basis"])
    _, indices = index.search(embed_query(model, f["query"]), index.ntotal)
    return [int(i) for i in indices[0] if i >= 0 and keep[i]][:TOP_K]


def rank_hybrid(f, metadata, index, model, bm25):
    results = hybrid_retrieve(
        f["query"], metadata, index, model, bm25, top_k=TOP_K,
        company=f["company"], basis=f["basis"], keyword_query=f["keyword_query"],
    )
    return [r["chunk_index"] for r in results]


def build_filters(question: dict, mode: str) -> dict:
    query = question["question"]
    if mode == "auto":
        intent = understand(query)
        return {"query": query, "company": intent.company,
                "basis": intent.basis, "keyword_query": intent.keyword_query}
    return {
        "query": query,
        "company": question["company"] if mode == "oracle" else None,
        "basis": None,
        "keyword_query": query,
    }


RANKERS = {"bm25": rank_bm25, "dense": rank_dense, "hybrid": rank_hybrid}


# --------------------------------------------------------------- scoring

def normalize(text: str) -> str:
    """Make answer matching robust to formatting: '**41, 527**' == '41,527'."""
    text = text.lower().replace("\u00a0", " ")
    return re.sub(r"[\s*_`|]+", "", text)


def is_hit(meta: dict, question: dict, strict: bool) -> bool:
    """A result counts if it comes from a correct page AND, in strict mode,
    the chunk itself contains the answer.

    Page-only matching over-counts: a page can hold several chunks, and
    retrieving the footnote chunk of the right page doesn't give the LLM
    the number it needs.
    """
    if (
        meta["document_name"] != question["document_name"]
        or meta["page_number"] not in question["expected_pages"]
    ):
        return False
    answers = question.get("answer_contains") or []
    if not strict or not answers:
        return True  # no answer string labelled: fall back to page level
    text = normalize(meta.get("text", ""))
    return any(normalize(a) in text for a in answers)


def first_hit_rank(ranked, metadata, question, strict=True) -> int | None:
    """1-based rank of the first correct result, else None."""
    for rank, i in enumerate(ranked, start=1):
        if is_hit(metadata[i], question, strict):
            return rank
    return None


def check_labels(questions: list[dict], metadata: list[dict]) -> None:
    """Warn about questions that can never score: no chunk on a labelled
    page contains the labelled answer. That's a label bug (or a chunker
    change that split the answer), not a retrieval failure."""
    for question in questions:
        if not any(is_hit(meta, question, strict=True) for meta in metadata):
            logger.warning(
                "Label check: no chunk on pages %s contains %s (question %s)",
                question["expected_pages"], question.get("answer_contains"),
                question["id"],
            )


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

def evaluate(mode: str) -> None:
    questions = load_questions()
    metadata, index, model, bm25 = load_data()
    check_labels(questions, metadata)

    per_question = {name: [] for name in RANKERS}

    intent_errors = []
    for question in questions:
        filters = build_filters(question, mode)
        if mode == "auto" and filters["company"] != question["company"]:
            intent_errors.append((question["id"], filters["company"]))
        for name, ranker in RANKERS.items():
            ranked = ranker(filters, metadata, index, model, bm25)
            rank = first_hit_rank(ranked, metadata, question, strict=True)
            page_rank = first_hit_rank(ranked, metadata, question, strict=False)
            top = metadata[ranked[0]] if ranked else {}
            per_question[name].append({
                "id": question["id"],
                "type": question["type"],
                "rank": rank,
                "page_rank": page_rank,
                "top_result": f"{top.get('company', '')} p{top.get('page_number', '')}",
            })

    # ---- overall and by question type
    overall = {
        name: summarize([r["rank"] for r in rows])
        for name, rows in per_question.items()
    }
    print_table(
        f"Retrieval quality on {len(questions)} questions "
        f"(filters: {mode}, fusion weights bm25={settings.bm25_weight} "
        f"dense={settings.dense_weight})",
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

    lenient = summarize([r["page_rank"] for r in per_question["hybrid"]])
    print(
        f"\nFor comparison, page-level (lenient) hybrid: "
        f"hit@10 {lenient['hit@10']:.2f}, mrr@10 {lenient['mrr@10']:.2f}"
    )

    if mode == "auto":
        print(f"\nCompany detected correctly for "
              f"{len(questions) - len(intent_errors)}/{len(questions)} questions")
        for qid, got in intent_errors:
            print(f"  {qid}: detected {got}")

    # ---- what hybrid got wrong: these are the cases to study
    print("\nHybrid misses (no chunk containing the answer in the top 10):")
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
        "filter_mode": mode,
        "fusion_weights": {"bm25": settings.bm25_weight, "dense": settings.dense_weight},
        "index_info": index_info,
        "overall": overall,
        "hybrid_page_level": lenient,
        "per_question": per_question,
    }
    model_tag = index_info.get("embedding_model", "unknown").split("/")[-1]
    out_path = settings.eval_results_dir / (
        f"{datetime.now():%Y%m%d-%H%M%S}_{model_tag}"
        f"_{mode}.json"
    )
    out_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    logger.info("Saved run: %s", out_path)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filters", choices=["none", "oracle", "auto"], default="auto",
        help="none | oracle (labelled company) | auto (from the question)",
    )
    evaluate(mode=parser.parse_args().filters)
