"""End-to-end evaluation: run the full RAG pipeline and grade the ANSWERS.

Retrieval evaluation (evaluate_retrieval.py) asks: did we find the chunk?
This asks: is the final answer right, supported, and honest?

Per answerable question (data/eval/retrieval_eval.jsonl):
  correct          answer contains a labelled answer string   (deterministic)
  supported        that string appears in a source the answer CITES
  right_page       a cited source is one of the labelled pages
  answer_in_context  some source we SENT contains the answer  (retrieval worked?)
  unverified_numbers numbers in the answer found in no source (hallucination signal)
  judge            LLM-as-judge correctness + faithfulness    (optional, --judge)

Per unanswerable question (data/eval/unanswerable.jsonl):
  refused          did the system say it couldn't find it?

Failure analysis: every wrong answer is classified as
  retrieval_miss   the answer wasn't in the sources -> fix retrieval
  false_refusal    it was there, but the model said "not found" -> fix prompt
  generation_error it was there, the model answered wrongly -> fix prompt/model

Usage:
  python src\\evaluate_answers.py                  # all questions, no judge
  python src\\evaluate_answers.py --judge          # + LLM-as-judge
  python src\\evaluate_answers.py --limit 5        # quick trial
  python src\\evaluate_answers.py --rescore <file> # re-grade saved answers, no LLM calls
"""
import argparse
import csv
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from config import settings
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

UNANSWERABLE_PATH = settings.data_dir / "eval" / "unanswerable.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def norm(text: str) -> str:
    """Formatting-insensitive form: '₹ 2,67,021' -> '₹267021'."""
    return re.sub(r"[\s*_`|,]+", "", text.lower().replace(" ", " "))


def contains_any(text: str, needles: list[str]) -> bool:
    t = norm(text)
    return any(norm(n) in t for n in needles)


# ------------------------------------------------------------------ grading

def grade_answerable(q: dict, r: dict) -> dict:
    strings = q.get("answer_contains") or []
    sources = r["sources"]
    cited = [s for s in sources if s["cited"]]
    on_page = lambda s: (s["document_name"] == q["document_name"]
                         and s["page_number"] in q["expected_pages"])

    if strings:
        correct = contains_any(r["answer"], strings)
        supported = any(contains_any(s["text"], strings) for s in cited)
        in_context = any(contains_any(s["text"], strings) for s in sources)
    else:  # no answer string labelled: page-level proxies only
        correct = None
        supported = any(on_page(s) for s in cited)
        in_context = any(on_page(s) for s in sources)

    if correct:
        failure = "ok"
    elif correct is None:
        failure = "unknown (no answer string; see judge)"
    elif not in_context:
        failure = "retrieval_miss"
    elif not r["found"]:
        failure = "false_refusal"
    else:
        failure = "generation_error"

    return {
        "answered": r["found"],
        "correct": correct,
        "supported": supported,
        "right_page": any(on_page(s) for s in cited),
        "answer_in_context": in_context,
        "unverified_numbers": r["grounding"]["numbers_not_in_sources"],
        "failure": failure,
    }


def rate(values) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "  n/a"
    return f"{sum(bool(v) for v in values) / len(values):5.2f}  ({sum(bool(v) for v in values)}/{len(values)})"


def report(rows: list[dict], negatives: list[dict], judged: bool) -> dict:
    g = [r["grade"] for r in rows]
    print(f"\nPrompt version: {settings.prompt_version}")
    print(f"\n{'=' * 70}\nANSWERABLE QUESTIONS (n={len(rows)})\n{'=' * 70}")
    print(f"Answered (not refused)        {rate(x['answered'] for x in g)}")
    print(f"Correct (answer string)       {rate(x['correct'] for x in g)}")
    print(f"Supported by a cited source   {rate(x['supported'] for x in g)}")
    print(f"Cites a labelled page*        {rate(x['right_page'] for x in g)}")
    print(f"Answer was in the context     {rate(x['answer_in_context'] for x in g)}")
    print(f"Has unverified numbers        {rate(bool(x['unverified_numbers']) for x in g)}")
    print("  * lower bound: a fact often appears on more pages than we labelled,"
          " so 'Supported' is the better measure")

    if judged:
        verdicts = [r["judge"] for r in rows if r.get("judge")]
        counts = {k: sum(v["correctness"] == k for v in verdicts)
                  for k in ("correct", "partial", "incorrect", "error")}
        print(f"\nJudge correctness             {counts}")
        print(f"Judge: faithful               {rate(v.get('faithful') for v in verdicts)}")
        pairs = [(r["grade"]["correct"], r["judge"]["correctness"] == "correct")
                 for r in rows if r.get("judge") and r["grade"]["correct"] is not None
                 and r["judge"]["correctness"] != "error"]
        if pairs:
            agree = sum(a == b for a, b in pairs) / len(pairs)
            print(f"Judge agrees with string check {agree:5.2f}  (on {len(pairs)} questions)")

    failures = {}
    for r in rows:
        failures.setdefault(r["grade"]["failure"], []).append(r["id"])
    print("\nFailure analysis:")
    for kind, ids in sorted(failures.items()):
        print(f"  {kind:<38}{len(ids):>3}  {ids if kind != 'ok' else ''}")

    print(f"\n{'=' * 70}\nUNANSWERABLE QUESTIONS (n={len(negatives)})\n{'=' * 70}")
    print(f"Correctly refused             {rate(not n['found'] for n in negatives)}")
    for n in negatives:
        if n["found"]:
            print(f"  ANSWERED {n['id']}: {n['question']}\n      -> {n['answer'][:160]}")

    t = [r["timings"]["total_s"] for r in rows]
    tok = [r["usage"].get("total_tokens", 0) for r in rows]
    print(f"\nAvg time/question {sum(t) / len(t):.1f}s | avg tokens {sum(tok) / len(tok):.0f}")

    return {
        "answered": [x["answered"] for x in g].count(True) / len(g),
        "correct": rate(x["correct"] for x in g).strip(),
        "failures": {k: len(v) for k, v in failures.items()},
        "refusal_rate_unanswerable": sum(not n["found"] for n in negatives) / max(len(negatives), 1),
    }


# ------------------------------------------------------------------ main

def main(limit: int | None, use_judge: bool, rescore: str | None) -> None:
    questions = load_jsonl(settings.eval_path)[:limit]
    negatives_q = load_jsonl(UNANSWERABLE_PATH)[: (limit or 99)]

    if rescore:
        saved = json.loads(Path(rescore).read_text(encoding="utf-8"))
        rows, negatives = saved["answerable"], saved["unanswerable"]
        from answer import check_answer  # re-run checks with the current code
        by_id = {q["id"]: q for q in questions}
        for row in rows:
            row["grounding"] = check_answer(row["answer"], row["question"], row["sources"])
            row["grade"] = grade_answerable(by_id[row["id"]], row)
    else:
        from answer import RAGEngine
        engine = RAGEngine()
        rows, negatives = [], []
        started = time.perf_counter()
        all_q = [("pos", q) for q in questions] + [("neg", q) for q in negatives_q]
        for i, (kind, q) in enumerate(all_q, start=1):
            logger.info("[%d/%d] %s", i, len(all_q), q["question"])
            result = engine.answer(q["question"]).to_dict()
            result["id"] = q["id"]
            if kind == "pos":
                result["grade"] = grade_answerable(q, result)
                rows.append(result)
            else:
                negatives.append(result)
        logger.info("Answered %d questions in %.1f min", len(all_q),
                    (time.perf_counter() - started) / 60)

    if use_judge:
        from judge import judge_answer
        by_id = {q["id"]: q for q in questions}
        for row in rows:
            q = by_id[row["id"]]
            reference = ", ".join(q.get("answer_contains") or []) or \
                f"answer is on pages {q['expected_pages']} of {q['document_name']}"
            row["judge"] = judge_answer(
                q["question"], reference, row["answer"],
                [s for s in row["sources"] if s["cited"]],
            )

    summary = report(rows, negatives, judged=use_judge)

    # ---- save: full JSON for re-scoring, CSV for reading by a human
    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    run_path = settings.eval_results_dir / f"{stamp}_answers.json"
    run_path.write_text(json.dumps({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "llm_model": rows[0]["model"] if rows else None,
        "judge_model": settings.judge_model if use_judge else None,
        "reranker": settings.reranker_model,
        "prompt_version": settings.prompt_version,
        "summary": summary,
        "answerable": rows,
        "unanswerable": negatives,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = settings.eval_results_dir / f"{stamp}_answers_review.csv"
    by_id = {q["id"]: q for q in questions}
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "question", "expected", "answer", "string_correct",
                         "failure", "judge", "judge_faithful", "judge_reason",
                         "your_verdict"])
        for r in rows:
            q, j = by_id[r["id"]], r.get("judge") or {}
            writer.writerow([r["id"], q["question"], "; ".join(q.get("answer_contains") or []),
                             r["answer"], r["grade"]["correct"], r["grade"]["failure"],
                             j.get("correctness", ""), j.get("faithful", ""),
                             j.get("reason", ""), ""])
    logger.info("Saved: %s", run_path)
    logger.info("Review sheet (open in Excel): %s", csv_path)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--rescore", default=None, help="saved *_answers.json to re-grade")
    args = parser.parse_args()
    main(args.limit, args.judge, args.rescore)
