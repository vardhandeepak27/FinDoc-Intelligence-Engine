"""Perturbation test: can the judge actually catch wrong answers?

An agreement score is only meaningful if the judge has seen wrong answers.
Our system is ~97% correct, so a judge that always says "correct" would
also agree ~97% of the time. Here we take answers that are known to be
correct, BREAK them on purpose, and check that the judge notices:

  control   unchanged correct answer       -> judge should accept
  number    one digit of the figure changed -> judge should reject
  basis     standalone <-> consolidated     -> judge should reject
  unit      million <-> crore               -> judge should reject
  invented  an unsupported claim added      -> judge should flag unfaithful

Usage:
  python src\\judge_check.py data\\eval\\results\\<run>_answers.json
"""
import json
import logging
import re
import sys
from pathlib import Path

from config import settings
from evaluate_answers import contains_any, load_jsonl
from judge import judge_answer
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

N_ANSWERS = 8  # x up to 5 variants = up to 40 judge calls


def change_one_digit(answer: str, figure: str) -> str | None:
    """'926,240' -> '962,240' (swap two neighbouring different digits)."""
    if figure not in answer:
        return None
    digits = [i for i, ch in enumerate(figure) if ch.isdigit()]
    for a, b in zip(digits, digits[1:]):
        if figure[a] != figure[b]:
            changed = list(figure)
            changed[a], changed[b] = changed[b], changed[a]
            return answer.replace(figure, "".join(changed), 1)
    return None


def flip_basis(answer: str) -> str | None:
    if re.search(r"consolidated", answer, re.I):
        return re.sub(r"consolidated", "standalone", answer, count=1, flags=re.I)
    if re.search(r"standalone", answer, re.I):
        return re.sub(r"standalone", "consolidated", answer, count=1, flags=re.I)
    return None


def flip_unit(answer: str) -> str | None:
    if "million" in answer:
        return answer.replace("million", "crore", 1)
    if "crore" in answer:
        return answer.replace("crore", "million", 1)
    return None


def add_invented_claim(answer: str) -> str:
    return answer.rstrip() + (
        " This was a 37% increase over the previous year, driven mainly by "
        "a new contract with the Government of Japan [S1]."
    )


def judged_ok(verdict: dict) -> bool:
    """Did the judge accept the answer as correct AND faithful?"""
    return verdict.get("correctness") == "correct" and verdict.get("faithful") is True


def main(run_path: str) -> None:
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_jsonl(settings.eval_path)}

    # Answers known to be right: string check passes and sources are cited
    good = [
        r for r in run["answerable"]
        if questions[r["id"]].get("answer_contains")
        and contains_any(r["answer"], questions[r["id"]]["answer_contains"])
        and any(s["cited"] for s in r["sources"])
    ][:N_ANSWERS]

    results = {k: [] for k in ("control", "number", "basis", "unit", "invented")}
    for row in good:
        q = questions[row["id"]]
        figure = next((a for a in q["answer_contains"] if a in row["answer"]), "")
        variants = {
            "control": row["answer"],
            "number": change_one_digit(row["answer"], figure),
            "basis": flip_basis(row["answer"]),
            "unit": flip_unit(row["answer"]),
            "invented": add_invented_claim(row["answer"]),
        }
        cited = [s for s in row["sources"] if s["cited"]]
        reference = ", ".join(q["answer_contains"])
        for kind, text in variants.items():
            if text is None or (kind != "control" and text == row["answer"]):
                continue  # this perturbation doesn't apply to this answer
            verdict = judge_answer(q["question"], reference, text, cited)
            if verdict.get("correctness") == "error":
                continue
            results[kind].append((row["id"], judged_ok(verdict), verdict.get("reason", "")))
            logger.info("%s %-8s -> %s / faithful=%s", row["id"], kind,
                        verdict.get("correctness"), verdict.get("faithful"))

    print(f"\nJudge: {settings.judge_model} | {len(good)} correct answers, perturbed\n")
    print(f"{'variant':<10}{'n':>4}  {'judge accepted':>15}  {'should be':>10}  {'score':>6}")
    print("-" * 52)
    for kind, rows in results.items():
        if not rows:
            continue
        accepted = sum(ok for _, ok, _ in rows)
        if kind == "control":
            score = accepted / len(rows)          # want: accept all
            target = "accepted"
        else:
            score = 1 - accepted / len(rows)      # want: reject all
            target = "rejected"
        print(f"{kind:<10}{len(rows):>4}  {accepted:>15}  {target:>10}  {score:>6.2f}")

    print("\nMissed perturbations (judge accepted a broken answer):")
    for kind, rows in results.items():
        if kind == "control":
            continue
        for qid, ok, reason in rows:
            if ok:
                print(f"  {qid} {kind}: {reason}")


if __name__ == "__main__":
    setup_logging()
    if len(sys.argv) < 2:
        sys.exit("Usage: python src\\judge_check.py <run>_answers.json")
    main(sys.argv[1])
