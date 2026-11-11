"""LLM-as-judge: a second model grades an answer.

Used for what string matching can't check: is the answer complete, does
it state the right unit/basis/period, and is EVERY claim supported by the
cited sources (faithfulness)?

Known weaknesses of LLM judges, and what we do about each:
  - Self-preference: models favour text in their own style
        -> the judge is a different model family (Llama) from the answerer (gpt-oss)
  - Verbosity bias: longer answers look "better"
        -> the rubric rewards correctness and support, not length
  - Inconsistency: same input, different grade
        -> temperature 0, a fixed rubric, and a structured JSON verdict
  - It can simply be wrong
        -> we report its agreement with the deterministic checks, and you
           spot-check a sample by hand before trusting it
"""
import json
import logging
import re

from openai import NotFoundError

from config import settings
from llm import chat

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are grading an answer produced by a financial \
question-answering system that must answer ONLY from the given sources.

Question:
{question}

Reference (the correct figure or facts, from a human answer key):
{reference}

Sources the answer cited:
{sources}

Answer to grade:
{answer}

Grade strictly, using only the reference and the sources:
- "correctness": "correct" (states the reference figure/facts), "partial" \
(right figure but missing or wrong unit/period/basis, or incomplete), or \
"incorrect" (wrong or missing figure).
- "faithful": true only if EVERY factual claim in the answer is supported \
by the cited sources. Calculations count as supported if their inputs are.
- "unsupported_claims": list any claims not supported by the sources.
- "reason": one short sentence.

Return ONLY a JSON object, no other text. Example of the exact format:
{{"correctness": "correct", "faithful": true, "unsupported_claims": [], "reason": "States 926,240 million, matching the reference and source S1."}}"""


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in judge output: {text[:200]}")
    return json.loads(match.group())


def judge_answer(question: str, reference: str, answer: str, cited_sources: list[dict]) -> dict:
    sources = "\n\n".join(
        f"[{s['id']}] {s.get('context', '')}\n{s['text'][:3000]}" for s in cited_sources
    ) or "(no sources cited)"
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference or "(no reference given)",
        sources=sources, answer=answer,
    )
    try:
        response = chat(
            [{"role": "user", "content": prompt}],
            # No server-side JSON mode: it rejects replies cut off by hidden
            # reasoning (400 errors). We extract the JSON ourselves instead.
            max_tokens=2000, model=settings.judge_model, json_mode=False,
        )
        verdict = _parse_json(response.text)
    except NotFoundError as error:
        # Wrong/unavailable model: every call would fail the same way.
        raise RuntimeError(
            f"Judge model '{settings.judge_model}' is not available to your account. "
            "Run 'python src\\llm.py --models' and set FINDOC_JUDGE_MODEL."
        ) from error
    except Exception as error:  # one bad verdict must not crash the evaluation
        logger.warning("Judge failed: %s", error)
        return {"correctness": "error", "faithful": None,
                "unsupported_claims": [], "reason": str(error)[:200]}

    verdict["correctness"] = str(verdict.get("correctness", "error")).lower()
    return verdict
