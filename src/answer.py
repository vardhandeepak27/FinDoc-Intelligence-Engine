"""The full RAG pipeline: question -> retrieve -> prompt -> LLM -> checked answer.

    engine = RAGEngine()            # loads indexes and models ONCE (slow)
    result = engine.answer("...")   # per question (fast-ish)

After generation we run two cheap, deterministic checks:
  1. Citations:  every [S#] must point to a source we actually sent.
  2. Grounding:  every number in the answer must appear in a source the
                 answer cites. A number that doesn't is either a calculation
                 (fine, if shown) or a hallucination (not fine): we flag it.
LLMs can't be trusted to check themselves; plain code can check them.
"""
import logging
import re
import time
from dataclasses import asdict, dataclass, field

from config import settings
from hybrid_retrieve import load_data, search
from llm import chat
from prompt import NOT_FOUND, build_messages

logger = logging.getLogger(__name__)

CITATION = re.compile(r"\[S(\d+)\]")
NUMBER = re.compile(r"(?<![\w.])\(?-?\d[\d,]*(?:\.\d+)?\)?%?")
CHARS_PER_TOKEN = 3.5  # rough estimate for English + numbers


@dataclass
class AnswerResult:
    question: str
    answer: str
    found: bool
    sources: list[dict]
    grounding: dict
    intent: dict
    model: str = ""
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ checks

def normalize_number(raw: str) -> str:
    """'(41,527)' -> '41527', '2,67,021' -> '267021', '23.51%' -> '23.51'."""
    return re.sub(r"[(),%\s]", "", raw).lstrip("-")


def numbers_in(text: str) -> set[str]:
    return {normalize_number(m.group()) for m in NUMBER.finditer(text)} - {""}


def check_answer(answer: str, question: str, sources: list[dict]) -> dict:
    cited_ids = sorted({int(n) for n in CITATION.findall(answer)})
    valid_ids = [i for i in cited_ids if 1 <= i <= len(sources)]
    invalid_ids = [i for i in cited_ids if i not in valid_ids]

    answer_without_markers = CITATION.sub("", answer)
    cited_numbers = set().union(*(numbers_in(sources[i - 1]["text"]) for i in valid_ids)) \
        if valid_ids else set()
    all_numbers = set().union(*(numbers_in(s["text"]) for s in sources)) if sources else set()
    question_numbers = numbers_in(question)

    grounded, uncited, ungrounded = [], [], []
    for raw in NUMBER.findall(answer_without_markers):
        n = normalize_number(raw)
        if not n or n in question_numbers or len(n.replace(".", "")) <= 1:
            continue  # trivial numbers ("1", "2") and ones from the question
        if n in cited_numbers:
            grounded.append(raw)
        elif n in all_numbers:
            uncited.append(raw)    # true, but the citation points elsewhere
        else:
            ungrounded.append(raw)  # calculation or hallucination: flag it

    return {
        "cited_sources": valid_ids,
        "invalid_citations": invalid_ids,
        "numbers_grounded": grounded,
        "numbers_in_uncited_source": uncited,
        "numbers_not_in_sources": ungrounded,
        "has_citations": bool(valid_ids),
    }


# ------------------------------------------------------------------ engine

def select_context(results: list[dict]) -> list[dict]:
    """Top chunks, in rank order, within the chunk count and token budget."""
    chosen, used = [], 0
    for chunk in results[: settings.answer_context_chunks]:
        tokens = len(chunk["text"]) / CHARS_PER_TOKEN
        if chosen and used + tokens > settings.answer_context_tokens:
            break
        chosen.append(chunk)
        used += tokens
    return chosen


class RAGEngine:
    def __init__(self):
        started = time.perf_counter()
        self.metadata, self.index, self.model, self.bm25 = load_data()
        logger.info("RAG engine ready in %.1fs", time.perf_counter() - started)

    def answer(self, question: str) -> AnswerResult:
        t0 = time.perf_counter()
        intent, results = search(
            question, self.metadata, self.index, self.model, self.bm25,
            top_k=settings.answer_context_chunks,
        )
        sources = select_context(results)
        t1 = time.perf_counter()

        response = chat(build_messages(question, sources))
        t2 = time.perf_counter()

        found = NOT_FOUND.lower()[:30] not in response.text.lower()
        grounding = check_answer(response.text, question, sources)

        if found and not grounding["has_citations"]:
            logger.warning("Answer has no citations")
        if grounding["numbers_not_in_sources"]:
            logger.warning(
                "Numbers not found in any source: %s", grounding["numbers_not_in_sources"]
            )

        cited = set(grounding["cited_sources"])
        return AnswerResult(
            question=question,
            answer=response.text,
            found=found,
            sources=[
                {
                    "id": f"S{i}",
                    "cited": i in cited,
                    "company": s["company"],
                    "document_name": s["document_name"],
                    "page_number": s["page_number"],
                    "printed_page": s.get("printed_page"),
                    "context": s.get("context", ""),
                    "text": s["text"],
                }
                for i, s in enumerate(sources, start=1)
            ],
            grounding=grounding,
            intent={"company": intent.company, "basis": intent.basis},
            model=response.model,
            timings={
                "retrieval_s": round(t1 - t0, 2),
                "llm_s": round(t2 - t1, 2),
                "total_s": round(t2 - t0, 2),
            },
            usage=response.usage,
        )
