"""Command-line interface: ask questions about the reports.

    python src\\main.py "What was Wipro's consolidated revenue for FY26?"
    python src\\main.py            # interactive: ask several questions

Interactive mode loads the models once and keeps them in memory, which is
how the FastAPI server will work too (Module 8).
"""
import sys

from answer import RAGEngine
from logging_setup import setup_logging


def show(result) -> None:
    print("\n" + "=" * 78)
    print(result.answer)
    print("=" * 78)

    print("\nSources:")
    for s in result.sources:
        marker = "*" if s["cited"] else " "
        printed = f", printed p.{s['printed_page']}" if s["printed_page"] else ""
        print(f" {marker} [{s['id']}] {s['company']} — PDF p.{s['page_number']}{printed}")
    print("   (* = cited in the answer)")

    g = result.grounding
    checks = []
    if g["invalid_citations"]:
        checks.append(f"invalid citations {g['invalid_citations']}")
    if g["numbers_in_uncited_source"]:
        checks.append(f"numbers from an uncited source {g['numbers_in_uncited_source']}")
    if g["numbers_not_in_sources"]:
        checks.append(f"numbers NOT in any source {g['numbers_not_in_sources']}")
    if result.found and not g["has_citations"]:
        checks.append("no citations")
    status = "OK" if not checks else "CHECK: " + "; ".join(checks)
    print(f"\nGrounding: {len(g['numbers_grounded'])} numbers verified | {status}")

    t = result.timings
    print(
        f"Time: retrieval {t['retrieval_s']}s + LLM {t['llm_s']}s = {t['total_s']}s"
        f" | model {result.model} | filters {result.intent}"
    )


def main() -> None:
    setup_logging()
    engine = RAGEngine()

    if len(sys.argv) > 1:
        show(engine.answer(" ".join(sys.argv[1:])))
        return

    print("\nAsk a question about the reports (empty line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        show(engine.answer(question))


if __name__ == "__main__":
    main()
