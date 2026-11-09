"""Turn a user question into search filters and a keyword-friendly query.

    "What was TCS's standalone revenue for FY 2026?"
      -> company = "Tata Consultancy Services"   (from the alias "TCS")
      -> basis   = "standalone"
      -> keyword query gets "Tata Consultancy Services" appended, because
         BM25 only matches exact words and the chunks use the full name.

Rule-based on purpose: it is fast, free, predictable and easy to test.
Later we can let an LLM do this for harder questions; rules stay as the
cheap first pass.
"""
import re
from dataclasses import dataclass, field

from config import settings
from manifest import company_aliases

BASIS_PATTERNS = {
    "consolidated": re.compile(r"\b(consolidated|group[- ]level|for the group)\b", re.I),
    "standalone": re.compile(r"\b(standalone|stand-alone|parent company)\b", re.I),
}


@dataclass
class QueryIntent:
    query: str
    companies: list[str] = field(default_factory=list)
    basis: str | None = None
    basis_source: str = "none"      # "question" | "default" | "none"
    keyword_query: str = ""

    @property
    def company(self) -> str | None:
        """A single-company filter only when exactly one company is named.
        Comparisons ("TCS vs Wipro") need multi-company retrieval: Module 7."""
        return self.companies[0] if len(self.companies) == 1 else None


def understand(query: str) -> QueryIntent:
    intent = QueryIntent(query=query)

    # ---- companies (longest alias first, so "Tata Consultancy" beats "Tata")
    found = []
    for alias, company in sorted(company_aliases().items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(alias)}\b", query, re.I) and company not in found:
            found.append(company)
    intent.companies = found

    # ---- basis: only if the question names exactly one
    matches = [b for b, pattern in BASIS_PATTERNS.items() if pattern.search(query)]
    if len(matches) == 1:
        intent.basis, intent.basis_source = matches[0], "question"
    elif not matches and settings.default_basis != "none":
        # Not mentioned: apply the financial-reporting convention
        intent.basis, intent.basis_source = settings.default_basis, "default"

    # ---- keyword query: add canonical names that the question didn't spell
    # out ("TCS" -> "Tata Consultancy Services"). Only useful when we can't
    # filter by one company: with a filter, every candidate already has the
    # name, and the extra words just add noise (measured: it hurt t06).
    missing_names = [c for c in found if c.lower() not in query.lower()]
    if intent.company is None and missing_names:
        intent.keyword_query = " ".join([query, *missing_names])
    else:
        intent.keyword_query = query

    return intent


if __name__ == "__main__":
    for q in [
        "What was TCS's standalone revenue for FY 2026?",
        "Wipro consolidated revenue from operations",
        "Compare TCS and Wipro operating margins",
        "What dividend per share is Capgemini proposing?",
    ]:
        print(understand(q))
