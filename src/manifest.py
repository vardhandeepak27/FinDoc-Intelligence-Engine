"""Read document facts from data/manifest.csv.

Everything we know about a report *as a document* lives in the manifest:
its company, period, currency, unit, the names people use for the
company (aliases), and report-specific vocabulary (e.g. what that report
calls its standalone accounts). Code reads it; code never hard-codes it.
"""
import csv
from functools import lru_cache

from config import settings

REQUIRED = [
    "filename", "company", "report_type", "period", "fiscal_year_start",
    "fiscal_year_end", "currency", "unit", "source_url",
]
LIST_COLUMNS = ["aliases", "standalone_markers"]  # optional, ";"-separated


@lru_cache
def load_manifest_rows() -> tuple[dict, ...]:
    with settings.manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        rows = []
        for row in reader:
            if not row["filename"].strip():
                continue
            clean = {k: (v or "").strip() for k, v in row.items()}
            for column in LIST_COLUMNS:
                clean[column] = [
                    item.strip() for item in clean.get(column, "").split(";")
                    if item.strip()
                ]
            rows.append(clean)
    return tuple(rows)


def company_aliases() -> dict[str, str]:
    """{'tcs': 'Tata Consultancy Services', 'wipro': 'Wipro', ...}"""
    lexicon = {}
    for row in load_manifest_rows():
        for name in [row["company"], *row["aliases"]]:
            lexicon[name.lower()] = row["company"]
    return lexicon


def standalone_markers(document_name: str) -> list[str]:
    for row in load_manifest_rows():
        if row["filename"] == document_name:
            return row["standalone_markers"]
    return []
