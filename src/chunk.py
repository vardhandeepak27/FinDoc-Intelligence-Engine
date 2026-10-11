
import json
import re
from pathlib import Path


# Resolve paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "documents.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "chunks.jsonl"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


FINANCIAL_HEADINGS = [
    "consolidated balance sheet",
    "standalone balance sheet",
    "statement of profit and loss",
    "statement of cash flows",
    "cash flow statement",
    "statement of changes in equity",
    "notes to financial statements",
    "notes to the financial statements",
    "revenue from operations",
    "total comprehensive income",
    "auditor's report",
    "independent auditor",
    "management discussion and analysis",
    "management discussion & analysis",
    "directors' report",
    "corporate governance",
    "risk management",
    "segment reporting",
]


def is_heading(line: str) -> bool:
    """Heuristically identify likely section headings."""
    line = line.strip()

    if not line or len(line) > 120:
        return False

    normalized = line.lower().strip(" :.-")

    if any(
        heading in normalized
        for heading in FINANCIAL_HEADINGS
    ):
        return True

    # Numbered headings such as "1. Overview" or "2.1 Revenue"
    if re.match(
        r"^\d+(?:\.\d+)*\.?\s+[A-Z]",
        line
    ):
        return True

    # Short, predominantly uppercase headings
    letters = [char for char in line if char.isalpha()]

    if len(letters) >= 5:
        uppercase_ratio = sum(
            char.isupper() for char in letters
        ) / len(letters)

        if uppercase_ratio >= 0.85:
            return True

    return False


def split_lines(
    lines: list[str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Create chunks from lines, preserving a small overlap."""
    if chunk_size <= overlap:
        raise ValueError(
            "chunk_size must be greater than overlap"
        )

    chunks = []
    current_lines = []
    current_length = 0

    for line in lines:
        line = line.strip()

        if not line:
            continue

        extra_length = len(line) + (
            1 if current_lines else 0
        )

        if (
            current_lines
            and current_length + extra_length > chunk_size
        ):
            chunks.append("\n".join(current_lines))

            # Preserve whole trailing lines for context
            overlap_lines = []
            overlap_length = 0

            for previous_line in reversed(current_lines):
                added_length = len(previous_line) + (
                    1 if overlap_lines else 0
                )

                if overlap_length + added_length > overlap:
                    break

                overlap_lines.insert(0, previous_line)
                overlap_length += added_length

            current_lines = overlap_lines
            current_length = overlap_length

        if current_lines:
            current_length += 1

        current_lines.append(line)
        current_length += len(line)

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Keep a simple chunk_text interface for existing callers."""
    lines = text.splitlines()
    return split_lines(lines, chunk_size, overlap)


def chunk_page(
    text: str,
    initial_section: str = "Unknown",
) -> tuple[list[dict], str]:
    """Chunk a page while tracking likely section headings."""
    lines = text.splitlines()

    chunks = []
    current_lines = []
    current_section = initial_section

    def save_current():
        if not current_lines:
            return

        for chunk in split_lines(current_lines):
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "section": current_section,
                })

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if is_heading(line):
            if current_lines:
                save_current()
                current_lines = []

            current_section = line

        current_lines.append(line)

    save_current()

    return chunks, current_section


def chunk_documents() -> None:
    """Convert page-level JSONL records into chunk-level records."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. "
            "Run ingest.py first."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Remember the latest section for each document
    document_sections = {}

    records_written = 0
    pages_processed = 0

    with (
        INPUT_PATH.open("r", encoding="utf-8") as source,
        OUTPUT_PATH.open("w", encoding="utf-8") as output,
    ):
        for line in source:
            if not line.strip():
                continue

            page_record = json.loads(line)
            pages_processed += 1

            document_id = page_record["document_id"]

            initial_section = document_sections.get(
                document_id, "Unknown"
            )

            page_chunks, final_section = chunk_page(
                page_record.get("text", ""),
                initial_section,
            )

            document_sections[document_id] = final_section

            for chunk_index, chunk in enumerate(page_chunks):
                chunk_text_value = chunk["text"]

                chunk_record = {
                    "chunk_id": (
                        f"{document_id}"
                        f"_p{page_record['page_number']}"
                        f"_c{chunk_index}"
                    ),
                    "document_id": document_id,
                    "document_name": page_record["document_name"],
                    "company": page_record["company"],
                    "report_type": page_record["report_type"],
                    "period": page_record["period"],
                    "source_url": page_record["source_url"],
                    "page_number": page_record["page_number"],
                    "page_index": page_record["page_index"],
                    "section": chunk["section"],
                    "text": chunk_text_value,
                    "char_count": len(chunk_text_value),
                }

                output.write(
                    json.dumps(
                        chunk_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                records_written += 1

    print("\nChunking complete")
    print(f"Pages processed: {pages_processed}")
    print(f"Chunks written: {records_written}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    chunk_documents()
