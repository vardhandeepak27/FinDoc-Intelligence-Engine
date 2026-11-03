import csv
import hashlib
import json
import logging
import re
from pathlib import Path

import pymupdf

from config import settings
from logging_setup import setup_logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize whitespace without removing meaningful line breaks."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(pdf_path: str | Path) -> str:
    """Extract text from a PDF, preserving page order."""
    pdf_path = Path(pdf_path)

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page in document:
            text = clean_text(page.get_text("text", sort=True))
            pages.append(text)

    return "\n\n".join(pages)


def get_document_id(pdf_path: Path) -> str:
    """Generate a stable ID from the PDF content."""
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return f"doc_{digest[:12]}"


def load_manifest() -> dict:
    """Load metadata indexed by PDF filename."""
    if not settings.manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {settings.manifest_path}"
        )

    with settings.manifest_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "filename",
            "company",
            "report_type",
            "period",
            "source_url",
        }

        if not reader.fieldnames or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                "Manifest must contain: filename, company, "
                "report_type, period, source_url"
            )

        metadata = {}

        for row in reader:
            filename = row["filename"].strip()

            if not filename:
                continue

            if filename in metadata:
                raise ValueError(
                    f"Duplicate filename in manifest: {filename}"
                )

            metadata[filename] = {
                "company": row["company"].strip(),
                "report_type": row["report_type"].strip(),
                "period": row["period"].strip(),
                "source_url": row["source_url"].strip(),
            }

    return metadata



def ingest_corpus() -> None:
    """Extract each PDF page and generate page-level diagnostics."""
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(settings.raw_dir.glob("*.pdf"))
    page_diagnostics = []

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {settings.raw_dir}"
        )

    manifest = load_manifest()
    records_written = 0
    empty_pages = 0

    with settings.documents_path.open("w", encoding="utf-8") as output:
        for pdf_path in pdf_files:
            filename = pdf_path.name

            if filename not in manifest:
                raise ValueError(
                    f"Missing metadata for {filename} "
                    "in data/manifest.csv"
                )

            metadata = manifest[filename]
            document_id = get_document_id(pdf_path)

            with pymupdf.open(pdf_path) as document:
                logger.info(
                    "Processing %s (%d pages)", filename, len(document)
                )
                doc_flagged = 0

                for page_index, page in enumerate(document):
                    # 1. Extract this page's text
                    text = clean_text(
                        page.get_text("text", sort=True)
                    )

                    # 2. Calculate diagnostics for this page
                    char_count = len(text)
                    image_count = len(page.get_images(full=True))
                    has_text = bool(text.strip())


                    review_reasons = []

                    if not has_text:
                        review_reasons.append("no_text")
                    elif char_count < 50:
                        review_reasons.append("low_text")

                    needs_review = bool(review_reasons)
                    review_reason = ";".join(review_reasons)

                    if not has_text:
                        empty_pages += 1

                    if needs_review:
                        doc_flagged += 1
                        logger.debug(
                            "%s page %d flagged: %s",
                            filename, page_index + 1, review_reason,
                        )

                    # 3. Create the page record
                    record = {
                        "document_id": document_id,
                        "document_name": filename,
                        "company": metadata["company"],
                        "report_type": metadata["report_type"],
                        "period": metadata["period"],
                        "source_url": metadata["source_url"],
                        "page_number": page_index + 1,
                        "page_index": page_index,
                        "text": text,
                        "char_count": char_count,
                        "has_text": has_text,
                        "image_count": image_count,
                        "needs_review": needs_review,
                        "review_reason": review_reason,
                    }

                    # 4. Write the full record to JSONL
                    output.write(
                        json.dumps(record, ensure_ascii=False)
                        + "\n"
                    )
                    records_written += 1

                    # 5. Collect a separate CSV diagnostic row
                    page_diagnostics.append({
                        "document_id": document_id,
                        "document_name": filename,
                        "company": metadata["company"],
                        "period": metadata["period"],
                        "page_number": page_index + 1,
                        "char_count": char_count,
                        "has_text": has_text,
                        "image_count": image_count,
                        "needs_review": needs_review,
                        "review_reason": review_reason,
                    })

            if doc_flagged:
                logger.warning(
                    "%s: %d pages flagged for review (see %s)",
                    filename, doc_flagged, settings.diagnostics_path.name,
                )

    # Write diagnostics once, after all PDFs are processed
    diagnostic_fields = [
        "document_id",
        "document_name",
        "company",
        "period",
        "page_number",
        "char_count",
        "has_text",
        "image_count",
        "needs_review",
        "review_reason",
    ]

    with settings.diagnostics_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=diagnostic_fields
        )
        writer.writeheader()
        writer.writerows(page_diagnostics)

    flagged_pages = sum(
        row["needs_review"] for row in page_diagnostics
    )

    logger.info(
        "Ingestion complete: %d PDFs, %d pages, %d empty, %d flagged",
        len(pdf_files), records_written, empty_pages, flagged_pages,
    )
    logger.info("Pages: %s", settings.documents_path)
    logger.info("Diagnostics: %s", settings.diagnostics_path)


if __name__ == "__main__":
    setup_logging()
    ingest_corpus()
