"""Stage 1 of the pipeline: PDFs -> page records (documents.jsonl).

Reads every PDF listed in data/manifest.csv, extracts each page with the
layout-aware extractor, attaches document metadata, and writes a
diagnostics CSV so extraction quality can be reviewed page by page.
"""
import csv
import hashlib
import json
import logging
import time
from pathlib import Path

from config import settings
from extract import EXTRACTOR_VERSION, extract_pdf
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "filename",
    "company",
    "report_type",
    "period",
    "fiscal_year_start",
    "fiscal_year_end",
    "currency",
    "unit",
    "source_url",
]

DIAGNOSTIC_FIELDS = [
    "document_name", "company", "page_number", "printed_page",
    "extraction_method", "char_count", "table_count", "merged_word_count",
    "fragmentation", "broken_text_layer", "numeric_cells_repaired",
    "rupee_glyphs_found", "rupee_glyphs_fixed", "rupee_leftovers",
    "detected_currency",
    "detected_unit", "basis", "image_count", "needs_review", "review_reason",
]


def get_document_id(pdf_path: Path) -> str:
    """Stable ID from the PDF's content: same bytes -> same ID."""
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return f"doc_{digest[:12]}"


def load_manifest() -> dict:
    """Load document metadata, keyed by PDF filename."""
    if not settings.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {settings.manifest_path}")

    with settings.manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        metadata = {}
        for row in reader:
            filename = row["filename"].strip()
            if not filename:
                continue
            if filename in metadata:
                raise ValueError(f"Duplicate filename in manifest: {filename}")
            metadata[filename] = {
                column: row[column].strip()
                for column in MANIFEST_COLUMNS
                if column != "filename"
            }

    return metadata


def review_reasons(page: dict, doc_meta: dict) -> list[str]:
    """Rules that decide whether a human should look at this page."""
    quality = page["quality"]
    reasons = []

    if quality["char_count"] == 0:
        reasons.append("no_text")
    elif quality["char_count"] < 50:
        reasons.append("low_text")

    if quality["broken_text_layer"] and page["extraction_method"] != "ocr":
        reasons.append("broken_text_layer")

    if quality["rupee_leftovers"] > 0:
        reasons.append("rupee_unfixed")

    if quality["merged_word_count"] >= 3:
        reasons.append("merged_words")

    # The page states its own unit: does it agree with the manifest?
    if page["detected_unit"] and (
        page["detected_unit"] != doc_meta["unit"]
        or page["detected_currency"] != doc_meta["currency"]
    ):
        reasons.append("unit_mismatch")

    return reasons


def ingest_corpus() -> None:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(settings.raw_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {settings.raw_dir}")

    manifest = load_manifest()

    unlisted = [p.name for p in pdf_files if p.name not in manifest]
    if unlisted:
        raise ValueError(f"PDFs missing from manifest.csv: {unlisted}")

    diagnostics = []
    total_pages = 0
    started = time.perf_counter()

    with settings.documents_path.open("w", encoding="utf-8") as output:
        for pdf_path in pdf_files:
            doc_meta = manifest[pdf_path.name]
            document_id = get_document_id(pdf_path)
            doc_started = time.perf_counter()
            flagged = 0
            ocr_pages = 0

            logger.info("Extracting %s", pdf_path.name)

            for page in extract_pdf(
                pdf_path,
                use_ocr_fallback=settings.use_ocr_fallback,
                tessdata=str(settings.tessdata_dir) if settings.tessdata_dir else None,
                broken_text_threshold=settings.broken_text_threshold,
            ):
                reasons = review_reasons(page, doc_meta)
                flagged += bool(reasons)
                ocr_pages += page["extraction_method"] == "ocr"

                record = {
                    "document_id": document_id,
                    "document_name": pdf_path.name,
                    **doc_meta,
                    **page,
                    "needs_review": bool(reasons),
                    "review_reason": ";".join(reasons),
                    "extractor_version": EXTRACTOR_VERSION,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

                diagnostics.append({
                    "document_name": pdf_path.name,
                    "company": doc_meta["company"],
                    **{k: page[k] for k in (
                        "page_number", "printed_page", "extraction_method",
                        "detected_currency", "detected_unit", "basis",
                        "image_count",
                    )},
                    **page["quality"],
                    "needs_review": bool(reasons),
                    "review_reason": ";".join(reasons),
                })
                total_pages += 1

                if reasons:
                    logger.debug(
                        "%s p%d flagged: %s",
                        pdf_path.name, page["page_number"], ";".join(reasons),
                    )

            logger.info(
                "%s: done in %.0fs | OCR pages: %d | flagged: %d",
                pdf_path.name, time.perf_counter() - doc_started,
                ocr_pages, flagged,
            )

    with settings.diagnostics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DIAGNOSTIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(diagnostics)

    logger.info(
        "Ingestion complete: %d PDFs, %d pages, %d flagged, %.0fs total",
        len(pdf_files), total_pages,
        sum(row["needs_review"] for row in diagnostics),
        time.perf_counter() - started,
    )
    logger.info("Pages: %s", settings.documents_path)
    logger.info("Diagnostics: %s", settings.diagnostics_path)


if __name__ == "__main__":
    setup_logging()
    ingest_corpus()
