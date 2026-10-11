import csv
import hashlib
import json
import re
from pathlib import Path

import fitz  # PyMuPDF


# Resolve paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
OUTPUT_PATH = PROCESSED_DIR / "documents.jsonl"
DIAGNOSTICS_PATH = PROCESSED_DIR / "page_diagnostics.csv"


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

    with fitz.open(pdf_path) as document:
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
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
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
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    page_diagnostics = []

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {RAW_DIR}"
        )

    manifest = load_manifest()
    records_written = 0
    empty_pages = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for pdf_path in pdf_files:
            filename = pdf_path.name

            if filename not in manifest:
                raise ValueError(
                    f"Missing metadata for {filename} "
                    "in data/manifest.csv"
                )

            metadata = manifest[filename]
            document_id = get_document_id(pdf_path)

            with fitz.open(pdf_path) as document:
                print(
                    f"Processing {filename} "
                    f"({len(document)} pages)"
                )

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

    with DIAGNOSTICS_PATH.open(
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

    print("\nIngestion complete")
    print(f"PDFs processed: {len(pdf_files)}")
    print(f"Page records written: {records_written}")
    print(f"Empty pages to inspect: {empty_pages}")
    print(f"Pages flagged for review: {flagged_pages}")
    print(f"JSONL output: {OUTPUT_PATH}")
    print(f"Diagnostics CSV: {DIAGNOSTICS_PATH}")

if __name__ == "__main__":
    ingest_corpus()
