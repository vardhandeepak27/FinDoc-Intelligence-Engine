"""Stage 2 of the pipeline: page records -> retrieval chunks.

Structure-aware chunking:
  1. Parse each page's Markdown into blocks: headings, tables, paragraphs.
  2. Track the current section (carried across pages).
  3. Pack blocks into chunks up to `chunk_size` characters, never cutting a
     table in the middle of a row. Long tables are split by rows, and every
     piece repeats the table's header rows.
  4. Prefix every chunk with a short context line (company, period, basis,
     section, unit) so each chunk makes sense on its own.
"""
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from config import settings
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

CHUNKER_VERSION = "structure-v2"

UNIT_LABELS = {
    ("INR", "crore"): "₹ crore",
    ("INR", "lakh"): "₹ lakh",
    ("INR", "million"): "₹ million",
    ("EUR", "million"): "€ million",
    ("USD", "million"): "US$ million",
}

HEADING_LINE = re.compile(r"^#{1,6}\s+")
TABLE_LINE = re.compile(r"^\s*\|")
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(₹€\d])")


@dataclass
class Block:
    kind: str  # "heading" | "table" | "text"
    text: str


# ------------------------------------------------------------------ parsing

def clean_heading(line: str) -> str | None:
    """'### **5.2.4 Consolidated Statement**' -> '5.2.4 Consolidated Statement'.

    Returns None for headings that are really numbers or decoration, e.g.
    '### ₹ **4,499 mn**' (big KPI figures are often styled as headings).
    """
    text = re.sub(r"[#*_`]", "", line)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" :|-")
    if len(re.findall(r"[A-Za-z]", text)) < 4 or len(text) > 150:
        return None
    return text


def parse_blocks(markdown: str) -> list[Block]:
    """Split page Markdown into heading / table / text blocks."""
    blocks: list[Block] = []
    buffer: list[str] = []
    buffer_kind = None

    def flush():
        nonlocal buffer, buffer_kind
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(Block(buffer_kind, text))
        buffer, buffer_kind = [], None

    for line in markdown.splitlines():
        stripped = line.strip()

        if not stripped:
            if buffer_kind == "text":
                flush()  # blank line ends a paragraph (tables continue)
            continue

        if HEADING_LINE.match(stripped):
            flush()
            heading = clean_heading(stripped)
            if heading:
                blocks.append(Block("heading", heading))
            continue

        kind = "table" if TABLE_LINE.match(stripped) else "text"
        if kind != buffer_kind:
            flush()
            buffer_kind = kind
        buffer.append(stripped)

    flush()
    return blocks


# ------------------------------------------------------------ splitting

def split_table(table: str, max_chars: int) -> list[str]:
    """Split a long Markdown table by rows, repeating the header rows."""
    if len(table) <= max_chars:
        return [table]

    lines = table.splitlines()
    sep_index = next(
        (i for i, l in enumerate(lines[:4]) if TABLE_SEPARATOR.match(l)), None
    )
    header = lines[: sep_index + 1] if sep_index is not None else []
    rows = lines[len(header):]
    header_text = "\n".join(header)

    pieces, current = [], []
    current_len = len(header_text)

    for row in rows:
        if current and current_len + len(row) + 1 > max_chars:
            pieces.append("\n".join(header + current))
            current, current_len = [], len(header_text)
        current.append(row)
        current_len += len(row) + 1

    if current:
        pieces.append("\n".join(header + current))
    return pieces


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split prose at sentence boundaries, with a small sentence overlap."""
    if len(text) <= max_chars:
        return [text]

    sentences = SENTENCE_END.split(text)
    pieces, current = [], []

    for sentence in sentences:
        # A single sentence longer than the limit: hard-wrap on words
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            sentences_left = sentence[cut:].strip()
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(sentence[:cut].strip())
            sentence = sentences_left

        if current and len(" ".join(current + [sentence])) > max_chars:
            pieces.append(" ".join(current))
            # carry the last sentence(s) forward, up to `overlap` chars
            carried = []
            for previous in reversed(current):
                if len(" ".join([previous] + carried)) > overlap:
                    break
                carried.insert(0, previous)
            current = carried
        current.append(sentence)

    if current:
        pieces.append(" ".join(current))
    return [p for p in pieces if p.strip()]


# ------------------------------------------------------------- context

def find_boilerplate_headers(pages: list[dict]) -> dict[str, set[str]]:
    """Header parts that appear on many pages of a document carry no
    information (e.g. 'WIPRO INTEGRATED ANNUAL REPORT 2025-26')."""
    counts: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()

    for page in pages:
        doc = page["document_id"]
        totals[doc] += 1
        for part in set(filter(None, page.get("page_header", "").split(" | "))):
            counts[doc][part.lower()] += 1

    return {
        doc: {
            part for part, n in counter.items()
            if n / totals[doc] > settings.boilerplate_header_ratio
        }
        for doc, counter in counts.items()
    }


def looks_garbled(text: str) -> bool:
    """'S tand al one Fi nanci al' - letter-spaced text from a broken PDF."""
    words = text.split()
    return len(words) >= 3 and sum(len(w) <= 2 for w in words) / len(words) > 0.3


def page_area(page: dict, boilerplate: set[str]) -> str:
    """The informative part of the running header, e.g.
    'Notes to the Consolidated Financial Statements'."""
    parts = [
        p for p in page.get("page_header", "").split(" | ")
        if p
        and p.lower() not in boilerplate
        and len(re.findall(r"[A-Za-z]", p)) > 3   # drop page numbers like "2 87"
        and not looks_garbled(p)
    ]
    return " > ".join(parts[:2])


STATEMENT_TITLE = re.compile(
    r"(statements? of [a-z ,&]+|balance sheets?)", re.IGNORECASE
)


def statement_title(page: dict) -> str | None:
    """Financial statement titles are often printed vertically in the margin
    (e.g. Wipro), so they never appear as a Markdown heading."""
    for part in page.get("rotated_text", "").split(" | "):
        part = part.strip()
        if len(part) <= 60 and STATEMENT_TITLE.fullmatch(part.rstrip(". ")):
            return part
    return None


def build_context(page: dict, area: str, section: str) -> str:
    """One line that makes a chunk self-explanatory."""
    unit_label = UNIT_LABELS.get(
        (page.get("detected_currency") or page["currency"],
         page.get("detected_unit") or page["unit"])
    )
    parts = [
        page["company"],
        f"{page['period']} (year ended {page['fiscal_year_end']})",
        page["basis"].capitalize() if page.get("basis") else None,
        area or None,
        section if section and section.lower() not in area.lower() else None,
        f"Amounts in {unit_label}" if unit_label else None,
    ]
    return " | ".join(p for p in parts if p)


# ---------------------------------------------------------------- packing

def chunk_page(page: dict, section: str, area: str) -> tuple[list[dict], str]:
    """Turn one page into chunks. Returns (chunks, section_at_page_end)."""
    max_chars = settings.chunk_size
    pieces: list[dict] = []          # {"kind", "text", "section"}
    buffer: list[str] = []
    buffer_kinds: set[str] = set()
    buffer_section = section

    def flush():
        nonlocal buffer, buffer_kinds
        if buffer:
            kind = buffer_kinds.pop() if len(buffer_kinds) == 1 else "mixed"
            pieces.append({
                "kind": kind,
                "text": "\n\n".join(buffer),
                "section": buffer_section,
            })
        buffer, buffer_kinds = [], set()

    def buffer_len():
        return sum(len(b) + 2 for b in buffer)

    for block in parse_blocks(page["text"]):
        if block.kind == "heading":
            # A heading starts a new chunk and travels with its content.
            # (Unless the buffer is still tiny: then keep them together,
            # otherwise KPI pages full of headings make 50-char chunks.)
            if buffer_len() >= settings.min_chunk_chars:
                flush()
            section = block.text
            buffer_section = section
            buffer.append(f"## {block.text}")
            buffer_kinds.add("text")
            continue

        splitter = split_table if block.kind == "table" else split_text
        args = (max_chars,) if block.kind == "table" else (
            max_chars, settings.chunk_overlap
        )
        parts = splitter(block.text, *args)

        for i, part in enumerate(parts):
            if buffer and buffer_len() + len(part) > max_chars:
                # A lead-in like "Information on X is as follows:" (and a
                # heading right before it) belongs with the table it
                # introduces, not stranded at the end of the previous chunk.
                lead_in = []
                if block.kind == "table" and i == 0:
                    while (
                        buffer
                        and (buffer[-1].rstrip().endswith(":")
                             or buffer[-1].startswith("## "))
                        and sum(map(len, lead_in)) + len(buffer[-1]) <= 600
                    ):
                        lead_in.insert(0, buffer.pop())
                if buffer:
                    flush()
                else:
                    buffer_kinds = set()
                buffer_section = section
                if lead_in:
                    buffer.extend(lead_in)
                    buffer_kinds.add("text")
            buffer.append(part)
            buffer_kinds.add(block.kind)

    flush()

    # Merge a tiny trailing piece into the previous one (same page)
    if (
        len(pieces) >= 2
        and len(pieces[-1]["text"]) < settings.min_chunk_chars
        and len(pieces[-2]["text"]) + len(pieces[-1]["text"]) <= max_chars * 1.25
    ):
        last = pieces.pop()
        pieces[-1]["text"] += "\n\n" + last["text"]
        if pieces[-1]["kind"] != last["kind"]:
            pieces[-1]["kind"] = "mixed"

    chunks = []
    for i, piece in enumerate(pieces):
        context = build_context(page, area, piece["section"])
        chunks.append({
            "chunk_id": f"{page['document_id']}_p{page['page_number']}_c{i}",
            "document_id": page["document_id"],
            "document_name": page["document_name"],
            "company": page["company"],
            "report_type": page["report_type"],
            "period": page["period"],
            "fiscal_year_start": page["fiscal_year_start"],
            "fiscal_year_end": page["fiscal_year_end"],
            "currency": page.get("detected_currency") or page["currency"],
            "unit": page.get("detected_unit") or page["unit"],
            "basis": page.get("basis"),
            "source_url": page["source_url"],
            "page_number": page["page_number"],
            "printed_page": page.get("printed_page"),
            "page_area": area,
            "section": piece["section"] or "Unknown",
            "chunk_type": piece["kind"],
            "context": context,
            "text": piece["text"],
            # What we embed and keyword-index: context + content
            "embed_text": f"{context}\n\n{piece['text']}",
            "char_count": len(piece["text"]),
            "extraction_method": page.get("extraction_method"),
            "needs_review": page.get("needs_review", False),
            "chunker_version": CHUNKER_VERSION,
        })

    return chunks, section


# -------------------------------------------------------------------- main

def load_pages() -> list[dict]:
    if not settings.documents_path.exists():
        raise FileNotFoundError(
            f"{settings.documents_path} not found. Run ingest.py first."
        )
    with settings.documents_path.open("r", encoding="utf-8") as file:
        pages = [json.loads(line) for line in file if line.strip()]

    if pages and "extractor_version" not in pages[0]:
        raise ValueError(
            "documents.jsonl was made by the old extractor. Re-run ingest.py."
        )
    return pages


def chunk_documents() -> None:
    pages = load_pages()
    boilerplate = find_boilerplate_headers(pages)

    stats = Counter()
    current_doc, section, previous_area, previous_basis = None, "", "", None

    with settings.chunks_path.open("w", encoding="utf-8") as output:
        for page in pages:
            if page["document_id"] != current_doc:  # reset per document
                current_doc, section, previous_area = page["document_id"], "", ""
                previous_basis = None

            area = page_area(page, boilerplate.get(page["document_id"], set()))

            # Section carry-over: a section continues onto the next page,
            # unless the page tells us we are somewhere new.
            title = statement_title(page)
            if title:
                section = title
            elif area and area != previous_area:
                section = ""  # new part of the report; don't carry stale section
            # Basis carry-over: statement pages often state "Standalone" or
            # "Consolidated" only on the first page of the statement.
            if page.get("basis"):
                previous_basis = page["basis"]
            elif not area or area == previous_area:
                page["basis"] = previous_basis
            else:
                previous_basis = None

            previous_area = area or previous_area

            chunks, section = chunk_page(page, section, area)

            for chunk in chunks:
                output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                stats[chunk["chunk_type"]] += 1
                stats["unknown_section"] += chunk["section"] == "Unknown"
            stats["pages"] += 1

    total = stats["table"] + stats["text"] + stats["mixed"]
    logger.info(
        "Chunking complete: %d pages -> %d chunks "
        "(table %d, text %d, mixed %d) | unknown section: %d",
        stats["pages"], total, stats["table"], stats["text"], stats["mixed"],
        stats["unknown_section"],
    )
    logger.info("Output: %s", settings.chunks_path)


if __name__ == "__main__":
    setup_logging()
    chunk_documents()
