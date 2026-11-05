"""Layout-aware PDF page extraction.

For every page this produces:
  - text:              the page body as Markdown (tables become Markdown
                       tables), with running headers/footers removed
  - page_header:       the header/footer text we removed (a section signal)
  - printed_page:      the page number printed on the page, which often
                       differs from the PDF page index a viewer uses
  - rotated_text:      vertical margin text (often statement titles + units)
  - detected_*:        currency, unit and basis found on the page itself
  - extraction_method: "layout" or "ocr"
  - quality:           metrics telling us whether to trust the extraction
"""
import logging
import re
from collections import Counter

import pymupdf
import pymupdf4llm

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "layout-md-v1"

# --------------------------------------------------------------- patterns

# Some Indian reports draw the ₹ sign with a custom font in which ordinary
# letters (I, H, J, K, C, ...) are *drawn* as ₹. Text extraction returns the
# letter. We find these spans by their font name, never by guessing.
RUPEE_FONT = re.compile(r"rupee", re.IGNORECASE)
RUPEE = "₹"

HEADER_CLASSES = {"page-header", "page-footer"}

UNIT_PATTERNS = [
    # (regex, currency, unit) - order matters: first match wins
    (re.compile(r"(?:₹|\bINR|\bRs\.?)\s*(?:in\s+)?crores?\b", re.I), "INR", "crore"),
    (re.compile(r"(?:₹|\bINR|\bRs\.?)\s*(?:in\s+)?lakhs?\b", re.I), "INR", "lakh"),
    (re.compile(r"(?:₹|\bINR|\bRs\.?)\s*(?:in\s+)?millions?\b", re.I), "INR", "million"),
    (re.compile(r"in\s+millions\s+of\s+euros|€\s*(?:in\s+)?millions?|€m\b", re.I), "EUR", "million"),
    (re.compile(r"(?:US\$|\$|\bUSD)\s*(?:in\s+)?millions?\b", re.I), "USD", "million"),
]

# Quality signals
MERGED_WORD = re.compile(r"[a-z][A-Z]{2,}[a-z]")          # "TotalEXPENSESincome"
TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}", re.MULTILINE)
COMMON_SHORT_WORDS = set("""
an of to in is on at by or as be we it us no if so up do my me he am
the and for are was its has our not but all any new per net may can who
how out one two use set key due tax end via own fee had did get see top
way low few big pay non year
""".split())

# A table cell holding only a number, possibly with stray spaces:
# "(41, 527 )" -> "(41,527)",  "(46, 7 24)" -> "(46,724)"
NUMERIC_CELL = re.compile(r"(?<=\|)([^|\n]*\d[^|\n]*)(?=\|)")
NUMERIC_CELL_BODY = re.compile(r"^[\s*()\-–,.\d]+$")


# ---------------------------------------------------------------- helpers

def normalize_spaces(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fragmentation_ratio(text: str) -> float | None:
    """Share of words that look like broken fragments ("stak eh ol ers").

    Healthy pages score ~0.01. Pages whose PDF text layer is broken score
    0.08+. Returns None when the page has too few words to judge.
    """
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 30:
        return None
    fragments = sum(
        1 for w in words
        if w.islower() and 2 <= len(w) <= 3 and w not in COMMON_SHORT_WORDS
    )
    return fragments / len(words)


def scan_spans(page: pymupdf.Page) -> tuple[list[str], list[tuple[str, str, str]]]:
    """One pass over the page's text spans.

    Returns:
      rotated_lines: text of lines that are not horizontal
      rupee_hits:    (text_before, glyph, text_after) for every ₹ span
    """
    rotated_lines, rupee_hits = [], []

    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line["spans"]
            _, dy = line["dir"]

            if abs(dy) > 0.5:  # vertical or upside-down text
                rotated_lines.append("".join(s["text"] for s in spans))

            for i, span in enumerate(spans):
                glyph = span["text"].strip()
                if glyph and RUPEE_FONT.search(span["font"]):
                    before = "".join(s["text"] for s in spans[:i])
                    after = "".join(s["text"] for s in spans[i + 1:])
                    rupee_hits.append((before, glyph, after))

    return rotated_lines, rupee_hits


def fix_rupee(text: str, rupee_hits: list[tuple[str, str, str]]) -> tuple[str, int]:
    """Replace mis-mapped ₹ glyphs, only where the font says a ₹ was drawn.

    The surrounding characters of each ₹ span are used to find the same
    spot in the extracted text. Returns (fixed_text, replacements_made).
    """
    if not rupee_hits:
        return text, 0

    text = text.replace(" ", " ")
    glyphs = {glyph for _, glyph, _ in rupee_hits}
    glyph_class = "[" + re.escape("".join(sorted(glyphs))) + "]"
    replaced = 0

    # Markdown may wrap the glyph as inline code: "`H`" -> "H"
    text = re.sub(r"`(" + glyph_class + r")`", r"\1", text)

    # 1. Context-anchored: "(H crore)" -> "(₹ crore)", "|I4,205|" -> "|₹4,205|"
    #    Markdown may drop spaces or add bold markers (**) around the glyph,
    #    so we anchor on the nearest *word* on each side and allow both.
    gap = r"\s*(?:\*\*|_)?\s*"
    contexts = Counter()
    for before, glyph, after in rupee_hits:
        before_words = before.replace("\u00a0", " ").split()
        after_words = after.replace("\u00a0", " ").split()
        before = before_words[-1][-6:] if before_words else ""
        after = after_words[0][:8] if after_words else ""
        if before or after:
            contexts[(before, glyph, after)] += 1

    for before, glyph, after in contexts:
        pattern = (
            (re.escape(before) + gap if before else r"(?<![A-Za-z])")
            + re.escape(glyph)
            + (r"(?=" + gap + re.escape(after) + ")" if after else r"(?![A-Za-z])")
        )
        text, n = re.subn(
            pattern, lambda m, g=glyph: m.group(0)[: -len(g)] + RUPEE, text
        )
        replaced += n

    # 2. A table cell or line holding only the glyph: "| **I** |"
    cell_only = re.compile(
        r"(?<=\|)(\s*(?:\*\*)?)" + glyph_class + r"((?:\*\*)?\s*(?:-\s*)?)(?=\|)"
    )
    text, n = cell_only.subn(lambda m: m.group(1) + RUPEE + m.group(2), text)
    replaced += n

    line_only = re.compile(
        r"^(\s*(?:\*\*)?)" + glyph_class + r"((?:\*\*)?\s*)$", re.MULTILINE
    )
    text, n = line_only.subn(lambda m: m.group(1) + RUPEE + m.group(2), text)
    replaced += n

    return text, replaced


def count_rupee_leftovers(text: str, rupee_hits) -> int:
    """How many glyph letters still sit right before a number?

    This is the quality signal we flag on. Comparing 'found' with 'fixed'
    over-reports, because the Markdown converter sometimes drops a glyph
    entirely (nothing left to fix, and nothing wrong in the text).
    """
    if not rupee_hits:
        return 0
    glyphs = "".join(sorted({glyph for _, glyph, _ in rupee_hits}))
    pattern = (
        r"(?<![A-Za-z])[" + re.escape(glyphs) + r"](?=\s*(?:\*\*)?\s*\(?\d)"
    )
    return len(re.findall(pattern, text))


def repair_numeric_cells(text: str) -> tuple[str, int]:
    """Remove stray spaces inside purely numeric table cells."""
    repaired = 0

    def fix(match):
        nonlocal repaired
        cell = match.group(1)
        if not NUMERIC_CELL_BODY.match(cell):
            return cell  # contains words: leave it alone
        inner = cell.strip()
        squeezed = re.sub(r"(?<=[\d,(])\s+(?=[\d,)])", "", inner)
        if squeezed != inner:
            repaired += 1
        return cell.replace(inner, squeezed)

    return NUMERIC_CELL.sub(fix, text), repaired


def collapse_letter_spacing(text: str) -> str:
    """'Integrated Annual Report 2 0 2 5- 2 6' -> '... 2025- 26'."""
    return re.sub(r"(?<=\b\w) (?=\w\b)", "", text)


def detect_unit(*texts: str) -> tuple[str | None, str | None]:
    joined = "\n".join(t for t in texts if t)
    for pattern, currency, unit in UNIT_PATTERNS:
        if pattern.search(joined):
            return currency, unit
    return None, None


def detect_basis(*texts: str) -> str | None:
    """'standalone' / 'consolidated' from headers and margin titles."""
    # Remove all whitespace first, so letter-spaced text ("S tand al one")
    # still matches.
    joined = re.sub(r"\s+", "", " ".join(t for t in texts if t)).lower()
    cons, stand = "consolidated" in joined, "standalone" in joined
    if cons != stand:
        return "consolidated" if cons else "standalone"
    return None


def ocr_page_text(page: pymupdf.Page, tessdata: str | None) -> str | None:
    """OCR the rendered page, ignoring its (broken) text layer."""
    try:
        textpage = page.get_textpage_ocr(full=True, dpi=300, tessdata=tessdata)
        return normalize_spaces(page.get_text(textpage=textpage, sort=True))
    except Exception as error:  # Tesseract missing / misconfigured
        logger.debug("OCR failed on page %d: %s", page.number + 1, error)
        return None


# ---------------------------------------------------------------- main API

def extract_pdf(
    pdf_path,
    use_ocr_fallback: bool = True,
    tessdata: str | None = None,
    broken_text_threshold: float = 0.08,
    show_progress: bool = True,
):
    """Yield one record per page with layout-aware text and diagnostics."""
    ocr_available = use_ocr_fallback  # switched off after the first failure

    with pymupdf.open(pdf_path) as doc:
        page_results = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,   # one result per page, with layout boxes
            header=False,       # keep running headers out of the body...
            footer=False,       # ...we capture them separately below
            use_ocr=False,      # OCR is decided per page, further down
            show_progress=show_progress,
        )

        for result in page_results:
            page_number = result["metadata"]["page_number"]  # 1-based
            page = doc[page_number - 1]

            rotated_lines, rupee_hits = scan_spans(page)

            # ---- headers / footers, located by the layout model
            header_lines = []
            for box in result.get("page_boxes", []):
                if box["class"] in HEADER_CLASSES:
                    rect = pymupdf.Rect(box["bbox"]) + (-2, -2, 2, 2)
                    for line in page.get_textbox(rect).splitlines():
                        line = collapse_letter_spacing(normalize_spaces(line))
                        if line:
                            header_lines.append(line)

            printed_page = next(
                (l for l in header_lines if re.fullmatch(r"\d{1,4}", l)), None
            )
            header_text = " | ".join(l for l in header_lines if l != printed_page)

            # ---- body
            method = "layout"
            body = result["text"]
            body, fixed_body = fix_rupee(body, rupee_hits)
            body, numeric_repairs = repair_numeric_cells(body)
            body = normalize_spaces(body)
            table_count = len(TABLE_SEPARATOR.findall(body))

            # ---- broken text layer? (judge the raw PDF text, not ours)
            frag = fragmentation_ratio(page.get_text())
            broken = frag is not None and frag > broken_text_threshold

            # Prose pages with a broken text layer: OCR gives correct words.
            # Table pages: keep the layout version, because table structure
            # matters more and the numeric-cell repair fixes the numbers.
            if broken and table_count == 0 and ocr_available:
                ocr_text = ocr_page_text(page, tessdata)
                if ocr_text:
                    body, method = ocr_text, "ocr"
                    if rupee_hits:
                        # OCR reads the ₹ glyph as "2": "(2 crore)"
                        body = re.sub(r"\(\s*2\s+(?=(?:in\s+)?(?:crore|lakh|million))",
                                      "(" + RUPEE + " ", body)
                else:
                    ocr_available = False
                    logger.warning(
                        "OCR unavailable (is Tesseract installed?). "
                        "Broken pages will be flagged instead."
                    )

            header_text, fixed_head = fix_rupee(header_text, rupee_hits)
            rotated_text, fixed_rot = fix_rupee(
                " | ".join(normalize_spaces(l) for l in rotated_lines if l.strip()),
                rupee_hits,
            )

            currency, unit = detect_unit(rotated_text, header_text, body[:1500])

            yield {
                "page_number": page_number,
                "page_index": page_number - 1,
                "printed_page": printed_page,
                "text": body,
                "page_header": header_text,
                "rotated_text": rotated_text,
                "detected_currency": currency,
                "detected_unit": unit,
                "basis": detect_basis(header_text, rotated_text),
                "extraction_method": method,
                "image_count": len(page.get_images(full=True)),
                "quality": {
                    "char_count": len(body),
                    "table_count": table_count,
                    "merged_word_count": len(MERGED_WORD.findall(body)),
                    "fragmentation": None if frag is None else round(frag, 3),
                    "broken_text_layer": broken,
                    "numeric_cells_repaired": numeric_repairs,
                    "rupee_glyphs_found": len(rupee_hits),
                    "rupee_glyphs_fixed": fixed_body + fixed_head + fixed_rot,
                    "rupee_leftovers": (
                        0 if method == "ocr"
                        else count_rupee_leftovers(body, rupee_hits)
                    ),
                },
            }
