"""PDF text extraction for V3 pipeline.

Uses PyMuPDF (fitz) to extract full text from PDF files.
Also provides reference section detection for citation context extraction.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


_REF_SECTION_PATTERNS = [
    r"\nReferences\s*\n",
    r"\nREFERENCES\s*\n",
    r"\nBibliography\s*\n",
    r"\nBIBLIOGRAPHY\s*\n",
    r"\n\s*R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S\s*\n",
]

# Section header patterns for results/experiments
_RESULTS_SECTION_PATTERNS = [
    re.compile(r"(?i)^(experiments?(?:\s+(?:and|&)\s+results?)?)$"),
    re.compile(r"(?i)^(results?(?:\s+(?:and|&)\s+(?:analysis|discussion))?)$"),
    re.compile(r"(?i)^(experimental\s+(?:results?|setup|evaluation))$"),
    re.compile(r"(?i)^(evaluation(?:\s+(?:results?|on))?)$"),
    re.compile(r"(?i)^(main\s+results?)$"),
    re.compile(r"(?i)^(quantitative\s+(?:results?|evaluation|analysis))$"),
    re.compile(r"(?i)^(benchmark\s+(?:results?|evaluation))$"),
]

_INTRO_SECTION_PATTERNS = [
    re.compile(r"(?i)^(introduction|intro)$"),
]

_CONCLUSION_SECTION_PATTERNS = [
    re.compile(r"(?i)^(conclusion|discussion|concluding\s+remarks|limitations?\s+(?:and|&)\s+future|summary)$"),
]

# Table-like content patterns to filter
_TABLE_START_PATTERNS = [
    re.compile(r"\\begin\{(?:table|tabular|longtable|supertabular)\*?\}"),
    re.compile(r"^\s*\|[\s\-:|]+\|\s*$", re.MULTILINE),  # Markdown table header sep
]

_LINE_OF_NUMBERS = re.compile(r"^\s*([\d\.\-+eE]+[\s,;|]+){4,}[\d\.\-+eE]+\s*$")
_LATEX_TABLE_ROW = re.compile(r"^\s*([\d\.\-+eE]+\s*[&]\s*){3,}[\d\.\-+eE]+\s*\\\\\s*$")


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract full text from a PDF file.

    Returns the concatenated text of all pages, separated by page markers.
    Raises ImportError if PyMuPDF is not installed.
    """
    if not HAS_FITZ:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF text extraction. "
            "Install it with: pip install PyMuPDF"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            pages.append(f"[Page {page_num}]\n{text}")
    doc.close()

    return "\n\n".join(pages)


def find_references_section(full_text: str) -> Optional[str]:
    """Extract the references section from the full text, if detectable.

    Searches for common reference section headings and returns everything after
    the last match.
    """
    best_pos = -1
    for pattern in _REF_SECTION_PATTERNS:
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
        if matches:
            pos = matches[-1].start()
            if pos > best_pos:
                best_pos = pos

    if best_pos < 0:
        return None

    return full_text[best_pos:]


def estimate_token_count(text: str) -> int:
    """Rough estimate of token count (4 chars ≈ 1 token for English)."""
    return len(text) // 4


def truncate_text(text: str, max_chars: int = 48000) -> str:
    """Truncate text to approximately max_chars characters at the last paragraph boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to cut at last double newline
    last_break = truncated.rfind("\n\n")
    if last_break > max_chars * 0.7:
        return truncated[:last_break]
    return truncated


# ── Section extraction for one-shot analysis ──────────────────────────

def _find_all_section_headers(full_text: str) -> list[tuple[int, int, str]]:
    """Find all section headers in text. Returns [(start, end, title), ...]."""
    cleaned = re.sub(r"[ \t]+", " ", full_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    section_pattern = re.compile(
        r"(?:^|\n)\s*"
        r"(?:(?:[IVX]+|\d+)[\.\)]\s+)?"         # optional number prefix
        r"([A-Z][A-Za-z\s\-&]+)"                  # section title
        r"(?:\s*\.{3,})?"                         # optional dots (TOC)
        r"\s*\n",
        re.MULTILINE,
    )

    headers = []
    for m in section_pattern.finditer(cleaned):
        title = m.group(1).strip()
        if len(title.split()) <= 8:  # Skip long non-header lines
            headers.append((m.start(), m.end(), title))
    return headers


def _filter_table_content(text: str) -> str:
    """Remove table-like content, keeping only textual paragraphs.

    Removes: LaTeX table environments, markdown tables, lines dominated by numbers.
    Keeps: paragraphs of prose text that discuss/interpret results.
    """
    if not text:
        return ""

    # Remove LaTeX table environments
    text = re.sub(
        r"\\begin\{(?:table|tabular|longtable|supertabular|tabularx)\*?\}.*?"
        r"\\end\{(?:table|tabular|longtable|supertabular|tabularx)\*?\}",
        "", text, flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove markdown tables: continuous lines starting/ending with |
    lines = text.split("\n")
    filtered = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        # Markdown table line: starts with |, has content, ends with |
        is_table_line = bool(re.match(r"^\|.+\|$", stripped))
        # Separator line like |---|---|
        is_sep_line = bool(re.match(r"^\|[\s\-:|]+\|$", stripped))
        # Number-dominated line
        is_number_line = bool(_LINE_OF_NUMBERS.match(stripped))
        # LaTeX table row
        is_latex_row = bool(_LATEX_TABLE_ROW.match(stripped))

        if is_table_line or is_sep_line:
            in_table = True
            continue
        elif is_number_line or is_latex_row:
            continue
        else:
            # End of table
            if in_table and stripped:
                in_table = False
            in_table = False
            filtered.append(line)

    result = "\n".join(filtered)
    # Collapse multiple blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def extract_results_section(full_text: str, max_chars: int = 3000) -> str:
    """Extract textual conclusions from the Results/Experiments section.

    Finds the section, removes tables and number-heavy rows, keeps prose
    that summarizes and interprets experimental findings.
    """
    if not full_text:
        return ""

    cleaned = re.sub(r"[ \t]+", " ", full_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    headers = _find_all_section_headers(cleaned)

    # Find the first results/experiments section
    results_start = None
    results_title = ""
    for start, end, title in headers:
        for pat in _RESULTS_SECTION_PATTERNS:
            if pat.match(title):
                results_start = end
                results_title = title
                break
        if results_start is not None:
            break

    if results_start is None:
        # Fallback: scan for "experiment" or "result" keyword in text
        for keyword in ["\nExperiment", "\nExperiments", "\nResults", "\nEvaluation"]:
            pos = cleaned.find(keyword)
            if pos >= 0:
                # Find next newline after keyword
                nl = cleaned.find("\n", pos + len(keyword))
                if nl >= 0:
                    # Check if it looks like a section header (next line is empty or starts with number)
                    after = cleaned[nl:nl + 100].strip()
                    if not after or after[0].isdigit():
                        results_start = nl + 1
                        results_title = keyword.strip()
                        break
        if results_start is None:
            return ""

    # Find the end: next section header or max_chars
    results_end = results_start + max_chars
    for start, end, title in headers:
        if start > results_start and start < results_end:
            results_end = start
            break

    raw = cleaned[results_start:results_end].strip()
    return _filter_table_content(raw)[:max_chars]


def extract_introduction_section(full_text: str, max_chars: int = 3000) -> str:
    """Extract the introduction section from full text."""
    if not full_text:
        return ""

    cleaned = re.sub(r"[ \t]+", " ", full_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    headers = _find_all_section_headers(cleaned)

    intro_start = None
    for start, end, title in headers:
        for pat in _INTRO_SECTION_PATTERNS:
            if pat.match(title):
                intro_start = end
                break
        if intro_start is not None:
            break

    if intro_start is None:
        # Fallback: use first ~max_chars
        return cleaned[:max_chars].strip()

    intro_end = intro_start + max_chars
    for start, end, title in headers:
        if start > intro_start and start < intro_end:
            intro_end = start
            break

    return cleaned[intro_start:intro_end].strip()[:max_chars]


def extract_conclusion_section(full_text: str, max_chars: int = 2000) -> str:
    """Extract the conclusion/discussion section from full text."""
    if not full_text:
        return ""

    cleaned = re.sub(r"[ \t]+", " ", full_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    headers = _find_all_section_headers(cleaned)

    # Find conclusion — prefer the LAST one (some papers have "Discussion" then "Conclusion")
    conclusion_start = None
    for start, end, title in headers:
        for pat in _CONCLUSION_SECTION_PATTERNS:
            if pat.match(title):
                conclusion_start = end
                # Don't break — keep looking for later conclusion

    if conclusion_start is None:
        # Fallback: last ~max_chars of text
        return cleaned[-max_chars:].strip()

    conclusion_end = conclusion_start + max_chars
    for start, end, title in headers:
        if start > conclusion_start and start < conclusion_end:
            conclusion_end = start
            break

    return cleaned[conclusion_start:conclusion_end].strip()[:max_chars]


def assemble_paper_text_for_one_shot(paper) -> str:
    """Assemble per-paper text for one-shot analysis.

    Format: title + year + month + abstract + introduction + results text + conclusion.

    Args:
        paper: Paper object with title, year, month, abstract, full_text.

    Returns:
        Formatted string ready for the one-shot prompt.
    """
    from paper import Paper

    year_month = str(paper.year)
    if getattr(paper, "month", 0):
        year_month = f"{paper.year}-{paper.month:02d}"

    parts = [f"Title: {paper.title}", f"Year: {year_month}"]

    if paper.abstract:
        parts.append(f"Abstract: {paper.abstract}")

    full_text = paper.full_text or ""

    intro = extract_introduction_section(full_text)
    if intro:
        parts.append(f"Introduction: {intro}")

    results = extract_results_section(full_text)
    if results:
        parts.append(f"Results (textual conclusions, tables removed): {results}")

    conclusion = extract_conclusion_section(full_text)
    if conclusion:
        parts.append(f"Conclusion: {conclusion}")

    return "\n\n".join(parts)
