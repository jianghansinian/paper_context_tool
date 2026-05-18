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
