"""Paper resolver for V3 pipeline.

Handles resolving a user-provided arXiv URL or PDF file path into a Paper object
with metadata from arXiv API, Semantic Scholar API, and local PDF extraction.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests

import config
from paper import Paper
from text_extractor import extract_text_from_pdf

_ARXIV_URL_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$",
    re.IGNORECASE,
)
_ARXIV_API_BASE = "https://export.arxiv.org/api/query"
_SS_API_BASE = "https://api.semanticscholar.org/graph/v1"
_ARXIV_HEADERS = {"User-Agent": "PaperContextTool/1.0 (mailto:research@example.com)"}


def _extract_arxiv_id(url_or_id: str) -> Optional[str]:
    """Extract arXiv ID from a URL or return the ID if already in canonical form."""
    s = url_or_id.strip()
    # Already a canonical ID
    if re.match(r"^\d{4}\.\d{4,5}$", s):
        return s
    m = _ARXIV_URL_RE.search(s)
    if m:
        return m.group(1)
    # Try to extract from any URL containing an arxiv ID pattern
    m = re.search(r"(\d{4}\.\d{4,5})", s)
    return m.group(1) if m else None


def _is_pdf_path(s: str) -> bool:
    """Check if the string looks like a local PDF file path."""
    path = Path(s)
    return path.exists() and path.suffix.lower() == ".pdf"


def _fetch_arxiv_metadata(arxiv_id: str) -> Optional[dict]:
    """Fetch paper metadata from arXiv API with retry for rate limits."""
    url = f"{_ARXIV_API_BASE}?id_list={arxiv_id}&max_results=1"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=_ARXIV_HEADERS,
                              timeout=(10, config.HTTP_TIMEOUT_SEC))
            if resp.status_code in (429, 503):
                wait = (attempt + 1) * 5
                print(f"arXiv API rate-limited (HTTP {resp.status_code}), "
                      f"retrying in {wait}s (attempt {attempt + 1}/3)...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.exceptions.Timeout as exc:
            print(f"arXiv API request timed out: {exc}")
            return None
        except requests.exceptions.ConnectionError as exc:
            print(f"arXiv API connection failed: {exc}")
            return None
        except Exception as exc:
            if attempt < 2:
                wait = (attempt + 1) * 3
                print(f"arXiv API request failed ({exc}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"arXiv API request failed after 3 attempts: {exc}")
            return None
    else:
        print(f"arXiv API rate-limited after 3 attempts, giving up.")
        return None

    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    entries = root.findall("atom:entry", ns)
    if not entries:
        return None

    entry = entries[0]
    title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
    abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
    published = entry.findtext("atom:published", default="", namespaces=ns)
    year = int(published[:4]) if published and published[:4].isdigit() else 0
    month = int(published[5:7]) if published and len(published) >= 7 and published[5:7].isdigit() else 0

    authors = []
    for author_node in entry.findall("atom:author", ns):
        name = author_node.findtext("atom:name", default="", namespaces=ns)
        if name:
            authors.append(name.strip())

    link = ""
    for link_node in entry.findall("atom:link", ns):
        if link_node.attrib.get("rel") == "alternate":
            link = link_node.attrib.get("href", "")
            break
    if not link:
        link = f"https://arxiv.org/abs/{arxiv_id}"

    return {
        "title": title,
        "abstract": abstract,
        "year": year,
        "month": month,
        "authors": authors,
        "url": link,
    }


def _fetch_ss_metadata_by_arxiv(arxiv_id: str) -> Optional[dict]:
    """Fetch paper metadata from Semantic Scholar by arXiv ID."""
    url = f"{_SS_API_BASE}/paper/ArXiv:{arxiv_id}"
    params = {
        "fields": "title,authors,year,abstract,citationCount,externalIds,url"
    }
    headers = {}
    if config.SS_API_KEY:
        headers["x-api-key"] = config.SS_API_KEY
    try:
        resp = requests.get(url, params=params, headers=headers,
                            timeout=config.SS_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if not data or "paperId" not in data:
        return None

    authors = [a.get("name", "") for a in data.get("authors", [])]
    return {
        "ss_id": data.get("paperId", ""),
        "title": data.get("title", ""),
        "abstract": data.get("abstract", ""),
        "year": data.get("year") or 0,
        "authors": authors,
        "citation_count": data.get("citationCount", 0),
        "url": data.get("url", f"https://arxiv.org/abs/{arxiv_id}"),
    }


def _download_arxiv_pdf(arxiv_id: str, timeout: int = 20) -> Optional[Path]:
    """Download PDF from arXiv with retry and return the local path.

    Retries up to 3 times with exponential backoff. On rate-limit (429)
    waits longer. Returns None only after exhausting all attempts.
    """
    cache_dir = config.PAPER_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = cache_dir / f"{arxiv_id}.pdf"
    if pdf_path.exists():
        return pdf_path

    urls = [
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        f"https://arxiv.org/pdf/{arxiv_id}",
    ]

    last_error = ""
    for attempt in range(3):
        try:
            resp = requests.get(urls[0], timeout=timeout)
            if resp.status_code == 429:
                wait = (attempt + 1) * 8
                last_error = "HTTP 429 rate-limited"
                time.sleep(wait)
                continue
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
            return pdf_path
        except requests.exceptions.Timeout:
            last_error = f"timeout ({timeout}s)"
        except requests.exceptions.ConnectionError as exc:
            last_error = f"connection error: {exc}"
        except Exception as exc:
            last_error = str(exc)

        if attempt < 2:
            wait = (attempt + 1) * 2
            time.sleep(wait)

    # Final fallback: try URL without .pdf suffix
    try:
        resp = requests.get(urls[1], timeout=timeout)
        resp.raise_for_status()
        pdf_path.write_bytes(resp.content)
        return pdf_path
    except Exception:
        pass

    # Last resort: try Semantic Scholar open access PDF
    try:
        ss_url = f"{_SS_API_BASE}/paper/ArXiv:{arxiv_id}"
        ss_params = {"fields": "openAccessPdf"}
        ss_resp = requests.get(ss_url, params=ss_params, headers=(
            {"x-api-key": config.SS_API_KEY} if config.SS_API_KEY else {}
        ), timeout=15)
        ss_resp.raise_for_status()
        oa = (ss_resp.json().get("openAccessPdf") or {})
        oa_url = oa.get("url")
        if oa_url:
            oa_resp = requests.get(oa_url, timeout=timeout)
            oa_resp.raise_for_status()
            pdf_path.write_bytes(oa_resp.content)
            return pdf_path
    except Exception:
        pass

    return None


def _parse_metadata_from_text(text: str) -> dict:
    """Heuristically extract title and authors from the first page of a paper."""
    lines = text.strip().split("\n")
    # Title is usually the first non-empty, non-affiliation line
    title = ""
    for line in lines[:20]:
        line = line.strip()
        if len(line) > 10:
            title = line
            break

    # Try to find year from first page
    year_match = re.search(r"\b(20\d{2})\b", "\n".join(lines[:50]))
    year = int(year_match.group(1)) if year_match else 0

    # Extract authors
    authors = []
    _NON_AUTHOR_WORDS = {"propose", "method", "approach", "model", "framework",
                         "introduce", "present", "demonstrate", "achieve",
                         "abstract", "experiment", "result", "problem",
                         "however", "therefore", "furthermore", "paper",
                         "index terms", "keywords"}

    def _has_affiliation(line: str) -> bool:
        return bool(re.search(r"[*†‡§¶#‖]\d*|\d{1,2}(?:\s|,|$)", line))

    def _clean_name(name: str) -> str:
        name = re.sub(r"[*†‡§¶#‖]\d*", "", name)        # remove marker chars + optional digits
        name = re.sub(r"^\d{1,2}\s*", "", name)          # leading superscript numbers
        name = re.sub(r"[,\s]*\d{1,2}[,\s]*[A-Z]?", "", name)  # trailing numbers + optional letter
        return name.strip()

    def _is_affiliation_line(line: str) -> bool:
        """Check if a line looks like an affiliation rather than an author name."""
        lower = line.lower()
        return any(w in lower for w in {
            "university", "institute", "department", "college", "school",
            "laboratory", "lab", "research", "center", "ltd", "inc", "corp",
            "gmbh", "llc", "china", "usa", "france", "germany", "japan",
            "equal contribution", "corresponding author", "these authors",
            "work done", "internship",
        })

    # Strategy 1: consecutive lines each with affiliation markers (common: one author per line)
    for i, line in enumerate(lines[:40]):
        stripped = line.strip()
        if len(stripped) < 5:
            continue
        lower = stripped.lower()
        if any(w in lower for w in _NON_AUTHOR_WORDS):
            continue
        if not _has_affiliation(stripped):
            continue

        # Collect consecutive lines with affiliation markers
        candidate_names = []
        for j in range(i, min(i + 15, len(lines))):
            nl = lines[j].strip()
            if len(nl) < 3:
                if candidate_names:
                    break
                continue
            nl_lower = nl.lower()
            if any(w in nl_lower for w in _NON_AUTHOR_WORDS):
                break
            if _has_affiliation(nl) and not _is_affiliation_line(nl):
                name = _clean_name(nl)
                if len(name) > 1 and len(name) < 40:
                    candidate_names.append(name)
            elif candidate_names and not _has_affiliation(nl):
                break  # end of author block

        if len(candidate_names) >= 2:
            authors = candidate_names
            break

    # Strategy 2: space-separated authors with markers on one or two lines
    # e.g. "Linhan Wang * 1 Zichong Yang 2 Chen Bai 3 ..."
    if not authors:
        for i, line in enumerate(lines[:40]):
            stripped = line.strip()
            if len(stripped) < 10:
                continue
            lower = stripped.lower()
            if any(w in lower for w in _NON_AUTHOR_WORDS):
                continue
            if not _has_affiliation(stripped):
                continue
            # Must have space-separated sections but no commas
            if "," in stripped:
                continue
            # Split on affiliation markers: replace "marker N" or "marker" with delimiter
            delimited = re.sub(r"\s*[*†‡§¶#‖]\s*\d*\s*", "|", stripped)
            # Also split on standalone digits preceded by spaces
            delimited = re.sub(r"\s+\d{1,2}\s+", " | ", delimited)
            delimited = re.sub(r"\s+\d{1,2}$", "|", delimited)
            parts = [p.strip() for p in delimited.split("|") if p.strip()]
            name_parts = []
            for p in parts:
                cleaned = re.sub(r"\d{1,2}$", "", p).strip()  # trailing digit
                if len(cleaned) > 2 and len(cleaned) < 40 and not _is_affiliation_line(cleaned):
                    name_parts.append(cleaned)
            if len(name_parts) >= 2:
                # Check next line for continuation authors (same format, no commas)
                if i + 1 < len(lines):
                    nl = lines[i + 1].strip()
                    nl_lower = nl.lower()
                    if (not any(w in nl_lower for w in _NON_AUTHOR_WORDS)
                          and _has_affiliation(nl) and "," not in nl):
                        delimited2 = re.sub(r"\s*[*†‡§¶#‖]\s*\d*\s*", "|", nl)
                        delimited2 = re.sub(r"\s+\d{1,2}\s+", " | ", delimited2)
                        delimited2 = re.sub(r"\s+\d{1,2}$", "|", delimited2)
                        parts2 = [p.strip() for p in delimited2.split("|") if p.strip()]
                        for p in parts2:
                            cleaned = re.sub(r"\d{1,2}$", "", p).strip()
                            if len(cleaned) > 2 and len(cleaned) < 40 and not _is_affiliation_line(cleaned):
                                name_parts.append(cleaned)
                authors = name_parts
                break

    # Strategy 3: clean comma-separated authors (no affiliation markers)
    # e.g. "Yuzhou Huang, Benjin Zhu, Hengtong Lu, Victor Shea-Jay Huang"
    if not authors:
        for i, line in enumerate(lines[:30]):
            stripped = line.strip()
            if len(stripped) < 15:
                continue
            lower = stripped.lower()
            if any(w in lower for w in _NON_AUTHOR_WORDS):
                continue
            if _has_affiliation(stripped):
                continue
            # Must have commas and look like names (not a sentence)
            if "," not in stripped:
                continue
            parts = [p.strip() for p in stripped.split(",")]
            # Each part should be a short name
            if all(len(p) > 2 and len(p) < 40 for p in parts) and len(parts) >= 2:
                # Check that these look like names (2-4 words max per part)
                if all(p.count(" ") <= 3 for p in parts):
                    authors = parts
                    break

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": "",
        "url": "",
    }


def resolve_paper(url_or_path: str, user_description: str = "") -> Paper:
    """Resolve a paper from an arXiv URL, PDF path, or other identifier.

    Args:
        url_or_path: arXiv URL, local PDF file path, or arXiv ID.
        user_description: Optional natural language description from the user.

    Returns:
        A Paper object with metadata populated. full_text is extracted if a PDF
        is available.
    """
    # --- Case 1: Local PDF file ---
    if _is_pdf_path(url_or_path):
        path = Path(url_or_path)
        print(f"Resolving from local PDF: {path}")
        full_text = extract_text_from_pdf(path)
        meta = _parse_metadata_from_text(full_text)
        paper = Paper(
            id=f"local:{path.stem}",
            title=meta["title"],
            authors=meta["authors"],
            year=meta["year"],
            abstract=meta["abstract"],
            full_text=full_text,
            url=url_or_path,
            source="pdf_file",
            user_description=user_description,
        )
        return paper

    # --- Case 2: arXiv URL or ID ---
    arxiv_id = _extract_arxiv_id(url_or_path)
    if arxiv_id:
        print(f"Resolving arXiv paper: {arxiv_id}")

        # Fetch metadata from arXiv
        arxiv_meta = _fetch_arxiv_metadata(arxiv_id)

        # Enrich with Semantic Scholar
        ss_meta = _fetch_ss_metadata_by_arxiv(arxiv_id)

        # Merge metadata (SS takes priority where both available)
        merged = arxiv_meta or {}
        if ss_meta:
            merged["title"] = ss_meta.get("title") or merged.get("title", "")
            merged["abstract"] = ss_meta.get("abstract") or merged.get("abstract", "")
            merged["citation_count"] = ss_meta.get("citation_count", 0)
            merged["ss_id"] = ss_meta.get("ss_id", "")
            merged["url"] = ss_meta.get("url") or merged.get("url", "")
            if ss_meta.get("authors"):
                merged["authors"] = ss_meta["authors"]
            if ss_meta.get("year"):
                merged["year"] = ss_meta["year"]

        paper_id = merged.get("ss_id", "") or f"arxiv:{arxiv_id}"

        # Download PDF and extract text
        pdf_path = _download_arxiv_pdf(arxiv_id)
        full_text = None
        if pdf_path:
            try:
                full_text = extract_text_from_pdf(pdf_path)
            except Exception as exc:
                print(f"Text extraction failed: {exc}")

        # Fallback: parse metadata from extracted text if APIs failed
        if (not merged.get("title") or not merged.get("authors")) and full_text:
            parsed = _parse_metadata_from_text(full_text)
            if not merged.get("title"):
                merged["title"] = parsed.get("title", "")
            if not merged.get("authors"):
                merged["authors"] = parsed.get("authors", [])
            if not merged.get("year"):
                merged["year"] = parsed.get("year", 0)

        paper = Paper(
            id=paper_id,
            arxiv_id=arxiv_id,
            title=merged.get("title", ""),
            authors=merged.get("authors", []),
            year=merged.get("year", 0),
            abstract=merged.get("abstract", ""),
            full_text=full_text,
            citation_count=merged.get("citation_count", 0),
            url=merged.get("url", f"https://arxiv.org/abs/{arxiv_id}"),
            source="arxiv",
            user_description=user_description,
        )
        return paper

    # --- Case 3: Other URL — try Semantic Scholar ---
    print(f"Trying to resolve via Semantic Scholar: {url_or_path}")
    try:
        resp = requests.get(
            f"{_SS_API_BASE}/paper/search",
            params={"query": url_or_path, "limit": 1,
                     "fields": "title,authors,year,abstract,citationCount,externalIds,url"},
            headers={"x-api-key": config.SS_API_KEY} if config.SS_API_KEY else {},
            timeout=config.SS_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        papers = data.get("data", [])
        if papers:
            p = papers[0]
            authors = [a.get("name", "") for a in p.get("authors", [])]
            external = p.get("externalIds", {})
            arxiv_id = external.get("ArXiv", "")
            return Paper(
                id=p.get("paperId", ""),
                arxiv_id=arxiv_id,
                title=p.get("title", ""),
                authors=authors,
                year=p.get("year") or 0,
                abstract=p.get("abstract", "") or "",
                citation_count=p.get("citationCount", 0),
                url=p.get("url", ""),
                source="semantic_scholar",
                user_description=user_description,
            )
    except Exception as exc:
        print(f"Semantic Scholar search failed: {exc}")

    raise ValueError(
        f"Could not resolve paper from: {url_or_path}\n"
        "Please provide a valid arXiv URL (e.g. https://arxiv.org/abs/2203.17270), "
        "arXiv ID (e.g. 2203.17270), or local PDF file path."
    )
