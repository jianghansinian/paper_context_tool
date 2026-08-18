"""OpenCitations/COCI API client — third fallback data source for V3.1.

COCI (OpenCitations Index of Crossref open DOI-to-DOI citations):
  - 1.5B+ citation links, CC0 license, no API key required
  - REST endpoints: /citations/{doi} (forward) and /references/{doi} (backward)
  - Returns DOI→DOI links only — no paper metadata
  - Limitation: DOI-dependent. Papers without registered DOIs are skipped.

Used only when both Semantic Scholar and OpenAlex APIs fail.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional

_COCI_API = "https://opencitations.net/index/api/v1"
_COCI_TIMEOUT = 20
_COCI_USER_AGENT = "paper-context-tool/3.1 (mailto:research@example.com)"


def _coci_request(endpoint: str, doi: str) -> list[dict] | None:
    """Make a request to OpenCitations COCI API. Returns list of citation entries or None."""
    doi_enc = urllib.parse.quote(doi, safe="")
    url = f"{_COCI_API}/{endpoint}/{doi_enc}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", _COCI_USER_AGENT)
            with urllib.request.urlopen(req, timeout=_COCI_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return None
            if e.code == 429:
                time.sleep((attempt + 1) * 5)
                continue
            if attempt < 2:
                time.sleep(2)
                continue
            return None
        except Exception:
            if attempt < 2:
                time.sleep(2)
                continue
            return None
    return None


def _extract_doi(paper: dict) -> str:
    """Extract DOI from a paper dict (tries _doi field first, then other sources)."""
    doi = paper.get("_doi", "")
    if doi:
        return doi.strip()
    return ""


def _coci_get_references(doi: str) -> list[str]:
    """Get list of DOIs that the given paper CITES (backward references).

    Returns list of DOI strings (not paper metadata).
    """
    if not doi:
        return []
    data = _coci_request("references", doi)
    if not data:
        return []

    dois = []
    for entry in data:
        cited_doi = entry.get("cited", "")
        if cited_doi:
            dois.append(cited_doi.strip())
    return dois


def _coci_get_citations(doi: str) -> list[str]:
    """Get list of DOIs that CITE the given paper (forward citations).

    Returns list of DOI strings (not paper metadata).
    """
    if not doi:
        return []
    data = _coci_request("citations", doi)
    if not data:
        return []

    dois = []
    for entry in data:
        citing_doi = entry.get("citing", "")
        if citing_doi:
            dois.append(citing_doi.strip())
    return dois
