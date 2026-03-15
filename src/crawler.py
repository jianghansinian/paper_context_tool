import json
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests

from config import (
    ARXIV_MAX_PAPERS,
    HTTP_TIMEOUT_SEC,
    MAX_PAPERS,
    OPENALEX_MAX_PAPERS,
    PAPERS_PATH,
)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _paper_key(paper: Dict) -> str:
    return (paper.get("title") or "").strip().lower()


def _dedupe_papers(papers: List[Dict]) -> List[Dict]:
    seen = set()
    output = []
    for paper in papers:
        key = _paper_key(paper)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(paper)
    return output


def fetch_from_arxiv(keyword: str, limit: int) -> List[Dict]:
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{keyword}",
        "start": 0,
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = _normalize_space(entry.findtext("atom:title", default="", namespaces=ns))
        abstract = _normalize_space(entry.findtext("atom:summary", default="", namespaces=ns))
        published = entry.findtext("atom:published", default="", namespaces=ns)
        year = int(published[:4]) if published[:4].isdigit() else 0
        link = ""
        for link_node in entry.findall("atom:link", ns):
            href = link_node.attrib.get("href", "")
            rel = link_node.attrib.get("rel", "")
            if rel == "alternate" and href:
                link = href
                break
        if not link:
            id_text = entry.findtext("atom:id", default="", namespaces=ns)
            link = id_text or ""

        if not title:
            continue

        papers.append(
            {
                "title": title,
                "abstract": abstract,
                "year": year,
                "citation_count": 0,
                "link": link,
                "source": "arxiv",
            }
        )
    return papers


def fetch_from_openalex(keyword: str, limit: int) -> List[Dict]:
    url = "https://api.openalex.org/works"
    params = {
        "search": keyword,
        "per-page": min(limit, 200),
        "sort": "cited_by_count:desc",
    }

    response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
    response.raise_for_status()
    payload = response.json()

    papers = []
    for work in payload.get("results", []):
        title = _normalize_space(work.get("display_name", ""))
        abstract = ""
        if isinstance(work.get("abstract_inverted_index"), dict):
            token_positions = []
            for token, positions in work["abstract_inverted_index"].items():
                for pos in positions:
                    token_positions.append((pos, token))
            token_positions.sort(key=lambda item: item[0])
            abstract = _normalize_space(" ".join(token for _, token in token_positions))
        year = int(work.get("publication_year") or 0)
        citation_count = int(work.get("cited_by_count") or 0)
        link = work.get("primary_location", {}).get("landing_page_url") or work.get("id") or ""

        if not title:
            continue

        papers.append(
            {
                "title": title,
                "abstract": abstract,
                "year": year,
                "citation_count": citation_count,
                "link": link,
                "source": "openalex",
            }
        )
    return papers


def save_papers(papers: List[Dict]) -> None:
    PAPERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PAPERS_PATH.open("w", encoding="utf-8") as file:
        json.dump(papers, file, ensure_ascii=False, indent=2)


def fetch_papers(keyword: str) -> List[Dict]:
    papers = []
    errors = []

    try:
        papers.extend(fetch_from_arxiv(keyword, ARXIV_MAX_PAPERS))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"arXiv fetch failed: {exc}")

    try:
        papers.extend(fetch_from_openalex(keyword, OPENALEX_MAX_PAPERS))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"OpenAlex fetch failed: {exc}")

    papers = _dedupe_papers(papers)[:MAX_PAPERS]
    if papers:
        save_papers(papers)
        return papers

    if PAPERS_PATH.exists():
        with PAPERS_PATH.open(encoding="utf-8") as file:
            local_papers = json.load(file)
        if local_papers:
            print("Remote crawl unavailable. Falling back to local data/papers.json.")
            return local_papers

    if errors:
        raise RuntimeError(" | ".join(errors))
    raise RuntimeError("No papers collected from remote sources.")
