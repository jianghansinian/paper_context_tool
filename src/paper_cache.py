"""Paper cache management.

Cache directory: data/paper_cache/
  {arxiv_id}.pdf    — downloaded PDF
  {arxiv_id}.txt    — extracted full text
  index.json        — cache registry (maps arxiv_id → metadata)

Usage:
    cache = PaperCache()
    cache.ensure_papers(paper_list, domain="Trajectory Prediction")
    papers = cache.load_papers(domain="Trajectory Prediction")
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import PAPER_CACHE_DIR
from paper import Paper


class PaperCache:
    """Manages the paper cache directory and its index."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.dir = Path(cache_dir or PAPER_CACHE_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self._index: dict = self._load_index()

    # ── index I/O ──────────────────────────────────────────────────────

    def _load_index(self) -> dict:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    def _save_index(self):
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── register / query ───────────────────────────────────────────────

    def register(self, arxiv_id: str, title: str, year: int = 0,
                 month: int = 0, domain: Optional[str] = None):
        """Add or update a paper entry in the cache index."""
        entry = self._index.get(arxiv_id, {})
        entry["arxiv_id"] = arxiv_id
        entry["title"] = title
        if year:
            entry["year"] = year
        if month:
            entry["month"] = month

        # domains list
        domains = set(entry.get("domains", []))
        if domain:
            domains.add(domain)
        entry["domains"] = sorted(domains)

        # file state
        pdf_path = self.dir / f"{arxiv_id}.pdf"
        txt_path = self.dir / f"{arxiv_id}.txt"
        entry["has_pdf"] = pdf_path.exists()
        entry["has_text"] = txt_path.exists()
        if pdf_path.exists():
            entry["pdf_size"] = pdf_path.stat().st_size
        if txt_path.exists():
            entry["text_size"] = txt_path.stat().st_size

        entry["cached_at"] = datetime.now(timezone.utc).isoformat()
        self._index[arxiv_id] = entry
        self._save_index()

    def get(self, arxiv_id: str) -> Optional[dict]:
        """Get metadata for a cached paper."""
        return self._index.get(arxiv_id)

    def list(self, domain: Optional[str] = None) -> list[dict]:
        """List all cached papers, optionally filtered by domain."""
        entries = list(self._index.values())
        if domain:
            entries = [e for e in entries if domain in e.get("domains", [])]
        entries.sort(key=lambda e: e.get("cached_at", ""), reverse=True)
        return entries

    def check_missing(self, arxiv_ids: list[str]) -> list[str]:
        """Return arXiv IDs that are NOT in cache (no PDF, no metadata)."""
        missing = []
        for aid in arxiv_ids:
            if not aid:
                continue
            entry = self._index.get(aid)
            if not entry or not entry.get("has_pdf"):
                # Also check filesystem directly
                pdf_path = self.dir / f"{aid}.pdf"
                if not pdf_path.exists():
                    missing.append(aid)
        return missing

    def has(self, arxiv_id: str) -> bool:
        """Check if a paper is cached (PDF exists)."""
        if not arxiv_id:
            return False
        pdf_path = self.dir / f"{arxiv_id}.pdf"
        return pdf_path.exists()

    # ── download ───────────────────────────────────────────────────────

    def download_pdf(self, arxiv_id: str, timeout: int = 60) -> Optional[Path]:
        """Download a single paper PDF from arXiv."""
        from paper_resolver import _download_arxiv_pdf
        return _download_arxiv_pdf(arxiv_id, timeout=timeout)

    # ── ensure / load ──────────────────────────────────────────────────

    def ensure_papers(self, paper_specs: list[dict],
                      domain: str = "",
                      download_timeout: int = 60,
                      on_slow_download = None) -> list[Paper]:
        """Ensure all specified papers are cached. Downloads missing PDFs.

        Args:
            paper_specs: list of {arxiv_id, title, [year], [month], [abstract]}
            domain: tag all papers with this domain
            download_timeout: seconds per paper before triggering slow-download callback
            on_slow_download: callable(title, elapsed_sec, remaining_count) -> str
                return "continue" | "skip" | "quit"

        Returns:
            list of Paper objects (with full_text extracted if available)
        """
        import time as _time
        papers = []
        skip_remaining = False
        for i, spec in enumerate(paper_specs):
            arxiv_id = spec.get("arxiv_id", "")
            title = spec.get("title", "")
            year = spec.get("year", 0)
            month = spec.get("month", 0)
            abstract = spec.get("abstract", "")

            # Download PDF if missing
            if arxiv_id and not self.has(arxiv_id) and not skip_remaining:
                print(f"  [{i+1}/{len(paper_specs)}] Downloading: {title[:60]}...")
                t0 = _time.time()
                result = self.download_pdf(arxiv_id, timeout=download_timeout)
                dt = _time.time() - t0

                if result:
                    print(f"    -> {arxiv_id}.pdf saved ({dt:.0f}s)")
                elif on_slow_download and dt >= download_timeout:
                    remaining = len(paper_specs) - i - 1
                    action = on_slow_download(title, dt, remaining)
                    if action == "skip":
                        print(f"    -> Skipping remaining {remaining} downloads")
                        skip_remaining = True
                    elif action == "quit":
                        print("    -> User quit during download")
                        return papers
                    # "continue" falls through
                else:
                    print(f"    -> FAILED (will use metadata only)")

            # Extract text if PDF exists but text doesn't
            txt_path = self.dir / f"{arxiv_id}.txt" if arxiv_id else None
            full_text = ""
            if txt_path and txt_path.exists():
                full_text = txt_path.read_text(encoding="utf-8", errors="replace")
            elif arxiv_id and self.has(arxiv_id):
                try:
                    import fitz
                    pdf_path = self.dir / f"{arxiv_id}.pdf"
                    doc = fitz.open(str(pdf_path))
                    full_text = "\n\n".join(page.get_text() for page in doc)
                    doc.close()
                    txt_path.write_text(full_text, encoding="utf-8")
                    print(f"    -> Text extracted: {len(full_text):,} chars")
                except Exception as exc:
                    print(f"    -> Text extraction failed: {exc}")

            # Register in index
            if arxiv_id:
                self.register(arxiv_id, title, year, month, domain=domain)

            paper = Paper(
                id=arxiv_id or title,
                arxiv_id=arxiv_id or None,
                title=title,
                year=year or 0,
                month=month or 0,
                abstract=abstract,
                full_text=full_text,
            )
            papers.append(paper)

        return papers

    def load_papers(self, domain: Optional[str] = None,
                    arxiv_ids: Optional[list[str]] = None) -> list[Paper]:
        """Load cached papers as Paper objects.

        Args:
            domain: load all papers tagged with this domain
            arxiv_ids: load specific arXiv IDs
        """
        if arxiv_ids:
            entries = []
            for aid in arxiv_ids:
                e = self._index.get(aid)
                if e:
                    entries.append(e)
                else:
                    print(f"  WARNING: {aid} not in cache index")
        elif domain:
            entries = [e for e in self._index.values()
                       if domain in e.get("domains", [])]
        else:
            entries = list(self._index.values())

        papers = []
        for e in entries:
            aid = e.get("arxiv_id", "")
            full_text = ""
            txt_path = self.dir / f"{aid}.txt"
            if txt_path.exists():
                full_text = txt_path.read_text(encoding="utf-8", errors="replace")

            paper = Paper(
                id=aid or e.get("title", ""),
                arxiv_id=aid or None,
                title=e.get("title", ""),
                year=e.get("year", 0),
                month=e.get("month", 0),
                full_text=full_text,
            )
            papers.append(paper)

        papers.sort(key=lambda p: (p.year, p.month))
        return papers

    def print_summary(self, domain: Optional[str] = None):
        """Print a human-readable summary of cached papers."""
        entries = self.list(domain=domain)
        if not entries:
            print("No papers in cache.")
            return

        has_text = sum(1 for e in entries if e.get("has_text"))
        has_pdf = sum(1 for e in entries if e.get("has_pdf"))
        domains_all = set()
        for e in entries:
            domains_all.update(e.get("domains", []))

        print(f"Cache: {len(entries)} papers ({has_pdf} PDFs, {has_text} texts)")
        if domains_all:
            print(f"Domains: {', '.join(sorted(domains_all))}")
        print()

        for e in entries:
            aid = e.get("arxiv_id", "?")
            title = e.get("title", "?")[:65]
            y = e.get("year", 0)
            m = e.get("month", 0)
            ym = f"{y}-{m:02d}" if y else "????"
            flags = []
            if e.get("has_pdf"):
                flags.append("PDF")
            if e.get("has_text"):
                flags.append("TXT")
            domains = e.get("domains", [])
            domain_str = f" [{', '.join(domains)}]" if domains else ""
            print(f"  {aid}  {ym}  {title}{domain_str}  ({' '.join(flags)})")


# ── Singleton ──────────────────────────────────────────────────────────

_cache: Optional[PaperCache] = None


def get_paper_cache() -> PaperCache:
    global _cache
    if _cache is None:
        _cache = PaperCache()
    return _cache
