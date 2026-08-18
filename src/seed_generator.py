"""V3 Step 1-2: LLM seed generation + multi-API seed resolution.

V3.3: OpenAlex API primary (with Key), SS fallback, local cache.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _resolve_model, _extract_json_object
from paper_retriever import _openalex_request, _openalex_paper_to_dict, _openalex_search


_SEED_SYSTEM = """\
You are a research historian who traces how scientific fields evolve through
distinct technical paradigms. Your task: analyze the technical development
trajectory of a research field, identify its key paradigms, and extract the
most representative papers for each paradigm.

You think like this:
  1. First, write the field's evolution STORY — the causal chain of WHY each
     new approach emerged. What problem was unsolved? What failure of the
     prior approach motivated the next?
  2. From this story, identify the distinct TECHNICAL PARADIGMS. A paradigm is
     a fundamental belief about what problem to solve and how to solve it.
  3. Paradigms can be SEQUENTIAL (one replaces another, the field's belief
     shifted) or COEXISTING (multiple philosophies compete at the same time).
     Both are valid — do not force one into the other.
  4. For each paradigm, pick the ORIGINAL paper that introduced the idea and
     the most INFLUENTIAL paper that advanced or established it.
  5. Also identify FOUNDATIONAL works (datasets, benchmarks, infrastructure)
     that the field depends on across paradigms.

CRITICAL REQUIREMENTS:
1. TEMPORAL COVERAGE: Papers must span the FULL history — from earliest
   foundational work to latest breakthroughs. Do NOT cluster in one or two years.
2. PARADIGM COVERAGE: Every major paradigm must be represented. A field with
   5 distinct paradigms should have seeds from all 5, not just the 3 most
   popular ones.
3. ORIGINALITY: For each paradigm, list the ORIGINAL paper that started it,
   not derivative works. Do NOT list incremental improvements of the same idea.
4. SPECIFICITY: Provide exact titles and author names. Do NOT fabricate papers.
5. VENUE COVERAGE: List the top 3-6 venues (conferences/journals) where this
   field's papers are most commonly published. Include full name, abbreviation,
   and the years to search (LLM determines the active period).

Return ONLY a JSON object. No other text."""

_SEED_PROMPT = """\
Trace the technical development trajectory of **{field_name}** and extract seed papers.

STEP 1 — Write the evolution story:
  First, describe the field's development as a CAUSAL NARRATIVE. Go through the
  key moments in order, but focus on WHY each new approach emerged — what gap
  or failure of the prior work motivated it. This story will ground the
  paradigm identification in the next step.

STEP 2 — From the story, identify the distinct TECHNICAL PARADIGMS:
  A paradigm is a fundamental belief about what problem to solve and how to
  solve it. Paradigms can be:
  - SEQUENTIAL: one paradigm replaced another (the field's belief shifted)
  - COEXISTING: multiple paradigms compete at the same time (different
    philosophical approaches to the same problem)
  - HYBRID: a later paradigm combines elements from two prior ones

  For each paradigm, identify:
  - What CORE IDEA defines it? What does the field believe?
  - What PERIOD does it cover? (may overlap with others)
  - What MOTIVATED its emergence? (what prior paradigm's failure?)
  - What is its RELATIONSHIP to prior paradigms? (replaced / coexists_with / hybrid_of)

  Then identify:
  - FOUNDATIONAL works the field depends on
    (Datasets, benchmarks, infrastructure, theoretical foundations)
  - Recent BREAKTHROUGHS (2024-2025) that represent emerging new directions

STEP 3 — For each paradigm and foundational area, extract the key papers:
  - The ORIGINAL paper that opened or defined the paradigm
  - The most INFLUENTIAL paper that advanced or established the paradigm
  - Recent BREAKTHROUGHS that represent genuinely new directions
  - Do NOT include variants or incremental improvements (e.g., if you
    include the paper that opened a paradigm, do NOT include a later
    incremental variant of the same paradigm)

STEP 4 — Identify the key VENUES (conferences/journals):
  - What are the top 3-6 venues where this field's papers are published?
  - Include full name, abbreviation, and the years to search.
  - LLM determines the active period based on the field's evolution.

COVERAGE CHECKLIST:
  - Does the list include papers from ALL major paradigms?
  - Does it span from the earliest foundational work to 2024-2025?
  - Are there at least 5 distinct years represented?
  - Are datasets/benchmarks included if they are field-defining?

Return JSON:
```json
{{
  "analysis": {{
    "story": "A 2-3 paragraph evolution narrative of the field, explaining the causal chain of why each paradigm emerged",
    "paradigms": [
      {{
        "name": "Paradigm Name",
        "core_idea": "Describe the core technical idea of this paradigm",
        "period": "YYYY-YYYY",
        "motivation": "What gap or failure of prior work motivated this paradigm",
        "relationship_to_prior": "replaced | coexists_with | hybrid_of",
        "key_papers": [
          {{
            "title": "exact paper title",
            "first_author": "LastName",
            "year": 2020,
            "contribution": "what this paper contributed"
          }}
        ]
      }}
    ],
    "foundational_works": [
      {{
        "title": "exact paper title",
        "first_author": "LastName",
        "year": 2013,
        "contribution": "what this work provides"
      }}
    ],
    "recent_breakthroughs": [
      {{
        "title": "exact paper title",
        "first_author": "LastName",
        "year": 2024,
        "contribution": "what this paper contributed"
      }}
    ]
  }},
  "seeds": [
    {{
      "title": "exact paper title",
      "first_author": "LastName",
      "year": 2020,
      "contribution": "what this paper introduced",
      "paradigm": "which paradigm this belongs to"
    }}
  ],
  "venues": [
    {{
      "name": "VenueAbbreviation",
      "full_name": "Full Venue Name",
      "years": [2020, 2021, 2022]
    }}
  ]
}}
```

RULES FOR SEED EXTRACTION:
- Each paradigm should contribute at least 1 seed, at most 3 seeds
- Foundational works can contribute 1-2 seeds total
- Recent breakthroughs can contribute 1-2 seeds total
- Total seeds: 15-25 papers"""


def _generate_seeds(
    field_name: str,
    client: OpenAI,
    model: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> list[dict]:
    """Step 1: LLM generates seed paper list.

    When output_dir is provided, saves a debug JSON with prompts, raw response,
    parsed analysis, and seed validation results.

    Returns list of {title, first_author, year, contribution}.
    Returns [] on failure — caller should fall back to OpenAlex top-15.
    """
    model = _resolve_model(model)
    if not model or not client:
        print("  Seed generation: no LLM available, using fallback")
        return []

    prompt = _SEED_PROMPT.format(field_name=field_name)

    debug = {
        "meta": {
            "field_name": field_name,
            "model": model,
            "base_url": str(getattr(client, "base_url", config.LLM_BASE_URL)),
            "timestamp": datetime.now().isoformat(),
        },
        "prompts": {
            "system": _SEED_SYSTEM,
            "user": prompt,
        },
        "raw_response": "",
        "parsed": None,
        "extraction": {"success": False, "error": None},
        "seeds_valid": [],
        "seeds_invalid": [],
        "statistics": {},
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SEED_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=config.LLM_ANALYZER_MAX_TOKENS,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = resp.choices[0].message.content or ""
        debug["raw_response"] = raw

        data = _extract_json_object(raw)
        if not data:
            debug["extraction"]["error"] = "Failed to parse JSON from LLM response"
            print("  Seed generation: failed to parse JSON, using fallback")
            _save_debug(output_dir, debug)
            return []

        debug["extraction"]["success"] = True
        debug["parsed"] = data

        seeds = data.get("seeds", [])
        print(f"  Seed generation: {len(seeds)} seeds from LLM")

        # Validate seeds have required fields
        valid = []
        for s in seeds:
            if s.get("title") and s.get("year"):
                valid.append(s)
            else:
                print(f"  Skipping invalid seed (missing title/year): {s.get('title', '?')[:60]}")
                debug["seeds_invalid"].append(s)

        debug["seeds_valid"] = valid

        # ★ Venue expansion: supplement seeds with top-venue papers
        venues = data.get("venues", [])
        n_venue_supplement = 0
        venue_papers_found = []
        if venues:
            print(f"  Venues from LLM: {len(venues)} — {', '.join(v.get('name', '?')[:20] for v in venues)}")
            venue_papers_found = _expand_from_venues(venues, field_name, max_per_year=25)
            if venue_papers_found:
                existing_titles = {s["title"].lower().strip() for s in valid}
                for vp in venue_papers_found:
                    t = vp["title"].lower().strip()
                    if t not in existing_titles:
                        # Carry the full OA record (_oa_id + metadata) so Step 2
                        # can reuse it directly instead of re-searching by title
                        # (V3.3.15: title re-search failed on punctuation/prefix
                        # differences, sending real papers to unresolved).
                        seed = dict(vp)
                        seed["contribution"] = f"venue supplement from {vp.get('_venue_name', '?')}"
                        valid.append(seed)
                        existing_titles.add(t)
                        n_venue_supplement += 1
            print(f"  Venue supplement: +{n_venue_supplement} seeds")

        debug["venues"] = venues
        debug["venue_papers"] = venue_papers_found
        debug["statistics"]["n_venue_supplement"] = n_venue_supplement

        if valid:
            years = sorted({s["year"] for s in valid})
            # Print analysis summary if available
            analysis = data.get("analysis", {})
            story = analysis.get("story", "")
            paradigms = analysis.get("paradigms", [])
            if paradigms:
                paradigm_names = [s.get("name", "?")[:50] for s in paradigms]
                paradigm_periods = [s.get("period", "?") for s in paradigms]
                paradigm_info = [f"{paradigm_names[i]} ({paradigm_periods[i]})" for i in range(len(paradigms))]
                print(f"  Paradigms: {len(paradigms)} — {' | '.join(paradigm_info)}")

            # Record paradigm-to-years mapping
            paradigm_papers = {}
            for s in paradigms:
                papers = s.get("key_papers", [])
                yrs = sorted({p.get("year", 0) for p in papers if p.get("year")})
                paradigm_papers[s.get("name", "?")] = {
                    "papers": papers, "years": yrs, "period": s.get("period", "?"),
                    "motivation": s.get("motivation", ""),
                    "relationship_to_prior": s.get("relationship_to_prior", ""),
                }

            debug["statistics"] = {
                "n_total": len(seeds),
                "n_valid": len(valid),
                "n_invalid": len(debug["seeds_invalid"]),
                "years": years,
                "year_span": f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0]),
                "n_distinct_years": len(years),
                "n_paradigms": len(paradigms),
                "paradigm_papers": paradigm_papers,
                "has_story": bool(story),
                "story_length": len(story) if story else 0,
                "n_foundational": len(analysis.get("foundational_works", [])),
                "n_recent_breakthroughs": len(analysis.get("recent_breakthroughs", [])),
            }
            print(f"  Year span: {years[0]}-{years[-1]} ({len(years)} distinct years)")
        else:
            debug["statistics"] = {
                "n_total": len(seeds),
                "n_valid": 0,
                "n_invalid": len(debug["seeds_invalid"]),
                "years": [],
                "year_span": "",
                "n_distinct_years": 0,
                "n_paradigms": 0,
                "paradigm_papers": {},
                "has_story": False,
                "story_length": 0,
                "n_foundational": 0,
                "n_recent_breakthroughs": 0,
            }

        _save_debug(output_dir, debug)
        return valid

    except Exception as e:
        print(f"  Seed generation failed: {e}")
        debug["extraction"]["error"] = str(e)
        _save_debug(output_dir, debug)
        return []


def _save_debug(output_dir: Optional[Path], debug: dict):
    """Save seed generation debug JSON if output_dir is set."""
    if output_dir is None:
        return
    try:
        path = output_dir / "step_1_seed_debug.json"
        path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
        paradigms = debug.get("statistics", {}).get("n_paradigms", 0)
        print(f"  -> Saved seed debug -> {path.name}  (paradigms={paradigms}, "
              f"seeds={debug.get('statistics', {}).get('n_valid', 0)})")
    except Exception as e:
        print(f"  Warning: failed to save seed debug JSON: {e}")


def _fallback_seeds(field_name: str, count: int = 18) -> list[dict]:
    """Fallback: search OpenAlex by field name, take top by citation."""
    print(f"  Using fallback: OA search for '{field_name}'")
    results = _openalex_search(field_name, limit=count * 2)
    if not results:
        return []
    results.sort(key=lambda p: -p.get("citation_count", 0))
    top = results[:count]
    print(f"  Fallback: {len(top)} seeds from OA search top-{count}")
    return top


def _openalex_title_search(title: str, limit: int = 5) -> list[dict]:
    """Search OpenAlex by title using filter=title.search (exact title match).

    Unlike _openalex_search which uses the `search=` parameter (keyword search),
    this uses `filter=title.search` which matches against the title field only.
    This is critical for papers with unique titles like "Lift, Splat, Shoot"
    that the keyword search fails to find.

    V3.3.10: dual query. OA title.search has AND token semantics with
    punctuation-sensitive matching — a full-title query can drop the
    published-version record when punctuation differs (straight vs curly
    apostrophe, e.g. BEVFormer: full title matched only the arXiv record,
    missing the 1263-citation ECCV record). A second query of the first two
    whitespace-separated words recovers those records. Results are merged
    and deduped by _oa_id.

    Returns list of paper dicts in unified format.
    """
    results: list[dict] = []
    seen_oa: set[str] = set()

    def _query(q: str) -> None:
        # Strip only commas and colons — they cause HTTP 400 in filter=title.search.
        # Apostrophes, hyphens, and other punctuation are harmless.
        q = q.replace(",", " ").replace(":", " ").strip()
        q = re.sub(r"\s+", " ", q)
        params = {
            "filter": f"title.search:{q}",
            "sort": "cited_by_count:desc",
            "per_page": str(min(limit, 50)),
        }
        data = _openalex_request("works", params)
        if not data:
            return
        for work in data.get("results", [])[:limit]:
            d = _openalex_paper_to_dict(work)
            if d and d["_oa_id"] not in seen_oa:
                seen_oa.add(d["_oa_id"])
                results.append(d)

    _query(title)
    words = [w for w in title.split() if w.strip()]
    if len(words) > 2:
        _query(" ".join(words[:2]))

    return results


def _normalize_title(title: str) -> str:
    """Normalize title for comparison: lowercase, remove punctuation.

    Literal "\\n"/"\\r"/"\\t" escape sequences (some OA records embed these
    two-char sequences instead of real whitespace) are replaced with spaces so
    they normalize identically to real whitespace.
    """
    title = title.replace(r"\n", " ").replace(r"\r", " ").replace(r"\t", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", title.lower())).strip()


# ── Venue-based seed expansion ──

_VENUE_SOURCE_CACHE: dict[str, str | None] = {}


def _resolve_venue_source_id(venue_name: str, year: int) -> str | None:
    """Resolve a venue name + year to an OpenAlex per-year source ID.

    CS conferences in OA have per-year source IDs (e.g. CVPR 2022 is
    separate from CVPR 2023). This searches OA /sources with a
    year-specific query and matches against display_name.
    """
    cache_key = f"{venue_name}|{year}"
    if cache_key in _VENUE_SOURCE_CACHE:
        return _VENUE_SOURCE_CACHE[cache_key]

    try:
        # Search OA sources with year-specific query
        # CS conferences: "2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
        # TPAMI etc: "IEEE Transactions on Pattern Analysis and Machine Intelligence" (no year prefix)
        search_queries = [
            f"{year} {venue_name}",
            venue_name,
        ]
        for sq in search_queries:
            data = _openalex_request("sources", {"search": sq, "per_page": "10"})
            if not data:
                continue
            results = data.get("results", [])
            name_lower = venue_name.lower().strip()
            year_str = str(year)

            # Exact match on display_name containing year + venue words
            for r in results:
                display = (r.get("display_name") or "").lower().strip()
                abbr = (r.get("abbreviated_title") or "").lower().strip()
                # Must contain the year and at least part of the venue name
                if year_str in display and any(w in display for w in name_lower.split()):
                    sid = r.get("id", "")
                    if sid:
                        short = sid.split("/")[-1]
                        _VENUE_SOURCE_CACHE[cache_key] = short
                        return short

                # For journals (no year in name): match display_name directly
                if year_str not in display and (display == name_lower or abbr == name_lower):
                    sid = r.get("id", "")
                    if sid:
                        short = sid.split("/")[-1]
                        _VENUE_SOURCE_CACHE[cache_key] = short
                        return short

            # Fuzzy: venue name appears in display_name
            for r in results:
                display = (r.get("display_name") or "").lower().strip()
                important_words = [w for w in name_lower.split() if len(w) > 3]
                if important_words and all(w in display for w in important_words):
                    # Check year presence if it's a conference
                    if year_str in display or all(w not in display for w in ["conference", "workshop", "symposium"]):
                        sid = r.get("id", "")
                        if sid:
                            short = sid.split("/")[-1]
                            _VENUE_SOURCE_CACHE[cache_key] = short
                            return short

        _VENUE_SOURCE_CACHE[cache_key] = None
        return None
    except Exception as e:
        print(f"  Warning: failed to resolve venue '{venue_name}' year {year}: {e}")
        _VENUE_SOURCE_CACHE[cache_key] = None
        return None


def _search_venue_papers(
    venue_name: str,
    full_name: str,
    field_keywords: str,
    years: list[int],
    max_per_year: int = 10,
) -> list[dict]:
    """Search OA for papers in a venue by field keywords.

    OA does NOT reliably link conference papers to their venue via
    primary_location.source (often null, arXiv is the primary location).
    So instead of filtering by source ID, we search a query string built
    from field keywords + venue name + year, then POST-FILTER by matching
    raw_source_name against the venue name. This is the only reliable way
    to find venue papers in OA.
    """
    all_papers: list[dict] = []
    seen_titles: set[str] = set()

    # Build venue-name match tokens from full name + abbreviation
    # e.g. "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
    #      -> (["cvpr"], ["computer vision", "pattern recognition"])
    full_acr, full_gen = _venue_match_tokens(full_name)
    abbr_acr, abbr_gen = _venue_match_tokens(venue_name)
    match_tokens = (full_acr + abbr_acr, full_gen + abbr_gen)

    for year in sorted(years, reverse=True):
        # Query string: field + venue + year. OA search matches title/abstract.
        query = f'{field_keywords} {venue_name} {year}'.strip()
        params = {
            "search": query,
            "filter": f"publication_year:{year}",
            "sort": "cited_by_count:desc",
            "per_page": "50",
        }
        try:
            data = _openalex_request("works", params)
            if not data:
                continue
            results = data.get("results", [])
            # Post-filter: keep only papers whose raw_source_name matches the venue
            matched = []
            for work in results:
                d = _openalex_paper_to_dict(work)
                if not d:
                    continue
                if _venue_matches_raw_source(d.get("_raw_source_name", ""), match_tokens):
                    matched.append(d)
            # Within this year, take top-N by citation count
            matched.sort(key=lambda p: -p.get("citation_count", 0))
            for d in matched[:max_per_year]:
                t = d["title"].lower().strip()
                if t not in seen_titles:
                    seen_titles.add(t)
                    d["venue_source_id"] = None
                    all_papers.append(d)
            if matched:
                verbose = [f"{p['title'][:40]}({p.get('citation_count',0)})" for p in matched[:3]]
                print(f"       {year}: {len(matched)}/{len(results)} matched ({', '.join(verbose)}{' ...' if len(matched) > 3 else ''})")
            else:
                print(f"       {year}: 0/{len(results)} matched venue")
        except Exception as e:
            print(f"       {year}: search failed: {e}")

        time.sleep(0.3)

    return all_papers


def _venue_match_tokens(venue_name: str) -> list[str]:
    """Break a venue name into meaningful lowercase match tokens.

    Returns (acronyms, generic_tokens). Acronyms are short all-caps
    abbreviations like "cvpr", "iccv", "eccv" — these are the strongest
    venue identifiers. Generic tokens are full-name phrases like
    "computer vision" that are too broad to discriminate reliably.
    """
    name = (venue_name or "").strip().lower()
    if not name:
        return [], []
    acronyms = []
    generic = []
    # Split on common punctuation and parens
    parts = re.split(r"[()/,]", name)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Short all-caps acronym (e.g. "cvpr", "iccv", "eccv") — strongest signal
        if 2 <= len(part) <= 6 and part.isalpha():
            acronyms.append(part)
            continue
        # Drop generic leading words, keep meaningful phrases
        stop = {"international", "conference", "ieee", "cvf", "on", "the",
                "of", "and", "acer", "proceedings", "association",
                "computing", "machinery", "annual", "symposium"}
        words = [w for w in part.split() if w not in stop and len(w) > 2]
        for n in (2, 1):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                if len(gram) > 3:
                    generic.append(gram)
    # De-dup, keep longest-first
    seen = set()
    uniq = []
    for t in acronyms + generic:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return acronyms, generic


def _venue_matches_raw_source(raw_source_name: str, tokens: tuple[list[str], list[str]]) -> bool:
    """Check if a paper's raw_source_name confirms it's from the target venue.

    - Empty or arXiv-hosted source: can't verify, keep by default. OA lists
      many venue papers only under the arXiv source; filtering them out
      loses exactly the arXiv-only papers this expansion exists to find.
    - Otherwise: the venue acronym MUST appear in raw_source_name. If the
      venue has no acronym, fall back to generic full-name tokens.
    """
    acronyms, generic = tokens
    raw = (raw_source_name or "").lower()
    if not raw or "arxiv" in raw:
        return True  # arXiv-hosted or unverifiable — keep, can't verify venue
    if acronyms:
        return any(a in raw for a in acronyms)
    return any(t in raw for t in generic)


def _expand_from_venues(
    venues: list[dict],
    field_keywords: str,
    max_per_year: int = 10,
) -> list[dict]:
    """Expand seed list by searching top venues for field papers.

    Args:
        venues: [{name, full_name, years}, ...] from LLM output.
        field_keywords: e.g. "BEV Perception" — used as OA search query.
        max_per_year: top-N papers per venue × year by citation count.

    Returns deduplicated list of paper dicts in unified format.
    """
    if not venues:
        return []

    total_found = 0
    all_papers: list[dict] = []
    seen_titles: set[str] = set()

    for v in venues:
        name = v.get("name", "")
        full_name = v.get("full_name", "")
        years = v.get("years", [])
        if not name or not years:
            continue

        year_str = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
        search_label = full_name or name
        print(f"    Venue: {search_label} ({year_str})")

        papers = _search_venue_papers(
            name, full_name, field_keywords, years, max_per_year=max_per_year,
        )

        for p in papers:
            t = p["title"].lower().strip()
            if t not in seen_titles:
                seen_titles.add(t)
                p["_venue_name"] = name
                all_papers.append(p)
                total_found += 1

        if papers:
            top_titles = [p["title"][:60] for p in papers[:3]]
            print(f"      -> {len(papers)} papers found (top: {', '.join(top_titles)}{' ...' if len(papers) > 3 else ''})")
        else:
            print(f"      -> 0 papers found")

        time.sleep(0.5)

    if total_found:
        print(f"  Venue expansion: {total_found} unique papers from {len(venues)} venues")

    return all_papers


def _is_arxiv_hosted(p: dict) -> bool:
    """True if the OA record is the arXiv preprint version of a paper."""
    src = (p.get("_raw_source_name") or "").lower()
    doi = p.get("_doi") or ""
    return "arxiv" in src or doi.startswith("10.48550")


def _resolve_seed(seed: dict, oa_results: list[dict]) -> Optional[dict]:
    """Find best OpenAlex match for a seed paper.

    Tries exact title match first, then fuzzy with year verification.

    V3.3.10: when multiple exact-title records exist (arXiv preprint +
    published version), prefer the published record — the preprint record
    undercounts citations and lacks references. The arXiv-hosted records
    are attached as _alt_oa_ids so citation expansion can crawl them too.
    """
    seed_title = _normalize_title(seed["title"])
    seed_year = seed.get("year", 0)
    seed_author = seed.get("first_author", "").lower().strip()

    # Phase 1: exact title match — collect ALL, prefer published record
    exact_matches = [r for r in oa_results if _normalize_title(r["title"]) == seed_title]
    if exact_matches:
        exact_matches.sort(
            key=lambda r: (_is_arxiv_hosted(r), -r.get("citation_count", 0))
        )
        best = exact_matches[0]
        alts = [
            r["_oa_id"]
            for r in exact_matches[1:]
            if r.get("_oa_id") and r["_oa_id"] != best.get("_oa_id")
        ]
        if alts:
            best["_alt_oa_ids"] = alts[:2]
        return best

    # Phase 2: first 50 chars match
    for r in oa_results:
        r_title = _normalize_title(r["title"])
        if seed_title[:50] and r_title[:50] and seed_title[:50] == r_title[:50]:
            return r

    # Phase 3: fuzzy with year verification
    for r in oa_results:
        r_title = _normalize_title(r["title"])
        if seed_title[:30] in r_title or r_title[:30] in seed_title:
            r_year = r.get("year", 0)
            if seed_year and r_year and abs(r_year - seed_year) <= 2:
                return r
            # Also check first author
            r_authors = [a.lower().strip() for a in (r.get("authors", []) or [])]
            if seed_author and any(seed_author in a for a in r_authors):
                return r

    # Phase 4: word-overlap match — handles cases where titles differ significantly
    # in wording but describe the same paper (e.g., "Waymo Open Dataset" vs
    # "Scalability in Perception for Autonomous Driving: Waymo Open Dataset").
    seed_words = set(seed_title.split())
    if len(seed_words) >= 3:
        for r in oa_results:
            r_title = _normalize_title(r["title"])
            r_words = set(r_title.split())
            overlap = seed_words & r_words
            # >50% of seed words appear in result title, and year matches
            if len(overlap) >= len(seed_words) * 0.5:
                r_year = r.get("year", 0)
                if seed_year and r_year and abs(r_year - seed_year) <= 2:
                    return r
                # Also check first author
                r_authors = [a.lower().strip() for a in (r.get("authors", []) or [])]
                if seed_author and any(seed_author in a for a in r_authors):
                    return r

    return None


# paper_meta cache schema version — bumped when resolution logic changes so
# wrongly-resolved entries are discarded instead of served for V3_CACHE_TTL_DAYS.
_META_CACHE_SCHEMA = 2


def _resolve_seeds(
    seeds: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Step 2: Resolve LLM-generated seeds to paper IDs (V3.3 OA-primary).

    OpenAlex is primary for title search via filter=title.search (precise).
    SS is removed from resolution pipeline due to API key constraints.
    OA keyword search is the final fallback.

    V3.3.10: resolution runs on a ThreadPoolExecutor (V3_API_WORKERS);
    API politeness is enforced by the global _oa_throttle limiter.
    """
    resolved = []
    unresolved = []

    cache_dir = Path(config.V3_CACHE_DIR) / "paper_meta"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_one(args: tuple[int, dict]) -> tuple[dict | None, dict | None]:
        i, seed = args
        title_snippet = seed["title"][:70]
        title_hash = hashlib.md5(seed["title"].lower().encode()).hexdigest()[:16]
        cache_path = cache_dir / f"{title_hash}.json"

        # 0. Check cache first
        paper = None
        api_used = "none"
        try:
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                age_days = (time.time() - cached.get("_cached_at", 0)) / 86400
                if cached.get("_schema") == _META_CACHE_SCHEMA and age_days < config.V3_CACHE_TTL_DAYS:
                    paper = cached
                    api_used = "cache"
                    print(f"  [{i+1}/{len(seeds)}] using cached (age={age_days:.0f}d)")
        except Exception:
            pass

        # 0.5 Venue record direct pass (V3.3.15): venue supplements carry the
        # full OA record from search time — reuse it instead of re-searching by
        # title (arXiv-hosted title punctuation/prefix mismatches sent real
        # papers to unresolved).
        if not paper and seed.get("_oa_id"):
            paper = dict(seed)
            paper.pop("contribution", None)
            paper.pop("_venue_name", None)
            api_used = "venue_record"

        # 1. OA primary: filter=title.search (precise title field match)
        if not paper:
            try:
                oa_results = _openalex_title_search(seed["title"], limit=5)
                if oa_results:
                    paper = _resolve_seed(seed, oa_results)
                    if paper:
                        api_used = "oa"
            except Exception as e:
                print(f"  [{i+1}/{len(seeds)}] OA title search error: {e}")

        # 2. OA keyword search fallback (broader search)
        if not paper:
            try:
                oa_results = _openalex_search(seed["title"], limit=5)
                if oa_results:
                    paper = _resolve_seed(seed, oa_results)
                    if paper:
                        api_used = "oa_keyword"
            except Exception as e:
                print(f"  [{i+1}/{len(seeds)}] OA keyword search error: {e}")

        if paper:
            paper["_seed_title"] = seed["title"]
            paper["_contribution"] = seed.get("contribution", "")

            # Write to cache (with schema version)
            try:
                paper["_schema"] = _META_CACHE_SCHEMA
                paper["_cached_at"] = time.time()
                cache_path.write_text(json.dumps(paper, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

            match_type = "exact" if _normalize_title(seed["title"]) == _normalize_title(paper["title"]) else "fuzzy"
            print(f"  [{i+1}/{len(seeds)}] ✓ ({match_type}, {api_used}) {paper['title'][:70]}"
                  f" ({paper.get('year','?')}, {paper.get('citation_count',0)} cit)")
            return paper, None

        year_str = str(seed.get("year", "?"))
        print(f"  [{i+1}/{len(seeds)}] ✗ unresolved: {title_snippet} ({year_str})")
        return None, {**seed, "reason": "no_match_in_any_source"}

    workers = max(1, config.V3_API_WORKERS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for paper, unres in ex.map(_resolve_one, enumerate(seeds)):
            if paper is not None:
                resolved.append(paper)
            if unres is not None:
                unresolved.append(unres)

    n_ok = len(resolved)
    n_total = len(seeds)
    pct = n_ok / n_total * 100 if n_total > 0 else 0
    print(f"  Seed resolution: {n_ok}/{n_total} resolved ({pct:.0f}%)")

    return resolved, unresolved
