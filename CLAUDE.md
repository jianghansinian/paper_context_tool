# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Paper Context Tool is a research assistant with two subsystems:

- **V3 — Single Paper Understanding**: Input a paper (arXiv URL/PDF), output a structured deep analysis.
- **V4 — Research Narrative Engine**: Input a field or seed paper, output a technical evolution narrative (Claim-driven, not citation-driven).

Architecture: **LLM-centric pipeline** with graceful degradation to heuristics. Full design: `docs/design.md`.

## Three Pipelines

This repo has three pipelines: **V2** (keyword-driven field mapping), **V3** (seed-paper-centric structured understanding), and **V4** (research narrative engine, in design). They coexist — V2 uses `src/main.py`, V3 uses `src/run_v3.py`.

### V2 Pipeline (keyword → research roadmap)
```
User prompt → LLM query refinement → arXiv + OpenAlex crawl
  → LLM relevance filtering → Embedding + UMAP + HDBSCAN clustering
  → LLM branch analysis → LLM evolution analysis → Bilingual markdown export → LLM validation
```

### V3 Pipeline (arXiv URL → paper deep analysis)
```
arXiv URL + description → Paper resolution (metadata + PDF + text extraction)
  → Structured Understanding (LLM full-text analysis: architecture, formulas, training/inference)
  → Backward citation mining (Semantic Scholar, recursive)
  → Forward citation mining → Key paper analysis → Technical route grouping
  → Comparative analysis → Bilingual markdown export (EN/ZH)
```

### V4 Pipeline (field → research narrative)
```
Field Name → Paper Retrieval V3 (seed-driven citation graph, 40 papers)
  → Cache + Full-text Download (paper_cache, abstract 降级)
  → One-Shot Analysis (analyze_field_one_shot: phases/shifts/claims/tensions, 1 LLM call)
  → Structured Narrative (generate_evolution_md: overview + per-phase + synthesis, N+2 LLM calls)
  → {slug}_evolution.md (6-section structured document)
```

E2E entry: `src/run_v4.py`（V3 检索 → one-shot 分析 → 叙事）。分析设计见 `docs/design_stage_boundary.md`（方案 B 主路径），检索设计见 `docs/design_retriever_v3.md`，编排设计见 `docs/design_pipeline_e2e.md`。多步管线（Scheme A：claim→relation→tension→phase→RQ→范式转移→叙事）为 >50 篇/高可控场景的扩展路径，设计见 `docs/design.md` §4.2。

See `docs/design.md` for full V3/V4 architecture, data models, and interaction design.

## Commands

```bash
# V2: Run pipeline with keyword
python src/main.py "BEV perception"
python src/main.py "world model" "World Models"

# V3: Run with arXiv URL/ID + optional description
cd src && python run_v3.py https://arxiv.org/abs/2203.17270
cd src && python run_v3.py 2203.17270 "focus on temporal attention"
cd src && python run_v3.py /path/to/paper.pdf "understand the architecture"
cd src && python run_v3.py 2203.17270 --route  # with tech route analysis

# V3: Run without LLM (test paper resolution + basic output)
cd src && V3_STRUCTURED_ANALYSIS_ENABLED=0 V3_CITATION_MINING_ENABLED=0 V3_ROUTE_ANALYSIS_ENABLED=0 python run_v3.py 2203.17270

# V4: E2E field evolution (retrieval → one-shot analysis → narrative)
python src/run_v4.py "BEV Perception"
python src/run_v4.py "BEV Perception" --no-download   # skip PDF downloads (abstract-only)
python src/run_v4.py --resume output/v4/xxx/one_shot_result.json  # re-generate narrative only

# Run all tests (from src/ directory)
cd src && python -m pytest ../tests/ -v

# Run V3 tests only
cd src && python -m pytest ../tests/test_paper.py ../tests/test_structured_analyzer.py ../tests/test_citation_miner.py ../tests/test_paper_resolver.py ../tests/test_text_extractor.py ../tests/test_route_and_export.py -v

# Run a single test file
cd src && python -m pytest ../tests/test_structured_analyzer.py -v

# Install
pip install -r requirements.txt
```

## Core Design Invariants

### Prompt Design Principles

See `docs/design_stage_boundary.md` Section 2 (universal research dimensions) and
the Prompt Design Rules appendix. Two hard rules govern all prompts in this project:

1. **No Numeric Constraints** — never put hardcoded numbers in prompts
2. **No Domain-Specific Assumptions** — prompts must generalize across all research fields

### Design Process: Design Document First, Then Code

**For any non-trivial feature or prompt design change: write/update the design document FIRST, get alignment, THEN implement.** This is not optional — it is the process.

This applies to:
- New prompt designs or prompt rewrites
- New pipeline steps or architectural changes
- Changes to output format or data models
- Any change that affects the conceptual framework (Phase/Stage/Paradigm definitions, boundary detection logic, etc.)

Why: code that drifts ahead of the design document creates a source-of-truth gap. The design document becomes stale, the code becomes the only record of decisions, and review becomes impossible because there's no reference to compare against.

The design document is the source of truth. Code is the implementation. When they disagree, the design document wins — update the code, not the other way around. If the code reveals a better approach during implementation, update the design document FIRST, then change the code.

What does NOT require a design document update:
- Bug fixes (the design was correct, the implementation was wrong)
- Refactoring that doesn't change behavior
- Adding tests
- Minor format/parsing improvements

### Every LLM analyzer function follows the same contract
- Accepts `client: Optional[OpenAI]` as the last positional parameter
- Returns structured data on success, `None` on failure
- On failure (API error / parse error / no client), the caller falls back to heuristics
- Never raises — JSON extraction heals markdown fences, truncation, and minor formatting issues

### Graceful degradation chain
| Step | LLM Available | LLM Unavailable |
|---|---|---|
| Relevance Filter | LLM judges relevance | Keep all papers |
| Branch Analysis | LLM narrative + key papers | `rank_key_papers()` heuristic scoring |
| Evolution Analysis | Cross-branch relationships | Skipped |
| Validation | Quality report | Skipped |

### Bilingual export strategy
- English markdown generated first
- Chinese translation is post-hoc via LLM (preserves paper titles and technical terms in original)
- Both exported from the same `field_map` dict with different `lang` parameter

## Key Files

### Shared
| File | Purpose |
|---|---|
| `src/config.py` | All env var configuration (V2 + V3) + `init_run_output()` |
| `src/llm_analyzer.py` | JSON extraction helpers, client builder — shared by V2 and V3 |

### V2 (keyword-driven)
| File | Purpose |
|---|---|
| `src/main.py` | V2 pipeline orchestration |
| `src/crawler.py` | arXiv + OpenAlex keyword search |
| `src/embedding.py` | Embedding API + local HashingVectorizer fallback |
| `src/cluster.py` | UMAP + HDBSCAN (fallback PCA + KMeans) |
| `src/branch_discovery.py` | Keyword extraction |
| `src/llm_namer.py` | LLM query refinement + branch naming |
| `src/key_paper.py` | Heuristic ranking fallback |
| `src/citation_graph.py` | OpenAlex citation graph |
| `src/markdown_export.py` | V2 markdown export (bilingual) |
| `src/timeline.py` | Chronological sort |

### V3 (seed-paper-centric)
| File | Purpose |
|---|---|
| `docs/design.md` | **System design (V3+V4 architecture, data models, interaction)** |
| `docs/design_v3.md` | V3 original design (archived, see design.md for current) |
| `docs/design_schema_driven.md` | V3 schema-driven architecture proposal (archived) |
| `src/run_v3.py` | V3 pipeline orchestration entry point |
| **`src/paper.py`** | **Data models: Paper, StructuredUnderstanding, Reference — also Claim, Branch for V4** |
| `src/paper_resolver.py` | arXiv URL/PDF resolution + metadata + PDF download |
| `src/text_extractor.py` | PyMuPDF-based PDF text extraction |
| **`src/structured_analyzer.py`** | **Core V3 abstraction: LLM-driven structured paper analysis** |
| `src/citation_miner.py` | Semantic Scholar API: backward/forward citation mining |
| `src/route_analyzer.py` | LLM technical route grouping + comparative analysis (legacy, replaced by V4) |
| `src/markdown_exporter_v3.py` | V3 rich markdown export (bilingual EN/ZH) |

### V4 (research narrative engine)
| File | Purpose |
|---|---|
| **`src/run_v4.py`** | **E2E entry point: V3 retrieval → one-shot analysis → structured narrative md** |
| **`src/paper_retriever_v3.py`** | **V3 retrieval entry: seed-driven citation graph (design: design_retriever_v3.md)** |
| **`src/one_shot_analyzer.py`** | **Scheme B main path: one-shot stage analysis (design: design_stage_boundary.md §5-§6.5)** |
| **`src/one_shot_narrative.py`** | **Scheme B structured narrative: overview + per-phase + synthesis → 6-section md (design: design_stage_boundary.md §6.6)** |
| `src/paper_retriever.py` | Legacy V2.3.1 two-phase retrieval (historical, for reference only) |
| `src/paper_cache.py` | PDF cache + download (used by run_v4) |
| `src/claim_extractor.py` | Claim extraction (Scheme A multi-step path, >50 papers) |
| `src/narrative_builder.py` | Scheme A narrative generation (multi-step, V8 engine) |
| `src/paradigm_shift_detector.py` | Paradigm shift detection (Scheme A) |
| `docs/design_pipeline_e2e.md` | E2E pipeline design (retrieval→analysis handoff contract) |

## Testing Patterns

Tests use `MockChatCompletion` fixtures in `tests/conftest.py` to simulate LLM responses:

```python
# Fixture pattern: mock client + canned JSON response
@pytest.fixture
def mock_llm_client():
    return MagicMock()

@pytest.fixture
def relevance_filter_response():
    return """[{"index": 0, "judgment": "relevant", ...}]"""
```

Canned responses cover: valid JSON, markdown-fenced JSON, malformed JSON, empty arrays, API errors. Test `llm_analyzer` JSON extraction helpers directly (`_extract_json_array`, `_extract_json_object`).

## Output Directory Structure

### V2 output
```
output/YYYY-MM-DD_HH-MM-SS_query-slug/
├── papers_raw.json        — All crawled papers
├── relevant_papers.json   — LLM-filtered paper subset
├── clusters.json          — Per-cluster heuristic rankings
├── research_graph.json    — Citation graph (node-link format)
├── field_map.md           — Final research map (English)
└── field_map.zh.md        — Final research map (Chinese)
```

### V3 output
```
output/v3/YYYY-MM-DD_HH-MM-SS_title-slug/
├── paper_analysis.md       — Structured analysis report (English)
├── paper_analysis.zh.md    — Structured analysis report (Chinese)
├── seed_paper.json         — Seed paper metadata
└── citation_graph.json     — Citation relationship data
```

## Key Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_API_KEY` | from `OPENAI_API_KEY` | Embedding API key |
| `EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible embedding endpoint |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `LLM_API_KEY` | from `DEEPSEEK_API_KEY` | LLM (chat) API key for naming + analysis |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | LLM chat endpoint |
| `LLM_ANALYZER_MODEL` | same as `LLM_BRANCH_NAMING_MODEL` | Model for analysis calls |
| `LLM_ANALYZER_TIMEOUT_SEC` | `60` | Timeout per LLM analysis call |
| `MAX_PAPERS` | `120` | Max total papers |
| `RELEVANCE_FILTER_ENABLED` | `1` | Toggle LLM relevance filtering |
| `BRANCH_ANALYSIS_ENABLED` | `1` | Toggle LLM branch analysis |
| `EVOLUTION_ANALYSIS_ENABLED` | `1` | Toggle LLM evolution analysis |
| `OUTPUT_VALIDATION_ENABLED` | `1` | Toggle LLM quality validation |
| `CITATION_WEIGHT` | `0.55` | Heuristic key paper weight (fallback) |
| `CENTRALITY_WEIGHT` | `0.30` | Heuristic centrality weight (fallback) |
| `RECENCY_WEIGHT` | `0.15` | Heuristic recency weight (fallback) |
| `MIN_PAPER_YEAR` | `0` | Minimum publication year for filtering |

### V3 Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `SS_API_KEY` | `""` | Semantic Scholar API key (optional, for higher rate limits) |
| `REFERENCE_MAX_DEPTH` | `2` | Max recursion depth for backward citation mining |
| `REFERENCE_TOP_K_LEVEL1` | `15` | Max references to process at level 1 |
| `REFERENCE_TOP_K_LEVEL2` | `20` | Max references to process at level 2 |
| `KEY_PAPERS_TOTAL` | `30` | Total key papers for deep analysis |
| `TEXT_CHUNK_SIZE` | `4000` | Chunk size (tokens) for long paper splitting |
| `V3_OUTPUT_DIR` | `output/v3` | V3 output directory |
| `PAPER_CACHE_DIR` | `data/paper_cache` | PDF/text cache directory |
| `V3_STRUCTURED_ANALYSIS_ENABLED` | `1` | Toggle LLM structured analysis |
| `V3_CITATION_MINING_ENABLED` | `1` | Toggle citation mining (Semantic Scholar) |
| `V3_ROUTE_ANALYSIS_ENABLED` | `0` | Toggle technical route analysis (user opt-in, default off) |
| `V3_W_CITATION` | `0.5` | Weight for citation count in key paper scoring |
| `V3_W_RECENCY` | `0.15` | Weight for recency in key paper scoring |
| `V3_W_CITATION_TYPE` | `0.15` | Weight for citation type in key paper scoring |
| `V3_W_REF_FREQ` | `0.2` | Weight for reference frequency in key paper scoring |

## Edge Cases Handled

- LLM API unavailable → every step falls back to heuristics, pipeline still completes
- Relevance filter batch fails → keeps that batch's papers unchanged
- LLM returns unparseable JSON → graceful degradation to no-LLM path
- Embedding API unavailable → HashingVectorizer local fallback (1024-dim L2 bigrams)
- No papers collected → falls back to local `papers.json`
- Clustering < 3 papers → single cluster
- HDBSCAN all noise (-1) → reassigned to cluster 0



「以第一性原理！从原始需求和问题本质出发，不从惯例或模板出发。
1. 不要假设我清楚自己想要什么。动机或目标不清晰时，停下来讨论。
2. 目标清晰但路径不是最短的，直接告诉我并建议更好的办法。
3. 遇到问题追根因，不打补丁。每个决策都要能回答"为什么"。
4. 输出说重点，砍掉一切不改变决策的信息。」


