import datetime
import os
import re
from pathlib import Path


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

PAPERS_PATH = DATA_DIR / "papers.json"
EMBEDDINGS_CACHE_PATH = DATA_DIR / "embeddings.pkl"
OUTPUT_MARKDOWN_PATH = OUTPUT_DIR / "field_map.md"
OUTPUT_GRAPH_PATH = OUTPUT_DIR / "research_graph.json"

# Base paths (overridden when run_output is set)
OUTPUT_PAPERS_PATH = OUTPUT_DIR / "papers_raw.json"
OUTPUT_CLUSTERS_PATH = OUTPUT_DIR / "clusters.json"
OUTPUT_RELEVANT_PATH = OUTPUT_DIR / "relevant_papers.json"
OUTPUT_MARKDOWN_EN_PATH = OUTPUT_DIR / "field_map.md"
OUTPUT_MARKDOWN_ZH_PATH = OUTPUT_DIR / "field_map.zh.md"

# Current run directory (set by init_run_output)
_current_run_dir: Path = OUTPUT_DIR


def _slugify(keyword: str) -> str:
    """Convert a keyword to a filesystem-safe slug."""
    s = keyword.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def init_run_output(keyword: str) -> Path:
    """Create a timestamped output directory for this run and return its path.

    Directory format: ``output/YYYY-MM-DD_HH-MM-SS_keyword-slug/``
    """
    global _current_run_dir, OUTPUT_PAPERS_PATH, OUTPUT_CLUSTERS_PATH
    global OUTPUT_RELEVANT_PATH, OUTPUT_MARKDOWN_PATH
    global OUTPUT_MARKDOWN_EN_PATH, OUTPUT_MARKDOWN_ZH_PATH
    global OUTPUT_GRAPH_PATH

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = _slugify(keyword)
    run_dir = OUTPUT_DIR / f"{timestamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _current_run_dir = run_dir
    OUTPUT_PAPERS_PATH = run_dir / "papers_raw.json"
    OUTPUT_CLUSTERS_PATH = run_dir / "clusters.json"
    OUTPUT_RELEVANT_PATH = run_dir / "relevant_papers.json"
    OUTPUT_GRAPH_PATH = run_dir / "research_graph.json"
    OUTPUT_MARKDOWN_PATH = run_dir / "field_map.md"
    OUTPUT_MARKDOWN_EN_PATH = run_dir / "field_map.md"
    OUTPUT_MARKDOWN_ZH_PATH = run_dir / "field_map.zh.md"

    return run_dir


def current_run_dir() -> Path:
    """Return the current run output directory."""
    return _current_run_dir


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", OPENAI_API_KEY)
EMBEDDING_API_KEY = os.getenv(
    "EMBEDDING_API_KEY",
    OPENAI_API_KEY or DEEPSEEK_API_KEY,
)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
ENABLE_LOCAL_EMBEDDING_FALLBACK = _env_flag("ENABLE_LOCAL_EMBEDDING_FALLBACK", "1")

CITATION_WEIGHT = float(os.getenv("CITATION_WEIGHT", "0.55"))
CENTRALITY_WEIGHT = float(os.getenv("CENTRALITY_WEIGHT", "0.30"))
TOP_K_PAPERS = 5

MAX_PAPERS = int(os.getenv("MAX_PAPERS", "120"))
ARXIV_MAX_PAPERS = int(os.getenv("ARXIV_MAX_PAPERS", "60"))
OPENALEX_MAX_PAPERS = int(os.getenv("OPENALEX_MAX_PAPERS", "60"))
LLM_API_KEY = os.getenv("LLM_API_KEY", DEEPSEEK_API_KEY)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_BRANCH_NAMING_MODEL = os.getenv("LLM_BRANCH_NAMING_MODEL", "deepseek-v4-flash")
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "30"))

RECENCY_WEIGHT = float(os.getenv("RECENCY_WEIGHT", "0.15"))
MIN_PAPER_YEAR = int(os.getenv("MIN_PAPER_YEAR", "0"))

# --- LLM Analyzer pipeline flags ---
RELEVANCE_FILTER_ENABLED = _env_flag("RELEVANCE_FILTER_ENABLED", "1")
RELEVANCE_MIN_SCORE = os.getenv("RELEVANCE_MIN_SCORE", "borderline")
BRANCH_ANALYSIS_ENABLED = _env_flag("BRANCH_ANALYSIS_ENABLED", "1")
EVOLUTION_ANALYSIS_ENABLED = _env_flag("EVOLUTION_ANALYSIS_ENABLED", "1")
OUTPUT_VALIDATION_ENABLED = _env_flag("OUTPUT_VALIDATION_ENABLED", "1")
LLM_ANALYZER_MODEL = os.getenv("LLM_ANALYZER_MODEL", LLM_BRANCH_NAMING_MODEL)
LLM_ANALYZER_TIMEOUT_SEC = float(os.getenv("LLM_ANALYZER_TIMEOUT_SEC", "60"))
LLM_ANALYZER_MAX_RETRIES = int(os.getenv("LLM_ANALYZER_MAX_RETRIES", "1"))
# Reasoning models burn token budget on hidden reasoning; 8192 leaves no room for output.
LLM_ANALYZER_MAX_TOKENS = int(os.getenv("LLM_ANALYZER_MAX_TOKENS", "32768"))

# --- V3: Seed-paper-centric structured understanding ---
SS_API_KEY = os.getenv("SS_API_KEY", "")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "ozQmZSrRp677FVcaslImKx")
REFERENCE_MAX_DEPTH = int(os.getenv("REFERENCE_MAX_DEPTH", "2"))
REFERENCE_TOP_K_LEVEL1 = int(os.getenv("REFERENCE_TOP_K_LEVEL1", "15"))
REFERENCE_TOP_K_LEVEL2 = int(os.getenv("REFERENCE_TOP_K_LEVEL2", "20"))
KEY_PAPERS_TOTAL = int(os.getenv("KEY_PAPERS_TOTAL", "30"))
TEXT_CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", "4000"))
TEXT_CHUNK_OVERLAP = int(os.getenv("TEXT_CHUNK_OVERLAP", "200"))
V3_OUTPUT_DIR = PROJECT_ROOT / os.getenv("V3_OUTPUT_DIR", "output/v3")
PAPER_CACHE_DIR = DATA_DIR / os.getenv("PAPER_CACHE_DIR", "paper_cache")
SS_REQUEST_TIMEOUT = float(os.getenv("SS_REQUEST_TIMEOUT", "15"))
SS_REQUEST_DELAY = float(os.getenv("SS_REQUEST_DELAY", "0.1"))
V3_STRUCTURED_ANALYSIS_ENABLED = _env_flag("V3_STRUCTURED_ANALYSIS_ENABLED", "1")
V3_CITATION_MINING_ENABLED = _env_flag("V3_CITATION_MINING_ENABLED", "1")
V3_ROUTE_ANALYSIS_ENABLED = _env_flag("V3_ROUTE_ANALYSIS_ENABLED", "1")
DOMAIN = os.getenv("DOMAIN", "")       # Explicit domain override ("ai_ml", "biology", "materials_science")

# Weights for key paper ranking (citation mining phase)
V3_W_CITATION = float(os.getenv("V3_W_CITATION", "0.5"))
V3_W_RECENCY = float(os.getenv("V3_W_RECENCY", "0.15"))
V3_W_CITATION_TYPE = float(os.getenv("V3_W_CITATION_TYPE", "0.15"))
V3_W_REF_FREQ = float(os.getenv("V3_W_REF_FREQ", "0.2"))

# --- V3.3: Data Source ---
# V3_PRIMARY_API and V3_FALLBACK_API removed in V3.3 — OA is the only primary source
# SS_API_KEY kept for V3 pipeline backward compatibility (citation_miner.py)
V3_CACHE_DIR = os.getenv("V3_CACHE_DIR", "data/paper_cache")
V3_CACHE_TTL_DAYS = int(os.getenv("V3_CACHE_TTL_DAYS", "30"))

# --- V3.1: Seed Generation ---
V3_SEED_COUNT_MIN = int(os.getenv("V3_SEED_COUNT_MIN", "12"))
V3_SEED_COUNT_MAX = int(os.getenv("V3_SEED_COUNT_MAX", "18"))

# --- V3.1: Citation Graph Construction ---
V3_L1_BACKWARD_LIMIT = int(os.getenv("V3_L1_BACKWARD_LIMIT", "30"))
V3_L1_FORWARD_LIMIT = int(os.getenv("V3_L1_FORWARD_LIMIT", "30"))
V3_L2_SEEDS = int(os.getenv("V3_L2_SEEDS", "15"))
V3_L2_BACKWARD_LIMIT = int(os.getenv("V3_L2_BACKWARD_LIMIT", "15"))
V3_L2_FORWARD_LIMIT = int(os.getenv("V3_L2_FORWARD_LIMIT", "15"))
V3_COUPLING_MIN_SHARED = int(os.getenv("V3_COUPLING_MIN_SHARED", "3"))
V3_COUPLING_THRESHOLD = float(os.getenv("V3_COUPLING_THRESHOLD", "0.15"))

# --- V3.3.14: Graph Ranking ---
# PageRank removed (snowball-graph coverage bias); citation signal = age-adjusted rate
V3_BETA_PROXIMITY = float(os.getenv("V3_BETA_PROXIMITY", "0.20"))
V3_GAMMA_CITATION = float(os.getenv("V3_GAMMA_CITATION", "0.80"))

# V3.3.4: Seed boost in graph ranking — compensates for OA citation undercount
V3_SEED_BOOST = float(os.getenv("V3_SEED_BOOST", "0.15"))

# --- V3.1: Selection ---
V3_MAX_PAPERS = int(os.getenv("V3_MAX_PAPERS", "40"))

# --- V3.3.12: Seed flood governance ---
# Auto seed promotion: non-seed cited by >= N seeds AND citation_count >= MIN
V3_PROMOTE_SEED_IN_EDGES = int(os.getenv("V3_PROMOTE_SEED_IN_EDGES", "2"))
V3_PROMOTE_MIN_CIT = int(os.getenv("V3_PROMOTE_MIN_CIT", "100"))
# Non-seed quota in final selection (graph-discovered papers guarantee)
V3_NONSEED_QUOTA = int(os.getenv("V3_NONSEED_QUOTA", "8"))
# Step 5 LLM classification candidate count
# V3.3.13: widened 150 -> 300 so graph-discovered field papers ranked below
# top-150 (BEVDet4D class, scores 0.26-0.28) enter the classification window
V3_CLASSIFY_TOP_K = int(os.getenv("V3_CLASSIFY_TOP_K", "300"))

# --- V3.3.13: Venue supplement admission gate ---
# Venue supplements enter Step 5 candidates only if their citation_count is
# >= this percentile of same-year papers in the graph
V3_VENUE_ADMIT_YEAR_PCTL = int(os.getenv("V3_VENUE_ADMIT_YEAR_PCTL", "50"))

# --- V3.3.10: Expansion + Performance ---
# Forward citation mixed sampling: fraction of the limit reserved for newest papers
V3_FORWARD_RECENT_FRACTION = float(os.getenv("V3_FORWARD_RECENT_FRACTION", "0.33"))
# Parallel API workers for seed resolution and citation expansion
V3_API_WORKERS = int(os.getenv("V3_API_WORKERS", "4"))


