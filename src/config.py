import os
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", OPENAI_API_KEY)
EMBEDDING_API_KEY = os.getenv(
    "EMBEDDING_API_KEY",
    OPENAI_API_KEY or DEEPSEEK_API_KEY,
)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
ENABLE_LOCAL_EMBEDDING_FALLBACK = _env_flag("ENABLE_LOCAL_EMBEDDING_FALLBACK", "1")

CITATION_WEIGHT = float(os.getenv("CITATION_WEIGHT", "0.7"))
CENTRALITY_WEIGHT = float(os.getenv("CENTRALITY_WEIGHT", "0.3"))
TOP_K_PAPERS = 5

MAX_PAPERS = int(os.getenv("MAX_PAPERS", "120"))
ARXIV_MAX_PAPERS = int(os.getenv("ARXIV_MAX_PAPERS", "60"))
OPENALEX_MAX_PAPERS = int(os.getenv("OPENALEX_MAX_PAPERS", "60"))
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "12"))
