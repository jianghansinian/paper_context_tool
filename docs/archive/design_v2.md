# Paper Context Tool – Design V2 (Current)

## 1. 项目目标

Paper Context Tool 是一个 **Research Assistant**，核心能力：

1. 用户输入文字描述（可选 + 当前阅读的关键论文），系统自动抓取相关论文并分析
2. 输出该子领域的**技术发展路线图**：
   - 有哪些技术分支？
   - 每个分支有哪些关键工作？
   - 技术路线发生了哪些分叉？
   - 发生了哪些技术范式/路线演化？
3. 结果输出为丰富的 Markdown 文档（中英文双语）

---

## 2. 整体架构

### Pipeline

```
User Prompt (±seed paper)
    ↓
[1] LLM Query Refinement — 用 LLM 优化搜索关键词
    ↓
[2] 论文抓取 — arXiv + OpenAlex API
    ↓
[3] LLM Relevance Filter — 每篇论文由 LLM 判断相关性 (relevant/borderline/irrelevant)
    ↓
[4] Embedding + Clustering — 对相关论文做向量化、降维、聚类
    ↓
[5] LLM Branch Analysis — 对每个 cluster，LLM 分析：
        • 精炼的分支名
        • 关键论文 + WHY 重要
        • 演化脉络 (narrative)
        • 范式转变 (paradigm shifts)
        • 技术分叉 (technical forks)
    ↓
[6] LLM Cross-Branch Evolution — LLM 分析分支间关系：
        • 领域概述
        • 分支间关系 (precursor / fork / parallel)
        • 时间顺序
    ↓
[7] Markdown Export — 双语 (EN/ZH) 输出
    ↓
[8] LLM Validation (非阻塞) — 质量评分、问题标记、改进建议
```

### 优雅降级 (Graceful Degradation)

每个 LLM 步骤都可独立开关（通过环境变量），失败时自动降级到启发式方法：

| 步骤 | LLM 可用 | LLM 不可用 |
|---|---|---|
| Relevance Filter | LLM 判断相关性 | 保留全部论文 |
| Branch Analysis | LLM 分析 narrative + key papers | `rank_key_papers()` 启发式打分 |
| Evolution Analysis | 分支间关系 | 跳过 |
| Validation | 质量报告 | 跳过 |

**设计原则：每一步都有 fallback，不要让单点故障终止整个 pipeline。**

---

## 3. 项目结构

```
paper_context_tool/

src/
  main.py                  — Pipeline 编排
  config.py                — 环境变量配置 + timestamped output 初始化
  crawler.py               — arXiv + OpenAlex 抓取
  embedding.py             — Embedding API + 本地 HashingVectorizer fallback + 缓存
  cluster.py               — UMAP + HDBSCAN (fallback PCA + KMeans)
  branch_discovery.py      — 关键词提取 + 分支命名
  llm_namer.py             — LLM 查询精炼 + 分支命名
  llm_analyzer.py          — LLM 相关性过滤 / 分支分析 / 演化分析 / 验证
  key_paper.py             — 启发式关键论文排序 (fallback)
  citation_graph.py        — 引用关系图 + NetworkX 导出
  timeline.py              — 时间线排序
  markdown_export.py       — Markdown 导出 (中英文)

data/
  papers.json              — V1 遗留的静态论文数据
  branches.json            — V1 遗留的分支定义 (不再使用)
  embeddings.pkl           — Embedding 缓存 (SHA-256 keyed)

output/
  YYYY-MM-DD_HH-MM-SS_query-term/
    papers_raw.json        — 所有抓取的论文
    relevant_papers.json   — LLM 过滤后的相关论文子集
    clusters.json          — 每 cluster 的排序 + 分数明细
    research_graph.json    — 引用图 (node-link JSON)
    field_map.md           — 研究路线图 (英文)
    field_map.zh.md        — 研究路线图 (中文)

docs/
  design_v2.md             — 本设计文档

tests/
  test_llm_analyzer.py     — LLM 分析器测试 (30+ 测试)
  test_main_pipeline.py    — 集成测试
  conftest.py              — 共享 fixtures
  ...
```

---

## 4. LLM Analyzer 模块详情

`src/llm_analyzer.py` 是 V2 的核心新增模块，包含 4 个 LLM 驱动分析函数，遵循统一约定：接收 `OpenAI` 兼容客户端、批处理 prompt、解析 JSON 响应，失败返回 `None`。

### 4.1 `filter_relevant_papers(papers, query, client, *, model, min_score)`

- 每次最多 30 篇，分批发送给 LLM
- LLM 返回 `relevant` / `borderline` / `irrelevant` 判断
- 保留 ≥ min_score 的论文 (默认 "borderline")
- 保存中间结果到 `output/.../relevant_papers.json`

### 4.2 `analyze_branch(papers, branch_info, client, *, model)`

- 读取 cluster 中全部论文的 title + abstract
- LLM 输出结构化分析：
  - `branch_name` — 精炼技术分支名
  - `narrative` — 2-3 段演化脉络
  - `key_papers` — 3-8 篇关键论文 + WHY 重要
  - `paradigm_shifts` — 范式转变 (from→to, trigger paper)
  - `technical_forks` — 技术分叉

### 4.3 `analyze_evolution(all_branches, field, client, *, model)`

- 读取所有分支的分析结果
- LLM 输出：
  - `overview` — 1-2 段领域演化概述
  - `cross_branch_relationships` — 分支间关系 (precursor/fork/parallel)
  - `temporal_ordering` — 按历史顺序排列

### 4.4 `validate_output(field_map, client, *, model)`

- 非阻塞质量校验
- 输出：`quality_score` (1-10)、`issues[]`、`missing_topics[]`、`suggested_improvements[]`

---

## 5. 输出目录结构

每次运行创建独立的时间戳目录：

```
output/
  2026-05-17_14-30-00_bev-perception/
    papers_raw.json
    relevant_papers.json
    clusters.json
    research_graph.json
    field_map.md             ← 英文版
    field_map.zh.md          ← 中文版
  2026-05-17_15-00-00_world-model/
    ...
```

目录名格式：`YYYY-MM-DD_HH-MM-SS_keyword-slug/`

---

## 6. 配置 (环境变量)

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EMBEDDING_API_KEY` | 同 `OPENAI_API_KEY` | Embedding API key (不设则本地 fallback) |
| `EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | Embedding endpoint |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding 模型 |
| `LLM_API_KEY` | 同 `DEEPSEEK_API_KEY` | LLM 分析 API key |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | LLM chat endpoint |
| `LLM_ANALYZER_MODEL` | 同 `LLM_BRANCH_NAMING_MODEL` | 分析用模型名 |
| `LLM_ANALYZER_TIMEOUT_SEC` | `60` | 每次 LLM 调用超时 |
| `MAX_PAPERS` | `120` | 最大论文数 |
| `RELEVANCE_FILTER_ENABLED` | `1` | 开关：LLM 相关性过滤 |
| `BRANCH_ANALYSIS_ENABLED` | `1` | 开关：LLM 分支分析 |
| `EVOLUTION_ANALYSIS_ENABLED` | `1` | 开关：LLM 演化分析 |
| `OUTPUT_VALIDATION_ENABLED` | `1` | 开关：LLM 质量校验 |

---

## 7. 运行方式

```bash
# 基本运行（纯本地，无需 API key）
python src/main.py "BEV perception"

# 带 seed paper（可选）
python src/main.py "world model" "World Models"

# 仅用本地 embedding 和关键词命名（完全不调 API）
export ENABLE_LOCAL_EMBEDDING_FALLBACK=1
python src/main.py "diffusion models"
```

### 推荐 API 配置 (DeepSeek)

```bash
export LLM_API_KEY="sk-deepseek-xxx"
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_ANALYZER_MODEL="deepseek-chat"
python src/main.py "BEV perception"
```

### 输出目录

```
output/2026-05-17_14-30-00_bev-perception/
├── papers_raw.json
├── relevant_papers.json
├── clusters.json
├── research_graph.json
├── field_map.md           # 英文版
└── field_map.zh.md        # 中文版
```

---

## 8. 成本估算 (DeepSeek chat)

每次运行约 < 20K tokens，成本 < $0.001：

| 步骤 | Token | 费用 |
|---|---|---|
| Relevance Filter (60 papers) | ~18.6K | ~$0.0003 |
| Branch Analysis (5 clusters) | ~11.5K | ~$0.0002 |
| Evolution Analysis | ~3K | ~$0.00006 |
| Validation | ~2K | ~$0.00004 |
| **总计** | **~20K** | **< $0.001** |

---

## 9. 测试

```bash
python -m pytest tests/ -v
```

60+ 测试覆盖：
- JSON 解析 (纯 JSON、markdown fences、畸形、空)
- LLM 分析器 (无 client、空输入、API 错误、解析失败)
- 集成测试 (完整 pipeline + mocked LLM)
- 分支分析 fallback 路径
