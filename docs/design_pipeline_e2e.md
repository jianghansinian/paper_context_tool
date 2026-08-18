# End-to-End Field Evolution Pipeline Design

## Overview

**目标**：输入领域名称 → 自动检索代表性论文 → 产出技术演化叙事。

**当前版本**：E2E V4（2026-08-17）

**架构决策**：检索与分析解耦。检索由 V3 种子驱动引用图管线完成（完整设计见 `design_retriever_v3.md`），演化分析由 one-shot 方案 B 完成（完整设计见 `design_stage_boundary.md` §5）。本文件只定义两者的交接契约与编排。

## E2E V4 Pipeline（当前版本）

```
User Input: "BEV Perception"
  │
  ├─ Step 1: Paper Retrieval (paper_retriever_v3.py)          [检索段 LLM 2 次]
  │     retrieve_field_papers_v3(field, client, max_papers=V3_MAX_PAPERS=40)
  │     输出: 40 篇 dict {title, year, abstract, arxiv_id, citation_count,
  │                       graph_score, is_seed, source, ...}
  │
  ├─ Step 2: Cache Check + Download (paper_cache.py)          [0 LLM]
  │     dict → spec（含 abstract 透传）→ ensure_papers
  │     有 arxiv_id: 下载 PDF 提取 full text
  │     无 arxiv_id: metadata-only（abstract 降级）
  │     输出: list[Paper]（full_text + abstract）
  │
  ├─ Step 3: One-Shot Analysis (one_shot_analyzer.py)         [LLM 1 次]
  │     analyze_field_one_shot(papers, field, client)
  │     输出: {phases[], shifts[], claims[], tensions[]}
  │
  └─ Step 4: Structured Narrative (one_shot_narrative.py)     [LLM N+2 次]
        generate_evolution_md(result, field, client)
          ├─ overview（1 次）
          ├─ 每 phase 叙事（N 次）
          └─ synthesis（1 次）
        输出: 6 节结构化文档 → {slug}_evolution.md
```

**LLM 调用总计**：检索段 2 次（seed 生成 + Step 5 分类；分类按 40 篇/块分块）+ 分析段 N+3 次（one-shot 分析 1 + overview 1 + 每 phase 1 + synthesis 1）= **5 + N + 分块数** 次（N = phase 数）。

### 编排入口

`src/run_v4.py` 是唯一 E2E 编排入口：

```bash
python src/run_v4.py "BEV Perception"                 # 完整运行
python src/run_v4.py "BEV Perception" --no-download   # 跳过 PDF 下载（metadata-only）
python src/run_v4.py --resume output/v4/xxx/one_shot_result.json  # 跳过检索+下载+分析，只重生叙事
```

输出目录 `output/v4/{ts}_{slug}/`：`selected_papers.json`（检索结果+report）、`one_shot_result.json`、`{slug}_evolution.md`。

---

## Step 1: Paper Retrieval（V3）

完整设计见 `design_retriever_v3.md`（seed 生成 → venue 扩展 → 种子解析 → 引用图 → ranking → diversified selection）。此处只写交接契约。

### 交接契约：输出 dict 字段

| 字段 | 类型 | 来源 | 下游用途 |
|------|------|------|---------|
| `title` | str | OA/SS/arXiv | Paper.title、one-shot 论文标识 |
| `year` | int | OA/SS | Paper.year、时间排序 |
| `abstract` | str | OA 重建 / SS | Paper.abstract（无全文论文的关键输入） |
| `arxiv_id` | str \| "" | arXiv-hosted 论文 | Step 2 下载全文；空则 metadata-only |
| `citation_count` | int | OA（SS 富化） | 叙事生成中的硬信号 |
| `graph_score` / `citation_rate` / `seed_proximity` | float | V3 ranking | 叙事生成中的硬信号 |
| `is_seed` / `source` | bool / str | V3 selection | 叙事生成中的信任信号 |
| `_llm_category` | str | Step 5 分类 | CORE/ADJACENT 等，debug 用 |

### 设计要点

- **max_papers = V3_MAX_PAPERS（40）**：one-shot 方案 B 适用上限为 50 篇，40 在其设计范围内。40 这个数字本身无强依据，后续按下游消费需求再校准（见附录 D 遗留）。
- **abstract 是交接关键**：约半数选中论文是 conference-paper 正式版记录（无 arxiv_id），它们的 abstract（OA inverted-index 重建）是 one-shot 分析这些论文的唯一输入。V3.3.15 起 venue 补充携带完整 OA 记录，abstract 有保证。
- **无 arxiv_id 不做 arXiv 版本解析**：通过 DOI 反查 arXiv 版本可提高全文覆盖率，但增加 API 调用与复杂度，暂列为遗留（附录 D）。

---

## Step 2: Cache Check + Download

`paper_cache.py` 的 `ensure_papers()`：检查缓存 → 下载缺失 PDF → 提取 full text。

**E2E V3 修复：abstract 透传**。旧实现只读 spec 的 arxiv_id/title/year/month，丢弃 abstract——导致无全文论文在 one-shot 输入中只剩标题和年份。现在 `ensure_papers()` 读取 `spec["abstract"]` 并传入 Paper 对象。

### 降级策略

| 场景 | 处理 |
|------|------|
| 有 arxiv_id 且缓存/下载成功 | full text 提取，one-shot 输入含 intro/results/conclusion |
| 有 arxiv_id 但下载失败 | metadata-only（abstract 仍在） |
| 无 arxiv_id（conference-paper 记录） | metadata-only（abstract 仍在） |
| `--no-download` | 全部 metadata-only，已有缓存 txt 仍读取 |

---

## Step 3: One-Shot Analysis（方案 B 主路径）

完整设计见 `design_stage_boundary.md` §5-§6.6（v4.5，3-axis 框架 + 结构化叙事）。此处只写编排视角的要点：

- **输入**：每篇论文 title + year-month + abstract + introduction + results textual conclusions + conclusion（由 `text_extractor.assemble_paper_text_for_one_shot()` 组装；无全文时退化为 title + abstract）
- **输出 JSON**：`{phases[], shifts[], claims[], tensions[]}`——phase 按构建机制（Axis 1）分组、按目的（Axis 3）切分；每篇论文恰好属于一个 stage
- **适用规模**：≤ 50 篇；超过时退回方案 A（多步提取，本管线未接入）
- **验证状态**：v4.4 5-run Jaccard 1.00，4 stages 稳定（12 篇 GT 集）

---

## Step 4: Structured Narrative Generation

`generate_evolution_md(result, field_name, client)`（`src/one_shot_narrative.py`）——在 one-shot 结果之上分三步生成 6 节结构化文档（领域全景 / 范式转移 / 阶段演化 / 开放问题 / 推荐阅读 / 领域趋势与展望）：

1. **overview**（1 次 LLM）：领域全景段落
2. **per-phase narrative**（每 phase 1 次 LLM）：叙事正文（背景/核心发现/转折点/关键认知/遗留问题）+ mermaid 关系图 + key_insight/unresolved
3. **synthesis**（1 次 LLM）：综合段 + 开放问题 + 推荐阅读

最后代码拼装 markdown（phase 总览表、关键论文与核心主张表、段落结构均为代码渲染，LLM 只产出文本内容）。

设计依据：`design_stage_boundary.md` §6.6（方案 B 结构化叙事，v4.5）。

---

## LLM 调用详解（E2E V4）

| # | Step | 函数 | 输入 | 输出 | fallback |
|---|------|------|------|------|----------|
| 1 | V3 Step 1 | `_generate_seeds()` | field_name | seeds + venues | OpenAlex top-15 |
| 2 | V3 Step 5 | 分类（分块） | 候选窗口 | CORE/ADJACENT/... | 全量按 graph_score 降级 |
| 3 | Step 3 | `analyze_field_one_shot()` | 40 篇论文文本 | phases/shifts/claims/tensions | 无（失败则中止，报告错误） |
| 4 | Step 4 | overview 调用 | phases/shifts 摘要 | 领域全景段落 | 空段落 |
| 5 | Step 4 | per-phase 调用（×N） | 单 phase 的 papers/claims/tensions | narrative/mermaid/key_insight/unresolved | 空小节 |
| 6 | Step 4 | synthesis 调用 | phase 摘要 + 全部 claims | 综合段/开放问题/推荐阅读 | 空章节 |

检索段内部还有 venue 扩展、种子解析、引用图构建等 0-LLM 步骤（API 调用为主），详见 `design_retriever_v3.md`。

---

## Error Handling

| 场景 | 处理 |
|------|------|
| 无 LLM client | 检索段走 `fast` 模式（OpenAlex top by citations）；分析段无法运行，报错退出 |
| V3 检索 0 篇 | 检索段内部已有 fallback 链（seed 失败→OpenAlex top-15；图空→OpenAlex top-40）；仍为空则报错退出 |
| 部分论文无 arxiv_id | metadata-only（abstract 保留），打印 FULL/META 标记 |
| 全文 0 篇 | 继续运行（one-shot 输入退化为 title+abstract），打印质量警告 |
| one-shot 分析失败（API/解析） | 报错退出；可用 `--resume` 从已保存的 one_shot_result.json 重跑 Step 4 |
| 叙事生成失败 | 报错退出；one_shot_result.json 已保存，可 --resume 重试 |
| 分类单块失败 | V3 Step 5 内部：只丢单块，不静默（见 design_retriever_v3.md） |

---

## 附录 A：历史版本 — V2.3.1 检索管线

> 以下为 E2E V2.3.1 的 Step 1 检索管线设计（2026-07-10），E2E V3 起被 V3 种子驱动引用图检索替代（见 `design_retriever_v3.md`）。保留全文供对照。

### V2.3.1 Pipeline 图

```
User Input: "BEV Perception"
  │
  ├─ Step 1: Paper Retrieval (paper_retriever.py)
  │   │
  │   ├─ 1.1 Query Expansion (1 LLM call)
  │   │     输入: field_name
  │   │     输出: {broad_queries, specific_queries,
  │   │            synonyms_and_variants,
  │   │            disambiguation: {core_field_markers,
  │   │                             field_specific_terms,
  │   │                             exclusion_terms}}
  │   │     fallback: field_name 作为唯一 query
  │   │
  │   ├─ 1.2 Multi-source Broad Recall (0 LLM call)
  │   │     OpenAlex 语义检索（主，3 条 broad_query × 200）
  │   │     + arXiv 关键词检索（辅，specific_queries × 100）
  │   │     → merge → dedup → 零过滤全量候选池
  │   │
  │   ├─ 1.3 Citation Expansion (0 LLM call)
  │   │     Top-15 领域相关（core_field_markers 过滤）高引论文
  │   │     → OpenAlex 一阶引用扩展 (15 refs/seed)
  │   │     → merge → dedup
  │   │
  │   ├─ 1.4 Survey Calibration — V2.2 起禁用
  │   │
  │   ├─ 1.2.5 Relevance-based Pre-rank (0 LLM call)
  │   │     对全量候选池逐篇计算 relevance_score
  │   │     → 按 score 降序 → Top-150 送入 LLM
  │   │     V2.3.1: milestone-discovered papers bypass pre-rank
  │   │
  │   ├─ 1.5 Milestone Generation (1 LLM call) + Guided Search + Embedding Match
  │   │     1.5a: LLM 生成里程碑描述（此时候选池已就绪）
  │   │     1.5b: 用 description 做语义搜索 + embedding 过滤（V2.3）
  │   │     1.5c: Embedding 匹配 —— milestone ↔ candidate
  │   │     → is_seminal 标记
  │   │
  │   ├─ 1.6 LLM Unified Selection (1 LLM call)
  │   │     输入: Top-150 候选 + embedding 匹配结果 + ambiguous 列表
  │   │     输出: selected + rejected + missing_papers
  │   │     fallback: _stratify_by_citation() (保底) → _embedding_select() (替换)
  │   │
  │   ├─ 1.7 Closed-loop Recovery (0 LLM call)
  │   │     对 missing_papers + unmatched_milestones → 二次检索
  │   │
  │   └─ 1.8 Final Output
  │         selected papers + confirmed_missing
  │
  ├─ Step 2: Cache Check + Download (paper_cache.py)
  │     下载缺失 PDF，提取 full text
  │     输出: list[Paper] (with full_text)
  │
  ├─ Step 3: One-Shot Analysis (one_shot_analyzer.py)
  │     1 LLM call: papers full text → phases + shifts + claims + tensions
  │     输出: dict (JSON)
  │
  └─ Step 4: Evolution Markdown (generate_evolution_md.py)
        每 phase 1 个 LLM call + overview + synthesis
        输出: field_evolution.md
```

**LLM 调用总计**：Step 1 内 3 次 + Step 3 内 1 次 + Step 4 内 N+2 次 = **6 + N** 次

### 核心架构决策（V2.3.1）

信息流反转——先尽量多地检索论文，再让 LLM 判断重不重要。LLM 从"法官"降级为"翻译官 + 最终裁定者"。所有中间步骤（1.2-1.5）不依赖 LLM 知识。

### V2.3.1 Step 1.1: Query Expansion（LLM Call 1）

**目的**：将用户输入的领域名翻译为多样的搜索词，覆盖不同子社区和时期的术语变体。

**核心约束**：LLM 不生成里程碑清单（这是 V1 的核心错误——让 LLM 知识边界成为召回上限）。

**System Prompt**：
```
You are a research librarian who maps the terminology landscape of academic fields.
Given a field name, you generate search queries designed to maximize recall.

Your role is TRANSLATOR: you convert a field name into the diverse vocabulary that
different research communities use to describe similar concepts. You do NOT judge
which papers are important — you only ensure we search with the right words.

Return ONLY a JSON object. No other text.
```

**User Prompt**：
```
Generate search queries for: **{field_name}**

Goal: produce queries that will find papers spanning the FULL evolution of this field —
from earliest foundational work through latest SOTA.

1. BROAD QUERIES: 3-5 wide queries for semantic search. Use natural-language descriptions
   of the core problems this field addresses. Include terminology from different eras
   (older papers use different words than recent ones).

2. SPECIFIC QUERIES: 5-8 targeted queries for keyword search. Include:
   - Specific model/algorithm/technique names commonly associated with this field
   - Sub-problem formulations
   - Task descriptions using precise technical terms

3. SYNONYMS AND VARIANTS: Map the main concepts to their alternative names, abbreviations,
   and rephrasings used in different sub-communities or time periods.

4. DISAMBIGUATION: If the field name contains abbreviations that could be confused with
   other fields, provide THREE lists:
   - core_field_markers: 3-8 words/phrases UNIQUE to this field — rarely appear in other
     fields' papers. These alone should distinguish this field from lookalikes.
     Do NOT include broad terms like "perception" or "object detection".
   - field_specific_terms: broader words/phrases for ranking/scoring (can include common terms)
   - exclusion_terms: words/phrases indicating the paper is about a DIFFERENT field

Return JSON: {broad_queries, specific_queries, synonyms_and_variants, disambiguation}
```

**输出格式**：

```json
{
  "broad_queries": ["natural language description 1", "..."],
  "specific_queries": ["keyword query 1", "..."],
  "synonyms_and_variants": {"concept_name": ["variant1", "variant2"]},
  "disambiguation": {
    "core_field_markers": ["unique distinguishing term", "..."],
    "field_specific_terms": ["term that must appear in relevant papers", "..."],
    "exclusion_terms": ["term that indicates wrong field", "..."]
  }
}
```

**输出字段用途**：

| 字段 | 用途 | 使用者 |
|------|------|--------|
| `broad_queries` | OpenAlex 语义检索（前 3 条） | Step 1.2 |
| `specific_queries` | arXiv 关键词检索（全部） | Step 1.2 |
| `synonyms_and_variants` | 扩展 field_specific_terms 用于 scoring | Step 1.2.5（仅 scoring，不过滤） |
| `core_field_markers` | Pre-rank 评分 + Citation expansion seed 过滤 | Step 1.2.5, 1.3, 1.5b |
| `field_specific_terms` | 并入 synonyms 用于 pre-rank 宽泛评分 | Step 1.2.5 |
| `exclusion_terms` | Pre-rank penalty + Citation expansion 排除 | Step 1.2.5, 1.3 |

**设计要点**：

- `core_field_markers` vs `field_specific_terms`：窄义区分词（用于过滤）vs 宽泛领域词（用于 scoring）。例：BEV 领域的 core_field_markers = "bird's eye view", "view transformation", "lift splat"；field_specific_terms = "autonomous driving", "object detection", "perception"
- 如果没有 core_field_markers，fallback 用 field_specific_terms
- max_tokens=1024 自然约束 query 总数

### V2.3.1 Step 1.2: Multi-source Broad Recall（0 LLM call）

**主检索：OpenAlex 语义搜索**（免费无需 key，覆盖标题+摘要，自带引用数据）。**辅检索：arXiv 关键词搜索**（覆盖 OpenAlex 语义搜索对缩写/专有名词的盲区）。

```
1. OpenAlex 语义检索（主）: 前 3 条 broad_query × per_page=200, query 间 sleep 1s
2. arXiv 关键词检索（辅）: 所有 specific_query + field_name, max_results=100, sleep 3s
   429 → 15s/30s/45s backoff retry
3. 合并去重，零过滤（arXiv ID 去重 → 标题 fuzzy 去重，全量候选保留）
```

### V2.3.1 Step 1.2.5: Relevance-based Pre-rank（0 LLM call）

**目的**：全量候选池（1200+）无法直接喂给 LLM。按领域相关性排序后取 Top-150 送入 LLM。不删除论文——全量候选池保留，只决定 LLM 看到哪些。

**评分公式（V2.3）**：

```
title_hit   = (标题中匹配的 core_field_markers 数) / (core_field_markers 总数)
abstract_hit = (摘要中匹配的 core_field_markers 数) / (core_field_markers 总数)
exclusion_penalty = 标题或摘要含 exclusion_term → -1.0

relevance_score = (title_hit × 0.6) + (abstract_hit × 0.4) + exclusion_penalty

排序: relevance_score ↓, citation_count ↓
```

**设计要点**：

- **V2.3 改用 core_field_markers 而非 field_specific_terms**："perception"、"object detection" 等宽泛词对所有 CV/AD 论文命中率相近，区分度差。core_field_markers 是领域独有词，区分度远高
- **V2.2 去掉 citation_norm 权重**：通用 ML 论文（ResNet 222k cit）引用量碾压领域论文，引用量是噪声不是信号
- **参数名历史遗留**：`_pre_rank()` 函数签名仍用 `field_specific_terms`，但调用方传入 `core_field_markers`

**V2.3.1: Milestone-discovered papers bypass pre-rank**：milestone-guided search 发现的论文（已通过 embedding 验证，sim > 0.5），在 pre-rank 阶段被 keyword-based relevance_score 排出 top-150（标题天然不含 core_field_markers，如 DETR3D、BEVDet）。修复：mg_papers 自动 insert(0) 提升到 top-150 候选池。

**原则**：embedding 已认证的论文，不需要 keyword 再次认证。insert(0) 对 `_embedding_select()` 无影响（按 sim 排序），对 `_llm_unified_select()` 有 primacy effect。Trade-off：promoted 论文等量挤出原 top-150，但被挤出的只有 keyword 评分，promoted 有 embedding 认证——信息增益。

**V2.3.1 已知局限：池内论文盲区**：LSS 已在 broad recall 池内（arXiv 找到），但因标题不含 core_field_marker，relevance_score=0，排在 1200+ 之后。`pool_titles` 检查跳过它，promotion 不生效。根因：promotion 只覆盖 mg search **新发现**的论文，不覆盖 broad recall **已发现但被 pre-rank 排除**的论文（见附录 D 遗留 D.1）。

### V2.3.1 Step 1.3: Citation Expansion（0 LLM call）

```
从候选池中选 Top-15 篇满足以下条件的论文：
  1. citation_count > 0
  2. 标题或摘要含 ≥1 个 core_field_marker（窄义领域独有词）
  3. 标题不含 exclusion_term

对每篇 seed: 拉取 OpenAlex referenced_works (limit=15) → 合并入候选池 → 重新 dedup
```

为什么 seed 过滤用 core_field_markers：V2.1 用 field_specific_terms（含 "perception"、"object detection"）过滤 seeds，导致通用 CV 论文通过过滤。core_field_markers 只含该领域独有术语。

### V2.3.1 Step 1.4: Survey Calibration — 禁用

OpenAlex survey 搜索对歧义性领域名（如 BEV = Bird's Eye View / Battery Electric Vehicle）返回完全不相关的论文。在找到可靠的 survey 发现机制前禁用。

### V2.3.1 Step 1.5: Milestone Generation + Guided Search + Embedding Match

**为什么此时才生成里程碑**：V1 的错误是 LLM 第一步就生成里程碑 → 用这个清单去检索 → LLM 不知道的论文永远不会进入系统。V2 先检索（候选池已就绪）→ 再让 LLM 生成里程碑描述 → 在候选池中做 embedding 匹配。LLM 知识是"标记哪些已知重要论文"的信号，不是"决定哪些论文进入系统"的关卡。

**1.5a Milestone Generation（LLM Call 2）**：temperature 0.3，max_tokens 2048。已知问题：run-to-run 方差——同一 field name 不同 run 生成不同 milestone 列表 → 不同的 mg search 结果 → 不同的最终输出。未来方案：输出缓存，或改为从候选池中自动发现里程碑。

**1.5b Milestone-guided Search（V2.3 embedding 过滤）**：桥接词汇鸿沟——用 description 做语义搜索 + embedding 过滤（阈值 0.5），取代 V2.2 的 keyword 精确匹配。成本：约 N_milestones × 15 × 2 次 embedding 计算。

**1.5c Embedding Match**：sim ≥ 0.85 → is_seminal（实测 0 匹配，text-embedding-3-small 对技术论文的 sim 天然偏低，建议下调至 0.75）；sim ≥ 0.60 → 模糊区交 Step 1.6 LLM 判定；< 0.60 → unmatched。

### V2.3.1 Step 1.6: LLM Unified Selection（LLM Call 3）

LLM 三任务：classify（MILESTONE/INCREMENTAL/DERIVATIVE/IRRELEVANT）+ select（≤ max_papers）+ list missing。temperature 0.3，max_tokens 4096。输入规模：Top-150 × ~150 tokens ≈ 22.5k tokens。

**当前状态（V2.3.1 时期）**：LLM JSON 解析持续失败（连续 5 run 全部 fallback 到 embedding selection）。可能原因：三任务 + ambiguous 判定输出复杂度高；150 candidates 中提取标题精确匹配难；DeepSeek 长 JSON 输出格式稳定性。

### V2.3.1 Step 1.6.1: Embedding Fallback（确定性安全网）

两层 fallback：`_llm_unified_select()` 内部 `_stratify_by_citation()` 保底（但 V2.1 实测选的是 ResNet/Adam 等通用 ML 论文——引用量碾压）→ 外层 `_embedding_select()` 替换（milestone 描述 embedding 锚定领域核心概念，与不相关论文 sim 天然低）。

局限：依赖 milestone 描述质量（幻觉会误导）；只对 top-150 排序，不能创造候选池里没有的论文；相似 milestone 的 sim 分布趋于平坦。

### V2.3.1 Step 1.7: Closed-loop Recovery（0 LLM call）

对 missing_papers + 未匹配 milestones 二次检索：OpenAlex 语义（description）→ arXiv 作者检索（au:first_author）→ OpenAlex title（known_title_keywords），任一路径命中即停止。与 V1 区别：V2 以 OpenAlex 语义检索为主，输入是自然语言描述而非关键词。

### V2.3.1 Step 1.8: Final Output

```json
{
  "selected_papers": [
    {"title": "...", "year": 2022, "citation_count": 500,
     "classification": "milestone|incremental|embedding_ranked",
     "rationale": "...", "source": "arxiv|openalex|milestone_guided"}
  ],
  "confirmed_missing": [
    {"description": "...", "known_title_keywords": "...",
     "first_author": "...", "year": 2020,
     "recovery_attempts": ["openalex_semantic", "arxiv_author", "openalex_title"],
     "all_failed": true}
  ]
}
```

**confirmed_missing 的意义**：V1 的 recovery 失败后静默跳过。V2 必须显式列出经检索仍未能找到的论文。成本 ≈0，价值极高。

### V2.3.1 Design Decisions & Trade-offs

- **为什么用 OpenAlex 替代 SS 做主检索**：SS 语义搜索更强但无 API key 时严重限流。OpenAlex 免费、限流宽松、自带引用数据。
- **为什么 pre-rank 用 core_field_markers 而非 field_specific_terms**：宽泛词对所有 CV/AD 论文命中率相近，零区分度。
- **为什么去掉 citation_norm 权重**：通用 ML 论文引用量对数归一化后仍碾压领域论文。
- **为什么 citation expansion seeds 用 core_field_markers 过滤**：不过滤 → 从 ImageNet 扩展出大量不相关论文。
- **为什么 V2.3.1 用 insert(0) 置顶而非 append**：对 embedding select 无影响；对 LLM select 有 primacy effect。

---

## 附录 B：Version Evolution History

### E2E V3 → E2E V4（2026-08-17）

- **Step 4 叙事从散文改为结构化文档**：E2E V3 的 `generate_narrative_from_one_shot()`（1 次 LLM，600-1200 词无结构散文）被用户确认为不符合预期文档形态。E2E V4 改用结构化叙事管线：experiments/generate_evolution_md.py 转正为 `src/one_shot_narrative.py`，在 one-shot 结果上分 overview + 每 phase 叙事 + synthesis 三步（N+2 次 LLM），代码拼装 6 节文档（领域全景/范式转移/阶段演化/开放问题/推荐阅读/趋势展望）。设计依据 `design_stage_boundary.md` §6.6（v4.5）。
- **散文函数删除**：`generate_narrative_from_one_shot()` 及 prompt 从 `one_shot_analyzer.py` 移除（无引用）。
- **LLM 调用数变化**：分析段 2 次 → N+3 次（N = phase 数）。

### V2.3.1 → E2E V3（2026-08-17）

- **Step 1 检索器整体替换**：V2.3.1 keyword/embedding 混合检索（query expansion → broad recall → pre-rank → milestone → LLM select）→ V3 种子驱动引用图检索（LLM seed + venue 扩展 → 种子解析 → 引用图 → ranking → diversified selection）。动机：V2.3.1 的 LLM selection JSON 解析持续失败、池内盲区、run-to-run 方差等根因无法在补丁层面解决；V3 的种子解析 294/294、0 unresolved（run 225520）。设计见 `design_retriever_v3.md`。
- **Step 4 更换叙事生成器**：`generate_evolution_md.py`（experiments/，每 phase 1 次 LLM + overview + synthesis）→ `generate_narrative_from_one_shot()`（1 次 LLM 调用）。依据 `design_stage_boundary.md` §5 方案 B 的 pipeline 定义。
- **abstract 透传修复**：`ensure_papers()` 丢弃 abstract 的 bug 修复——V3 输出的 abstract 是无全文论文的唯一分析输入。
- **编排入口**：experiments/run_field_evolution.py（接 V2 检索）删除，新 `src/run_v4.py` 单入口。

### V2.0 → V2.1

- Pre-rank 从 pre-filter 变为 scoring（零过滤 → 全量候选池保留）
- Citation expansion 新增 seed 过滤
- Survey calibration 新增领域消歧

### V2.1 → V2.2

- Disambiguation 拆分为 core_field_markers（过滤）和 field_specific_terms（scoring）
- 新增 Milestone-guided search（keyword-based）
- Citation expansion seeds 改用 core_field_markers
- Survey calibration 禁用（BEV → 植物学论文）
- Pre-rank 去掉 citation_norm
- LLM fallback 从 citation stratification 改为 embedding select

### V2.2 → V2.3

- Milestone-guided search 加 embedding 过滤（用 description 搜索 + sim > 0.5 过滤），取代 keyword 精确匹配
- Pre-rank 改用 core_field_markers 评分

### V2.3 → V2.3.1

- Milestone-discovered papers bypass pre-rank（embedding 认证 → 不需要 keyword 再次认证）

---

## 附录 C：V1 vs V2 Architecture Comparison

| 维度 | V1 (token-rule) | V2.3.1 (historical) | E2E V4 (current) |
|------|-----------------|---------------------|------------------|
| 信息流方向 | LLM 先说有什么 → 去找 | 先尽量多找到 → LLM 判断重不重要 | LLM 种子锚定 → 引用图扩展 → 排名+分类 |
| 主检索源 | arXiv keyword | OpenAlex 语义 + arXiv keyword（辅） | OpenAlex（SS 富化，COCI 兜底） |
| 匹配逻辑 | token overlap + 多条规则 | embedding 余弦相似度 + LLM 模糊区判定 | 引用图 seed_proximity + citation_rate |
| LLM 角色 | 法官（生成里程碑清单） | 翻译官 + 最终裁定者 | 策展人（种子）+ 分类器（Step 5） |
| 词汇鸿沟 | 规则补丁 | description + embedding 搜索（语义桥接） | venue 批量扩展 + 种子解析多链 |
| LLM 调用 | 2 | 3（+1 milestone generation） | 2（+分类分块） |
| 遗漏报告 | 静默跳过 | confirmed_missing 显式列出 | unresolved_seeds 显式列出 |

---

## 附录 D：Known Issues / 遗留

### D.1 池内论文盲区（V2.3.1 遗留，E2E V3 已通过架构替换解决）

V2.3.1 的现象：已在 broad recall 池中但 relevance_score=0 的论文（如 LSS）无法进入 top-150。E2E V3 的 seed→引用图架构不再有 pre-rank 概念，此问题随检索器替换而消除。V3 自身的对应遗留见 `design_retriever_v3.md` 遗留问题列表。

### D.2 无 arxiv_id 论文的全文缺失（E2E V4 已知）

V3 选中的 conference-paper 正式版记录（OA `type=conference-paper`，无 arXiv 镜像 ID）在 Step 2 只能 metadata-only。这些论文的 one-shot 输入只有 title+abstract，分析深度低于有全文的论文。修复方向：通过 DOI 反查 arXiv 版本（OA `locations`/arXiv API），或接受降级并在叙事中不区分对待。

### D.3 40 篇与 one-shot token 预算（E2E V4 已知）

40 篇 × (abstract + intro/results/conclusion) 可能逼近 LLM 上下文上限。方案 B 适用上限 50 篇，40 在范围内但输入组装无截断机制。观察点：one-shot 输出的论文覆盖率（input vs 分配数）。若出现覆盖缺失，跟进文本裁剪或退回方案 A。

### D.4 叙事双语导出未接入

V2/V3 的 bilingual 导出策略（EN 先，ZH post-hoc 翻译）尚未应用到 Step 4 叙事输出。

---

## File Map

| File | Role |
|------|------|
| `docs/design_pipeline_e2e.md` | This document |
| `docs/design_retriever_v3.md` | Step 1 检索完整设计（V3） |
| `docs/design_stage_boundary.md` | Step 3/4 演化分析完整设计（方案 B） |
| `src/paper_retriever_v3.py` | Step 1: V3 检索入口 `retrieve_field_papers_v3()` |
| `src/paper_cache.py` | Step 2: PDF cache + download + abstract 透传 |
| `src/one_shot_analyzer.py` | Step 3: `analyze_field_one_shot()` |
| `src/one_shot_narrative.py` | Step 4: `generate_evolution_md()` 结构化叙事（overview + per-phase + synthesis + 拼装） |
| `src/text_extractor.py` | `assemble_paper_text_for_one_shot()` 输入组装 |
| `src/run_v4.py` | E2E 编排入口 |
| `src/paper_retriever.py` | V2.3.1 检索器（历史，仅供对照） |
