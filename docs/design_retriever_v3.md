# Paper Retriever V3 — Seed-driven Citation Graph Architecture

> 本文档正文始终描述**当前版本**的整体设计。历史版本改动见「附录 A：版本历史」。

## 1. 动机

V2（broad recall → filter）经过五个版本的补丁，仍无法可靠提取关键论文。根本原因：从 1200+ 噪声中提信号，每层过滤都有盲区，补丁越打越多。

V3 的核心改变：**从锚点出发，通过 citation graph 扩展**。LLM 提供领域锚点（种子论文），citation graph 做真正的发现。这是 ConnectedPapers 的工作方式——从已知的论文出发，沿着引用链发现整个领域的知识图谱。

### V2 → V3 的关键认知转变

| 维度 | V2 | V3 |
|------|-----|-----|
| 核心假设 | 先广搜后过滤 | 从锚点向外扩展 |
| 发现机制 | keyword + semantic search | citation graph traversal |
| LLM 角色 | 翻译官 + 裁定者（3 calls） | 领域锚点提供者（1 call） |
| LLM 失败容忍度 | 低（selection 失败 → embedding fallback） | 高（seed 幻觉 → 跳过，图仍可用） |

---

## 2. 数据源策略

主数据源是 **OpenAlex（带 API Key）**。Semantic Scholar 降级为两个角色：① OA 数据异常时的引用兜底；② arXiv-hosted 节点的引用数富化（见 Step 3）。OpenCitations/COCI 作为 DOI 引用的备用。本地缓存作为最终兜底。

| 数据源 | 引用数据 | 额度 | 速率限制 | 角色 |
|--------|---------|------|---------|------|
| **OpenAlex (API Key)** | references + citations | 免费，无日配额 | polite pool ~10 req/s | **主数据源** |
| **Semantic Scholar (API Key)** | references + citations + 引用数 | 1 RPS（免费 key） | 1 req/s | OA 数据异常兜底 + arXiv-hosted 引用数富化 |
| **OpenCitations/COCI** | 1.5B 条 DOI citation links | 无限制，无 key | 无硬性限制 | 备用（仅覆盖有 DOI 论文） |
| **本地缓存** | 之前成功获取的数据 | 无 | 无 | 最终兜底 |

### Fallback 链

```
种子解析:
  1. OA filter=title.search 双查询（完整标题 + 短查询）→ 匹配验证
     ↓ 失败 → OA keyword search 备用
     ↓ 失败 → 缓存兜底

引用图构建:
  1. OA primary: /works/{oa_id} → referenced_works（参考文献）
     OA primary: /works?filter=cites:{oa_id}（被引论文）
     ↓ 数据异常（refs 为空或 <3；cites 为空且 OA citation_count > 50）→ COCI fallback
  2. COCI fallback: DOI-based 引用查询
     ↓ 失败 → SS fallback（仅 arXiv 论文）
  3. SS fallback: 用 ArXiv:{id} 查询 SS references/citations
     ↓ 失败 → 缓存兜底
  4. 本地缓存兜底 → miss 则跳过该节点
```

### 为什么 OA 是更好的主数据源

1. **引用图效率**：一篇论文的 API 响应直接包含 `referenced_works`（所有参考文献 OA ID），L1 展开只需每种子 1 次请求
2. **有 Key 后无配额限制**：单次运行约 100+ 次 API 调用，耗费约 $0.02
3. **filter=title.search 精确匹配**：完整标题 + 短查询双路合并（见 Step 2）
4. **前向引用**：`filter=cites:{oa_id}` 一次请求返回所有引用者

### 本地缓存

```
data/paper_cache/
├── oa_references/     # {oa_id}.json
├── oa_citations/      # {oa_id}.json
├── paper_meta/        # {title_hash}.json → 论文元数据（含 _schema 版本）
└── ss_citations/      # {normalized_title}.json → SS 富化引用数
```

TTL 统一 30 天（`V3_CACHE_TTL_DAYS`）。OA 与 SS 富化结果写缓存；SS fallback（异常兜底）结果不写缓存。`paper_meta` 条目带 `_schema` 版本，解析逻辑变更时递增使旧缓存失效，无需手工清理。

---

## 3. Pipeline 总览

```
User Input: "BEV Perception"
  │
  ├─ Step 1: Seed Generation（1 LLM call + venue API 调用）
  │     输入: field_name
  │     1a. LLM 生成: 领域演化故事 → 技术范式 → 种子论文（15-25）+ venues（3-6 个，含年份）
  │     1b. Venue 扩展: 对每个 venue × year，OA query-string 搜索 top-25 高引论文 → 补充种子
  │     输出: ~15 个 LLM 种子（高信任锚点）+ ~200+ venue 补充（批量检索补充，弱信任）
  │     fallback: OA keyword search top-18
  │
  ├─ Step 2: Seed Resolution（API 调用，0 LLM）
  │     对每个 seed: OA 双查询（完整标题 + 短查询）→ 4-phase 匹配验证 → 缓存
  │     输出: resolved_seeds（published 版本优先，附 _alt_oa_ids）
  │     幻觉处理: 解析失败的 seed 记录到 unresolved_seeds
  │
  ├─ Step 3: Citation Graph Construction（API 调用，0 LLM）
  │     API: OA 主 → COCI 备用 → SS 兜底 → 缓存
  │     3a. 直接引用扩展: L1（每个 LLM 种子 refs+cites，venue 补充不扩展）
  │         → L2（top-15 高引 L1）→ L3（2024+ 高引 L2 的 forward citations）
  │         Forward citations 混采: 67% 高引 + 33% 最新
  │     3b. Bibliographic coupling: L1 论文共享参考文献相似度 → coupling 边
  │     3c. SS 引用数富化: arXiv-hosted 节点查 SS citationCount（max 规则）
  │     同题节点合并: 归一化标题全等的节点合并为一个
  │     并行: 4 线程 + 全局限速 0.25s
  │
  ├─ Step 4: Graph Ranking（确定性，0 LLM）
  │     graph_score = 0.8·citation_rate + 0.2·seed_proximity（+ seed boost）
  │     citation_rate = 年龄校正引用强度（log1p(引用数/年龄)）；proximity 锚点只用 LLM 种子
  │     自动种子升级: 非种子被 ≥2 种子直接引用且 cit ≥100 → promoted（+boost + 入场资格）
  │
  ├─ Step 5: Diversified Selection（LLM 分类 + 确定性选择）
  │     候选入场: top-300 + LLM 种子（无条件）+ venue 补充（过同年份 P50 门槛）
  │               + promoted（无条件）
  │     LLM 分类: CORE/ADJACENT/DATASET/FOUNDATION/NOISE/REVIEW
  │               （venue 补充单独一批，带 venue 出处上下文）
  │     仅保留 CORE（ADJACENT LLM 种子受保护）→ 标题去重 → 年份分桶
  │     → 子方向去重 → 填充 → 种子保护 → 非种子配额（≥8）
  │
  └─ Step 6: Final Output
        selected_papers + graph_metadata + unresolved_seeds
```

LLM 调用：**1 次**（seed 生成）+ Step 5 分类（候选数/40 块，普通 + venue 两批）。无 LLM client 时 Step 5 退化为纯 graph_score 排序。

---

## 4. Step 1: Seed Generation（1 LLM call）

### 目的

让 LLM 分析该领域的技术演化，然后提取种子论文作为 citation graph 的锚点。

**核心约束**：LLM 不是在"决定哪些论文进入系统"，而是在"提供领域锚点"。citation graph 从锚点出发做扩展——LLM 不知道的论文，只要被锚点引用或引用锚点，就会被自动发现。

**关键认知**：种子论文不需要完美覆盖。它们是 citation graph 的**入口**，不是最终结果。但种子的**方向覆盖度**至关重要——如果一个技术路线的开创性论文不在种子中，citation graph 需要多跳才能发现它，而多跳论文的排名会因 seed_proximity 衰减而降低。

### Prompt 设计：story → paradigms → seeds → venues

不是直接让 LLM 列论文，而是先写领域的演化故事（因果链：每个新范式解决了什么未解决的问题），再从故事中识别技术范式，最后从每个范式提取种子。这比直接列论文更可靠：

1. 写演化故事是 LLM 更擅长的任务（综合判断 vs 精确回忆）
2. 范式分类天然保证覆盖度——每个范式至少 1 篇，不会遗漏整个子方向
3. 范式标签（sequential/coexisting/hybrid）供 debug 和后续分析

**System Prompt 要点**：研究历史学家视角；先写演化故事 → 识别范式（sequential/coexisting/hybrid）→ 每范式取开创论文 + 最具影响力论文 → 识别 foundational works → 列出 3-6 个主要 venues（含全名、缩写、搜索年份）。

**User Prompt 结构**（4 步）：
- STEP 1：写领域演化故事（因果叙事——每个新范式针对前一个范式的什么失败）
- STEP 2：从故事识别技术范式（core idea / period / motivation / relationship_to_prior）
- STEP 3：每范式提取 key papers（开创论文 + 最具影响力论文；禁止变体和增量改进）
- STEP 4：识别关键 venues（3-6 个，full name + abbreviation + 搜索年份，活跃期由 LLM 判断，不硬编码）

**覆盖检查清单**：全范式覆盖、时间跨度从最早 foundation 到 2024-2025、≥5 个不同年份、field-defining 数据集/基准纳入。

**输出 JSON 结构**：

```json
{
  "analysis": {
    "story": "2-3 段领域演化叙事",
    "paradigms": [
      {
        "name": "Paradigm Name",
        "core_idea": "...",
        "period": "YYYY-YYYY",
        "motivation": "前序范式的什么失败催生了它",
        "relationship_to_prior": "replaced | coexists_with | hybrid_of",
        "key_papers": [{"title", "first_author", "year", "contribution"}]
      }
    ],
    "foundational_works": [{"title", "first_author", "year", "contribution"}],
    "recent_breakthroughs": [{"title", "first_author", "year", "contribution"}]
  },
  "seeds": [
    {"title", "first_author", "year", "contribution", "paradigm"}
  ],
  "venues": [
    {"name": "VenueAbbreviation", "full_name": "Full Venue Name", "years": [2020, 2021, 2022]}
  ]
}
```

种子提取规则：每范式 1-3 篇、foundational 1-2 篇、recent breakthroughs 1-2 篇，总计 15-25 篇。参数：temperature 0.3，max_tokens 4096。

### Venue 扩展（1b）

**动机**：LLM 种子生成不稳定——同领域不同运行产生的种子列表不同，某些重要论文（如 Sparse4D）有时在种子中有时不在。一旦不在种子中，低引 arXiv-only 论文无法通过 citation graph 进入。Venue 扩展用批量检索补这个缺口：即使 LLM 没有直接生成某篇论文，只要它在该领域的主要 venue 发表过，就能被补充搜索发现。

**流程**：对 LLM 输出的每个 venue × year：

1. OA query-string 搜索：`search=field_keywords + venue_name + year`，`filter=publication_year:{year}`，`sort=cited_by_count:desc`，per_page=50
2. **后置过滤** `raw_source_name`：非空 source 必须包含 venue 缩写（如 `cvpr`），无缩写时回退 full-name 短语匹配；source 为空或 arXiv-hosted 的论文默认保留（OA 把许多会议论文只收录在 arXiv source 下，过滤掉它们恰好丢掉本扩展要找的 arXiv-only 论文）；source 是其他 venue 的记录丢弃
3. 取每年 top-25 高引论文（`max_per_year=25`——Sparse4D 在 ECCV 2022 按引用排第 22，25 才能覆盖）

**设计要点**：
- 年份范围由 LLM 决定，不硬编码（"强制搜索近 3 年"不鲁棒——领域活跃期不同）
- 按 field keywords 搜索，非全量抓取（顶会论文上千，全量不可行）
- 去重后追加到种子列表，标记 `contribution="venue supplement from {venue}"`
- **补充记录携带完整 OA 元数据**（`_oa_id` + 引用数 + authors 等），Step 2 直接复用该记录、**不再按标题重搜**（V3.3.15：此前丢弃 `_oa_id` 后按标题重搜，arXiv-hosted 论文标题标点/前缀差异导致重搜失败，误入 unresolved）

**信任分级**（贯穿全 pipeline 的核心原则）：

| 类别 | 来源 | 信任等级 | 待遇 |
|------|------|---------|------|
| LLM 种子 | LLM 策展的领域锚点 | 高 | 定义 proximity 锚点、seed boost +0.15、无条件入场、CORE/ADJACENT 保护 |
| Venue 补充 | 顶会×年份 top-25 批量检索 | 中 | 不扩展、不加 boost、不进锚点集、入场需过同年份 P50 门槛、无种子保护 |
| Promoted | 图结构证据（多种子引用 + 高引） | 中高 | boost +0.15、无条件入场；**不**进锚点集（图发现的后验不应反向定义先验） |
| 图发现 | citation graph 遍历 | 低（但靠分数说话） | 靠 graph_score 排名 + 非种子配额 |

Venue 扩展的完整输出（venues 列表 + 每 venue 找到的论文）写入 `step_1_seed_debug.json`，供人工检查 LLM 给出的顶会是否完整。

### Fallback

无 LLM client / JSON 解析失败：直接用 field_name 搜 OA，按引用量降序取 top-18 作为种子（已解析，跳过 Step 2）。

---

## 5. Step 2: Seed Resolution（0 LLM call）

将 LLM 输出的种子标题解析为 OA paper ID，同时验证种子的真实性——LLM 幻觉的论文无法解析。

### 解析流程

```
对每个 seed:
  0. 本地缓存检查: paper_meta/{title_hash}.json（_schema: 2，旧 schema 丢弃）
     ↓ hit → 使用缓存数据（跳过 API）
     ↓ miss → 进入 API fallback 链

  0.5 venue 记录直通（V3.3.15）: seed 携带 _oa_id（venue 补充的原始 OA 记录）
      → 直接作为解析结果（api_used="venue_record"，仍写缓存）
      → 不再按标题重搜

  1. OA filter=title.search 双查询:
     a. 完整标题（去逗号冒号后 URL 编码）
     b. 短查询（前几个显著词）——修复标点 token 不匹配导致的正式版记录被过滤
         （如直引号 vs 弯引号：BEVFormer 完整标题只返回 arXiv 版 38 引，
          短查询 "BEVFormer Learning" 返回 ECCV 正式版 1263 引）
     合并两路结果，按 _oa_id 去重（完整标题结果在前）
     ↓ 无结果 → OA keyword search 备用 → 失败则 unresolved

  2. 匹配验证（4-phase）:
     a. 精确标题匹配（lowercase + 去标点）——收集【所有】精确匹配记录，按优先级排序：
        published 版本优先（source 不含 arxiv 且 DOI 不以 10.48550 开头）
        → 同类内引用数高者优先。选中的记录附带 _alt_oa_ids（其余 arXiv 版记录，最多 2 个）
     b. 前 50 字符匹配
     c. 模糊匹配 + year 验证（差值 ≤ 2）
     d. 词重叠匹配（>50% seed words 出现在结果中 + year 或 author 验证）

  3. 匹配成功 → resolved（含 _alt_oa_ids），写入缓存；失败 → unresolved
```

### 设计要点

- **published 版本优先**：同一论文的正式版和 arXiv 版并存时选正式版（引用数高 1-2 个数量级、refs 完整、有 DOI 可走 COCI）
- **`_alt_oa_ids` 机制**：L1 forward 扩展时对 arXiv 版 ID 也拉引用者——修复"种子解析到正式版后，引用 arXiv 版的论文（多为新论文）丢失"
- **keyword search 备用链含 preprint + conference-paper**（V3.3.15）：OA `type` filter 不支持 OR 语法（`type:article|type:preprint` 返回 HTTP 400），故移除服务端 type filter，改为客户端白名单 `{article, conference-paper, preprint, review}` 过滤——旧 `type:article` 同时排除了 `preprint`（arXiv-only 论文）和 `conference-paper`（OA 给 CS 顶会的独立类型；StreamPETR 242 引正式版即此类型），是 StreamPETR 类种子解析失败及三篇不相关 venue 补充误入 unresolved 的根因之一；图节点转换不做此过滤（被引的 dataset 是合法图节点）
- 解析正确性直接影响下游：解析到 arXiv 版而非正式版会导致引用数低估 1-2 个数量级（历史教训，见附录 V3.3.10）

---

## 6. Step 3: Citation Graph Construction（0 LLM call）

从 resolved seeds 出发构建 citation graph。直接引用遍历发现"直系"相关论文；bibliographic coupling 发现"表亲"论文（共享参考文献但无直接引用关系）。

### 3a. 直接引用扩展

```
── Level 1: LLM 种子扩展 ──
对每个 resolved LLM seed（venue 补充不扩展，见下）:
  1. Fetch references:
     OA: GET /works/{oa_id} → referenced_works[:L1_BACKWARD_LIMIT=30]
     添加边: (seed → reference)
  2. Fetch citations:
     OA: GET /works?filter=cites:{oa_id}&per_page=200
     混采: 引用数最高的 67% + 年份最新的 33%（V3_FORWARD_RECENT_FRACTION）
     种子有 _alt_oa_ids 时对每个 alt ID 同样拉取（限额均分，按节点去重）
     添加边: (citation → seed)
  3. 节点插入统一走 _add_node()（同题合并）:
     归一化标题已存在 → 复用节点，citation_count 取 max，
     字段取非空并集，is_seed 取或

── Level 2: 高价值节点扩展 ──
从 L1 发现的论文中取 top-15 by citation_count:
  references[:15] + citations[:15]（同 L1 流程，混采 + _add_node）

── Level 3: 近期高引论文前向扩展 ──
对 L2 论文中 year ≥ 2024 且 citation_count > 50 的（最多 10 篇）:
  forward citations（limit 50）——捕获引用这些新论文的更近期论文
```

**Venue 补充不扩展**：venue 补充种子（~200+）不做 L1 引用扩展。它们保留为图节点（参与 ranking 和耦合），但扩展只为 LLM 种子做。理由：126 个 venue 补充全部扩展曾使图膨胀到 5695 节点、top-40 一半是噪声；且低引种子扩展出的引用多为无关领域论文。效果：图 5695 → 875 节点，API 调用减少 90%+。

**同题节点合并（_add_node）**：同一论文可通过两条路径进入图并得到不同 node_id（种子解析命中的 OA 条目 vs citation expansion 找到的正式版条目）。`_add_node()` 按归一化标题合并：citation_count 取 max，`_oa_id/_doi/arxiv_id` 保留非空者，`is_seed` 取或。这使得 Step 5 的 title 去重变成安全网而非主防线。

**节点 ID 方向约定**：`add_edge(citing, cited)`；coupling 边带 `edge_type="bibliographic_coupling"` 属性（citation 边无此属性）——该属性是 promotion 边判定的依据（见 Step 4）。

### 3b. Bibliographic Coupling

直接引用遍历的盲区：两篇论文讨论同一问题、引用相同的经典论文，但没有互相引用。Coupling 通过共享参考文献发现这种"表亲"关系。

```
算法:
  1. 对每个 L1 节点收集其 reference list（Step 3a 已获取，0 增量 API 调用）
  2. 对每对 L1 论文 (A, B):
     shared = |refs_A ∩ refs_B|
     if shared >= COUPLING_MIN_SHARED (3):
       coupling_score = shared / sqrt(|refs_A| * |refs_B|)   # Ochiai coefficient
       if coupling_score >= COUPLING_THRESHOLD (0.15):
         添加双向边 (A,B) 和 (B,A)，edge_type="bibliographic_coupling"
  3. 只对 L1 论文计算（~720 节点 → ~260k pairs，set intersection 可接受）
```

**设计要点**：coupling 是补充机制不是替代——直接引用遍历是图骨干，coupling 边是弱信号连接（默认权重 0，见 Step 4）。不用 LLM，纯数学计算。

### 3c. SS 引用数富化（arXiv-hosted 节点）

**问题**：OA 对 arXiv-only 论文的 `citation_count` 系统性低估——OA 的 arXiv 版记录不合并正式版引用。实测富化前 57% 的论文 SS ≥ 2x OA，中位 2.5x，极端 19.5x（PointPillars 241→4690）；有正式版记录的论文比值健康（1.1-1.6x）。

**方案**（缩小版 SS 富化）：
- **范围**：只覆盖 arXiv-hosted 图节点（`_raw_source_name` 含 "arxiv" 或 `_doi` 以 `10.48550` 开头）——覆盖全部低估案例，避免全图 ~200+ 请求
- **规则**：SS 标题精确匹配（归一化全等）→ `citation_count = max(OA, SS)`；无精确匹配 → 跳过（**宁缺勿错**，错误引用数比缺失更糟）
- **缓存**：`ss_citations/{normalized_title}.json`，TTL 30 天
- **限流**：复用 ss_client 1 RPS；无 key 时匿名池 429 → 静默跳过，管线不受影响
- **时机**：图构建尾部（L3 + coupling 之后、ranking 之前）

### 并行化

- `ThreadPoolExecutor(V3_API_WORKERS=4)` 并行三个串行热点：种子解析循环、L1 每种子 refs+cites 抓取、L2 循环
- `_openalex_request` 入口全局线程安全限速器（最小间隔 0.25s），聚合 ≤4 req/s（polite pool 上限 10 req/s）
- 缓存读写加 `threading.Lock`

### 预期规模

- ~15 LLM 种子 × (30 backward + 30 forward) + alt ID 扩展 ≈ 1000 L1 节点
- 15 L2 × (15 + 15) ≈ 400 L2 节点，去重后图 ~800-1200 节点
- Coupling 边 50-200 条（仅高相似度 pairs）

---

## 7. Step 4: Graph Ranking（确定性，0 LLM call）

### 排名公式

```
graph_score = γ × citation_rate + β × seed_proximity（+ seed boost）

默认权重: γ=0.8, β=0.2
```

| 信号 | 含义 | 计算 |
|------|------|------|
| citation_rate | 年龄校正引用强度 | log1p(引用数 / 论文年龄) 归一化到 [0,1]（除以池内最大值，SS 富化后引用数）。同一年份桶内等价于按引用数排序 |
| seed_proximity | 与 LLM 种子锚点的图距离 | BFS 从 LLM 种子出发：种子 1.0，1 跳 0.5，2 跳 0.33，3 跳 0.25，无路径 0 |

### 职责划分（V3.3.14 重构）

graph_score 的职责只有一条：**同领域内的重要性排序**。相关性判断由 Step 5 LLM 分类（CORE/ADJACENT/NOISE 门）负责（V3.3.8 已移交），score 不再承担相关性职责。由此三项决策：

- **删除 PageRank**：雪球图（种子 3 跳邻域）上的 PageRank 测的是"图内被引覆盖度"——由扩张顺序决定，不是重要性。实测 992 节点图 67% 论文 pr_norm=0（图内无 in-edge），corr(pagerank, 引用数)=0.24，近二值信号却在 α=0.45 下占据近半权重。
- **γ 换计量方式**：全局 log 引用密度 → 年龄校正引用强度（引用数/论文年龄）。与 venue 入场门槛的"同年份比较"是同一原则：重要性 = 同年份里的相对引用强度。年龄校正保证跨年份公平（2024+ 新论文只跟自己的年龄比），同年份桶内则天然等价于按引用数排序。
- **β 缩为 0.2**：实测 β=0.30 时 prox 0→0.5 的差距 = 0.15 分，大于 536 引论文的全部 citation 分（0.127 分）——M2BEV（246 引，图发现）0.2681 > MapTR（430 引，venue 补充）0.1230 即此机制。β=0.2 后差距缩到 0.1 分，proximity 保留为弱结构信号。

### 锚点收敛（V3.3.12）

seed_proximity 的锚点集 = **LLM 种子 only**：

```python
llm_seed_ids = {sid for sid in seed_ids
                if not (node_data[sid].get("_contribution") or "").startswith("venue supplement")}
```

理由：proximity 编码"离核心锚点多近"，venue 补充是批量检索补充、信任等级低于 LLM 种子，不应定义引力场。此前全部 ~260 个种子（含 venue 补充）都是锚点，venue 补充同样享受 proximity=1.0——这是"final 40 全是种子"洪泛的根因之一。

### 自动种子升级（Auto Promotion）

非种子节点被 **≥2 个种子直接引用**（边 s→n，s ∈ seed_ids，只统计 citation 边——coupling 边不是引用关系）且 citation_count **≥100**（SS 富化后）→ 升级为 **promoted**：

- 获得 seed boost +0.15（多种子引用 = 领域内关键论文的结构证据）+ Step 5 候选入场资格（不依赖排名）
- **不**进入锚点集（llm_seed_ids）——锚点是先验，promoted 是图发现的后验，后验不应反向定义先验

配置：`V3_PROMOTE_SEED_IN_EDGES=2`、`V3_PROMOTE_MIN_CIT=100`。实测 33-44 篇达标（含 DETR3D 1149 引类）——非种子首次获得确定性入场机制。

### Boost 规则汇总

| 论文类别 | boost | 锚点地位 |
|---------|-------|---------|
| LLM 种子 | +0.15 | ✓（proximity BFS 起点） |
| Promoted | +0.15 | ✗ |
| Venue 补充 | +0 | ✗ |
| 其他 | +0 | ✗ |

### 噪声过滤

**无关键词相关性机制**——手工停用词表（`_FIELD_STOP_WORDS` 类）是领域特定的，每新增一个领域都要维护新词表，不可泛化。graph_score 不乘任何 relevance 系数，噪声论文完全由 Step 5 的 LLM 分类（NOISE 类别）过滤。无 LLM 时噪声论文不降权（接受的降级）。

---

## 8. Step 5: Diversified Selection（LLM 分类 + 确定性选择）

### 流程

```
输入: ranked_papers（graph_score desc）, seed_ids, max_papers=40 (V3_MAX_PAPERS)

0. 候选入场（三类来源，按信任分级）:
   a. graph_score top-300（V3_CLASSIFY_TOP_K）
   b. LLM 种子: 无条件入场（策展先验）
   c. Venue 补充: 入场门槛 = citation_count ≥ 同年份 venue 池 P50
      （V3_VENUE_ADMIT_YEAR_PCTL；同年份 venue 池 < 3 篇 → 直接放行，数据不足不比较）
   d. Promoted: 无条件入场（结构证据）

   为什么 venue 补充要过门槛: 它们由"顶会 × 年份 top-25 高引"批量检索进入，
   曾在引用数低估时代享受无条件入场。SS 富化后引用数是诚实信号，无条件入场
   理由已过时——弱 venue 论文（如 19-32 引）曾占 final 近一半，而 500+ 引的
   图发现论文连分类资格都没有。门槛人口用 venue 池自身（顶会论文同年 cohort），
   不用全图 cohort——图人口被种子引用邻域拉高（精英偏置），比较必然不公平。

1. LLM 分类:
   - 普通候选一批 + venue 补充单独一批（带 venue 出处上下文）
   - 40 篇/块分块调用（单块失败只丢单块，不静默）
   - 每篇归类: CORE / ADJACENT / DATASET / FOUNDATION / NOISE / REVIEW
   - prompt entry 含 citation_count 与 graph_score 之外的硬信号
   - 结果落盘 step_5_classification.json（title/category/citation_count/graph_score/
     year/is_seed/promoted/venue_supplement）

2. 过滤 — 只保留 CORE:
   - CORE → core_pool
   - ADJACENT 且是 LLM 种子 → core_pool + protected
     （种子列表是比单次随机分类更强的策展信号；ADJACENT 是分类器低估；
      DATASET/FOUNDATION/NOISE/REVIEW 否决仍生效；venue 补充是弱信号不享受保护）
   - 分类失败或 CORE 为空 → core_pool = 全部 ranked_papers（降级）
   - core_pool 按 (graph_score desc, citation_count desc) 排序，归一化标题去重
     （同一论文可能经种子解析与引用扩展两条 OA 路径入图）

3. 年份分桶: 每桶基础配额 max_papers/n_years（至少 1），剩余按桶大小比例分配

4. 每桶内按 graph_score 选，跨桶 title 去重

5. 子方向去重: 同一 first_author 最多保留 2 篇（超出按 graph_score 去掉最低的）

6. 填充: 从 core_pool 剩余论文按分数填充到 max_papers

7. 最终 title 去重（归一化标题折叠所有空白字符）+ 排序

8. 种子保护: protected（CORE/ADJACENT 的 LLM 种子）未入选时，
   替换 selected 中最低分的非 protected 论文

9. 非种子配额: final 中 is_seed=False 的论文 ≥ V3_NONSEED_QUOTA（默认 8）。
   不足时用 core_pool 中最高分非种子替换最低分未保护种子，
   供给耗尽则打印 "supply exhausted"（透明度）
```

### Venue 批次分类的出处上下文

Venue 补充单独一批分类，user prompt 前置一段 venue 出处上下文（`_CLASSIFY_VENUE_CONTEXT`）：

- **做什么**：告诉分类器"这些论文由领域顶会 + 领域关键词搜索发现，出现于领域自己的 venue 是领域论文的证据"——pipeline 溯源信息，可泛化，不是领域假设
- **为什么**：reasoning 模型在 temperature=0 下分类仍有 CORE/ADJACENT 抖动（Sparse4D 实测 3 次翻转）；加上 venue 上下文后 3/3 稳定判 CORE
- **边界**：venue 溯源只帮助区分 CORE vs ADJACENT（领域内 vs 领域外），**不豁免方法要求**——主要贡献是数据集/基准/评价的论文仍判 DATASET，综述仍判 REVIEW，与是否来自领域 venue 无关
- 基础分类标准单一化：venue provenance 是批次级信息放在 user prompt 前置段，不塞进 system prompt 的分类规则

### 分类类别定义

| 类别 | 含义 | 待遇 |
|------|------|------|
| CORE | 领域方法论文 | 入选 core_pool |
| ADJACENT | 相关但非本领域方法（如 2D 检测之于 BEV） | LLM 种子受保护；其余排除 |
| DATASET | 数据集/基准论文 | 排除（种子保护也不覆盖） |
| FOUNDATION | 基础架构（ResNet/ViT/COCO 类） | 排除 |
| NOISE | 无关噪声 | 排除 |
| REVIEW | 综述/评价/教程——总结或评价领域但不贡献新方法 | 排除 |

---

## 9. Step 6: Final Output

```
selected_papers: list[paper_data]（清理内部键 _oa_id/_alt_oa_ids/_contribution 等）
report: {
  total_graph_nodes, total_graph_edges, coupling_pairs,
  resolved_seeds, unresolved_seeds, total_selected
}
```

---

## 10. 已知遗留问题

| # | 问题 | 原因 | 当前状态 |
|---|------|------|----------|
| 1 | OA 引用数与 SS 差距 | OA 不合并"引用 arXiv 版"的引用 | **已修（V3.3.11）**：arXiv-hosted 节点 SS 富化（max 规则 + 缓存） |
| 2 | arXiv-only GT 论文入场不稳定（BEVDet4D 536 引、PETRv2 335 引、BEVFormer v2 299 引） | 图内排名在分类窗口外 + 种子入边 ≤1 不触发 promotion；**入图本身依赖随机 LLM 种子与扩张路径** | V3.3.13 窗口 150→300 修复了"入图后进不了分类视野"（run 162930 BEVDet4D #27 ✓）；V3.3.14 重权修复了 venue 补充被锚点收敛惩罚（run 215612：PETRv2 #27、BEVFormer v2 ✓，Sparse4D 按同年份引用标准输给 BEVerse 283，设计行为）。**未根治**：run 215612 BEVDet4D 未入图——LLM 未生成该种子且 venue 搜索 query "BEV Perception CVPR 2022" 匹配不到标题 token "BEVDet4D"，窗口/权重修复只在论文入图后生效。方向：venue 搜索关键词多样性 |
| 3 | SS fallback 缺陷 | ① 结果不写缓存（V3.3.11 的 `ss_citations` 缓存修复了富化的一半）；② 非 arXiv 论文构造无效 `ArXiv:oa:W...` 查询；③ 429 重试 sleep 完整 Retry-After（实测 34s/请求） | 功能可用但低效；修复方向待定 |
| 4 | `OPENALEX_API_KEY` 硬编码默认值 | config.py 中共享 key，所有用户共用同一配额标识 | 应改为必填 env（无默认值） |
| 5 | `V3_DELTA_COUPLING=0.0` | 防弱信号膨胀的有意关闭 | 设计行为，文档与代码一致 |
| 6 | L3 扩展参数硬编码 | `year >= 2024`、`cited_by_count > 50`、max 10 篇 | 年份阈值随时间失效，应配置化 |
| 7 | 双编排入口 | `experiments/test_v3_retrieval.py` 与 `src/paper_retriever_v3.py` 各自维护一份 pipeline 编排 | 两处需同步修改，易漂移 |
| 8 | 种子标题 hallucination | LLM 生成标题与实际论文标题有差异（如 "SOLOFusion: ...BEV Perception" vs "...3D Object Detection"） | 影响 seed resolution 成功率，需改进种子 prompt。**V3.3.15 缓解并验证**（run 225520）：OA 标题规范化丢弃前缀（"StreamPETR: ..." → "Exploring Object-Centric ..."）与标题含 "?" 导致 title.search 失败的两类机制已修（venue 直通 + type 白名单含 preprint/conference-paper），StreamPETR 解析到 242 引正式版 |
| 9 | final 全部是种子、图排名零贡献 | 锚点含全部种子 + 候选入场无条件 | **已修（V3.3.12）**：锚点收敛 + 配额。验证 run 092507：top-60 种子占比 54/60→11/60，非种子 4/40 |
| 10 | GT 两篇 2026-05 arXiv 新论文（MindVLA-U1、ParkingWorld）未入图 | 无入场路径：venue 年份 LLM 只列到 2024；零引用进不了 forward 截断；只能靠 LLM 种子运气 | 结构性盲区。GT 仅人工对比用，不作为优化目标 |

---

## 11. V2 失败案例在 V3 中的路径

**LSS**（标题不含 "BEV"）：
```
Step 1: LLM seed 包含 LSS（foundational）
  或: BEVDet/DETR3D 的 references 包含 LSS → L1 扩展进入图
Step 4: 被 3+ 种子直接引用 → 多入边 hub，seed_proximity=1.0
→ 自然排在 Top-10
```

**Sparse4D**（sparse 方向，LLM 种子不稳定）：
```
Step 1b: venue 扩展（ECCV 2022 top-25）→ 补充种子
Step 5: venue 上下文分类稳定判 CORE + 过同年份 P50 门槛 → 入选
```

**2024 论文**：
```
Step 3a: forward citations 混采（33% 最新年份）自然包含 2024 论文
Step 5: 年份分桶确保 2024 论文有配额
→ 种子不需要包含 2024 论文
```

**关键原则**：种子是入口，不是全集。Citation graph 的 L1→L2 扩展 + bibliographic coupling 本身就是发现机制。不接受任何需要人工维护的领域知识库。

---

## 12. 与 V2 的代码复用

| V2 模块 | V3 复用 | 说明 |
|---------|---------|------|
| `_openalex_search()` | 主搜索 | OA 是 V3 主数据源 |
| `_openalex_get_references()` | 主引用 | 同上 |
| `_openalex_paper_to_dict()` | 复用 | 论文数据转换 |
| `_deduplicate()` | 复用 | 图节点去重 |
| `_extract_json_object()` | 复用 | LLM JSON 解析 |
| `config.py` | 复用 | 配置项 |

**V3 新增模块**：`src/open_citations_client.py`（COCI API）、`src/ss_client.py`（SS API）、`data/paper_cache/`（本地缓存）。

**不复用的 V2 模块**：query expansion、broad recall、keyword pre-rank、milestone guided search、LLM selection、embedding fallback——V3 从锚点出发，不需要广搜-过滤链。

---

## 13. 配置项

```python
# --- Data Source ---
V3_CACHE_DIR = os.getenv("V3_CACHE_DIR", "data/paper_cache")
V3_CACHE_TTL_DAYS = int(os.getenv("V3_CACHE_TTL_DAYS", "30"))

# --- Seed Generation ---
V3_SEED_COUNT_MIN = int(os.getenv("V3_SEED_COUNT_MIN", "12"))
V3_SEED_COUNT_MAX = int(os.getenv("V3_SEED_COUNT_MAX", "18"))

# --- Citation Graph Construction ---
V3_L1_BACKWARD_LIMIT = int(os.getenv("V3_L1_BACKWARD_LIMIT", "30"))
V3_L1_FORWARD_LIMIT = int(os.getenv("V3_L1_FORWARD_LIMIT", "30"))
V3_L2_SEEDS = int(os.getenv("V3_L2_SEEDS", "15"))
V3_L2_BACKWARD_LIMIT = int(os.getenv("V3_L2_BACKWARD_LIMIT", "15"))
V3_L2_FORWARD_LIMIT = int(os.getenv("V3_L2_FORWARD_LIMIT", "15"))
V3_COUPLING_MIN_SHARED = int(os.getenv("V3_COUPLING_MIN_SHARED", "3"))
V3_COUPLING_THRESHOLD = float(os.getenv("V3_COUPLING_THRESHOLD", "0.15"))

# --- Graph Ranking ---
V3_BETA_PROXIMITY = float(os.getenv("V3_BETA_PROXIMITY", "0.20"))
V3_GAMMA_CITATION = float(os.getenv("V3_GAMMA_CITATION", "0.80"))
V3_SEED_BOOST = float(os.getenv("V3_SEED_BOOST", "0.15"))           # LLM 种子与 promoted 加分

# --- Selection ---
V3_MAX_PAPERS = int(os.getenv("V3_MAX_PAPERS", "40"))
V3_CLASSIFY_TOP_K = int(os.getenv("V3_CLASSIFY_TOP_K", "300"))      # Step 5 分类候选数
V3_NONSEED_QUOTA = int(os.getenv("V3_NONSEED_QUOTA", "8"))          # final 非种子保底数
V3_PROMOTE_SEED_IN_EDGES = int(os.getenv("V3_PROMOTE_SEED_IN_EDGES", "2"))  # 自动升级种子入边数
V3_PROMOTE_MIN_CIT = int(os.getenv("V3_PROMOTE_MIN_CIT", "100"))    # 自动升级引用数门槛
V3_VENUE_ADMIT_YEAR_PCTL = int(os.getenv("V3_VENUE_ADMIT_YEAR_PCTL", "50"))  # venue 补充同年份入场百分位

# --- Expansion + Performance ---
V3_FORWARD_RECENT_FRACTION = float(os.getenv("V3_FORWARD_RECENT_FRACTION", "0.33"))  # forward 混采新论文占比
V3_API_WORKERS = int(os.getenv("V3_API_WORKERS", "4"))                # API 并行线程数

# --- 共享 LLM 配置 ---
LLM_ANALYZER_MAX_TOKENS = int(os.getenv("LLM_ANALYZER_MAX_TOKENS", "32768"))
```

**未配置化（硬编码在代码中，注意漂移）**：
- L3 扩展条件：`year >= 2024` 且 `cited_by_count > 50`，最多扩展 10 篇——年份阈值会随时间失效
- first_author 去重上限 2
- `OPENALEX_API_KEY` 硬编码默认值（见遗留问题 #4）

---

## 14. Error Handling

| 场景 | 处理 |
|------|------|
| LLM seed generation 失败 / JSON 解析失败 | fallback: field_name 搜 OA top-18 作为种子 |
| 种子论文无法解析（幻觉/极冷门） | OA 双查询 → keyword search → 缓存，全失败记录到 unresolved_seeds |
| OA 429 速率限制 | 10s/20s/30s backoff retry（最多 3 次），全失败 → COCI fallback |
| OA 403 budget exhaustion | **立即 fallback，不重试** → COCI |
| COCI 失败（无 DOI / 404） | → SS fallback（仅 arXiv 论文）→ 缓存兜底 |
| 所有 API + 缓存都失败 | 跳过该 seed/node，记录到 unresolved，不中断流程 |
| Citation graph 为空 / 无种子被解析 | fallback: field_name 直接搜 OA top-40 by citations |
| LLM 分类空响应/解析失败（单块） | 只丢单块；全部失败 → core_pool = 全部 ranked_papers |
| 缓存过期（> 30 天） | 删除条目，本次同步重新请求（无异步刷新） |
| OA 引用数据异常（refs 空/<3，或 cites 空且引用数高） | `_refs_anomalous` / `_cites_anomalous` 检测触发 SS fallback |
| SS 富化 429 / 无 key | 静默跳过，管线不受影响 |
| 非种子配额供给耗尽 | 打印 "supply exhausted"，不强行凑数 |

---

## 15. 依赖

无新增 Python 依赖：`networkx`（图构建 + BFS proximity）、`openai`（LLM）、`urllib`（HTTP，标准库）。

---

## 16. 测试计划

### 单次运行验证

```bash
python experiments/test_v3_retrieval.py
```

检查点：
1. Seed generation: 12-18 个 LLM 种子 + 200+ venue 补充种子，跨越 ≥5 年
2. Seed resolution: ≥85% 解析成功（OA 双查询主）
3. Citation graph: ≥500 节点，≥1000 边
4. LSS 是否在 Top-40（V2 的关键失败案例）
5. Sparse4D 是否在 Top-40（venue 扩展 + venue 上下文分类）
6. 是否有 2024+ 论文（forward 混采 + 年份分桶）
7. 最终 Top-40 无综述/数据集/噪声论文（LLM 分类过滤）
8. LLM 种子中分类 CORE/ADJACENT 的论文全部在 Top-40（种子保护）
9. 无重复论文（title 去重）
10. 非种子 ≥ 配额（或打印 supply exhausted）
11. venue 通过/淘汰数打印（Step 5 入场门槛）
12. `step_1_seed_debug.json` 中 venues 完整（人工检查顶会是否覆盖）
13. `step_5_classification.json` 分类结果完整

### MVP Ground Truth 验证

以下三个领域的代表性论文作为测试 ground truth，用于验证 seed generation 和最终选择（40 篇）的覆盖率。**GT 仅用于人工对比，不作为优化目标**。

**BEV Perception（12 篇）**：LSS 2008.05711 (2020)、BEVDet 2112.11790 (2021)、BEVDepth 2206.10092 (2022)、BEVDet4D 2203.17054 (2022)、BEVFormer 2203.17270 (2022)、BEVFormerV2 2211.10439 (2022)、SparseBEV 2308.09244 (2023)、Sparse4D 2211.10581 (2022)、Sparse4Dv2 2305.14018 (2023)、UniAD 2212.10156 (2022)、VAD 2303.12077 (2023)、SparseDrive 2405.19620 (2024)。
验证目标：最终选择中至少 8/12，且每个技术路线至少 1 篇。

**Trajectory Prediction（13 篇）**：Social LSTM (2016)、Social GAN (2018)、DESIRE (2017)、LaneGCN (2020)、Trajectron++ (2020)、MultiPath (2019)、CoverNet (2020)、AgentFormer (2021)、Scene Transformer (2021)、TNT (2020)、MotionDiffuser (2023)、GameFormer (2023)、MultiPath++ (2022)。
验证目标：至少 10/13。

**End-to-End Driving（17 篇）**：ALVINN (1989)、NVIDIA E2E (2016)、Learning to Drive in a Day (2018)、Conditional Imitation Learning (2017)、Learning by Cheating (2020)、NEAT (2022)、ST-P3 (2022)、TCP (2022)、Think Twice (2023)、UniAD (2022)、VAD (2023)、DriveAdapter (2023)、DiffusionDrive (2024)、Hydra-MDP (2024)、DriveGPT4 (2023)、LMDrive (2024)、Reason2Drive (2023)。
验证目标：至少 12/17。

检查命令：
```bash
python experiments/test_v3_retrieval.py 2>&1 | grep -E "(LSS|BEVDet|BEVDepth|BEVFormer|SparseBEV|Sparse4D|UniAD|VAD|SparseDrive)"
```

### 5-run 稳定性测试

```bash
for i in $(seq 1 5); do python experiments/test_v3_retrieval.py; done
```

检查点：Top-40 的 Jaccard 相似度；LSS、Sparse4D、2024 论文是否每次都在；coupling 边数量跨 run 方差。

### API 故障注入测试

```bash
export OPENALEX_API_KEY="invalid" && python experiments/test_v3_retrieval.py   # OA budget exhaustion
export OPENALEX_DISABLED=1 && python experiments/test_v3_retrieval.py         # 全部不可用
```

---

## 附录 A：版本历史

> 每条一行摘要 + 关键验证数据。设计细节已折叠进正文对应章节。

| 版本 | 日期 | 改动摘要 | 关键验证数据 |
|------|------|---------|-------------|
| V3.1 | — | 多 API fallback（SS 主 + OA 备用）+ 本地缓存；bibliographic coupling 引入 | — |
| V3.2 | — | Seed prompt 重写：先分析路线/范式再提取种子；MVP ground truth 建立 | — |
| V3.3 | — | 主数据源 SS → OA（带 key，免费无日配额） | — |
| V3.3.1 | 2026-07-13 | 修复 OA 引用数据缺陷链：`_doi` 提取解锁 COCI fallback、DOI 精确解析（`filter=doi:`）、低引种子自适应 forward limit、L3 扩展（2024+ 高引 L2 的前向）、per-seed 引用数据 dump | SparseDrive/Scene Transformer 入图 ✓；Sparse4D 仍缺（OA 无数据，非代码可修 → 催生 V3.3.3 venue 扩展） |
| V3.3.3 | 2026-07 | Venue 扩展：LLM 输出 venues（3-6 个 + 年份范围），OA query-string 搜索 + `raw_source_name` 后置过滤（OA 的 `primary_location.source` 对会议论文常为 null，按 source ID 过滤不可靠） | Sparse4D 经 ECCV 2022 venue 补充稳定入场 |
| V3.3.4 | 2026-07 | 防分数膨胀：field_relevance 自动 1.0 只给 LLM 种子；`_dedup_by_title` 标题去重 | venue 补充低引噪声不再虚高 |
| V3.3.5 | 2026-08 | Venue 补充不做 L1 扩展（图 5695→875 节点，API -90%）；领域关键词从 LLM 种子标题提取（删硬编码 BEV 词表）；max_per_year 10→25 | Sparse4D（ECCV 2022 引用排名第 22）不再被截断 |
| V3.3.6 | 2026-08 | 分类可靠性：`max_tokens` 8192→32768（reasoning 模型隐藏推理耗尽 token → 空响应）+ 40 篇/块分块 + 无静默失败；恢复种子保护（CORE/ADJACENT LLM 种子强制入选）；venue 补充单独批次带出处上下文（稳定 temperature=0 下的 CORE/ADJACENT 抖动） | run 191208：274/274 分类成功；Sparse4D 加 venue 上下文后 3/3 稳定判 CORE（之前 2 次翻转） |
| V3.3.7 | 2026-08 | 输出清洗：标题归一化折叠空白字符（LSS 双条目漏检修复）；新增 REVIEW 类别（综述不再落到 CORE）；venue 上下文 prompt 重写（溯源不豁免方法要求，数据集论文仍判 DATASET） | run 221810：40 篇无重复、无综述、无数据集论文，Sparse4D 稳定入选；GT 6/12（BEVDet 等 arXiv-only 入场不稳定 → 遗留问题 #2） |
| V3.3.8 | 2026-08 | **删除 field_relevance 机制**（~220 词手工停用词表不可泛化）；LLM 分类成为唯一相关性门控，graph_score 不再乘 relevance 系数 | 决策理由：CORE 池内二次关键词排序边际收益 ≈0；噪声由 NOISE 类别过滤（V3.3.6/7 已验证） |
| V3.3.9 | 2026-08 | 诊断：OA 对 arXiv-only 论文引用数系统性低估（中位 3.7x，最差 53x）；提出 SS 富化方案 A/B/C | 后续证实极端值一半是解析错误而非数据短板（→ V3.3.10 修复） |
| V3.3.10 | 2026-08 | 解析正确性 + 合并 + 性能：Seed 解析双查询 + published 版本优先 + `_alt_oa_ids`；图级同题节点合并（`_add_node`）；forward 混采（33% 最新年份）；4 线程并行 + 全局限速 0.25s；缓存 `_schema: 2` 失效机制 | BEVFormer 解析 38 引（arXiv 版）→ 1263 引（ECCV 正式版）✓ |
| V3.3.11 | 2026-08 | SS 引用数富化（只覆盖 arXiv-hosted 节点，max 规则 + `ss_citations` 缓存，宁缺勿错）；venue boost 0.15→0（富化修复了引用低估根因，不再需要补偿）；`_raw_source_name` 解析 bug 修复（读错 JSON 字段导致 venue 后置过滤永不生效，修复后 arXiv-hosted source 默认保留） | run 173336：Sparse4D 30→209 引、#15 入选 ✓；BEVFormer 2019、MapTR 430（OA 69）✓；final 40 仍全种子（→ V3.3.12） |
| V3.3.12 | 2026-08 | 种子洪泛治理（4 项）：① 锚点收敛——proximity 与 PageRank personalization 只用 LLM 种子；② 自动种子升级——非种子被 ≥2 种子引用（citation 边 only）且 cit≥100 → promoted（boost + 入场，不进锚点集）；③ 非种子配额 8；④ 分类 prompt entry 加 citation_count + 分类结果落盘 + venue 上下文从 system prompt 移入 user prompt 前置段 | run 092507：top-60 种子占比 **54/60→11/60** ✓；非种子 4/40（VoxelNet 3457/UniAD 1527/PointPainting 1125/M2BEV 246）✓；Sparse4D #23 ✓；promotion 命中 33 篇（CORE 仅 2——分类器判 promoted 为 ADJACENT/DATASET/FOUNDATION，判得对）；配额 8 未满（CORE 非种子供给恰好 4） |
| V3.3.13 | 2026-08 | Venue 补充入场门槛：citation_count ≥ 同年份 venue 池 P50（小样本 <3 直接放行）+ 分类窗口 150→300。**首次验证（run 161904）失败与校准**：初版门槛人口用全图 cohort（精英偏置）→ venue 通过 61/210、CORE 供给 55→16、final 塌到 19。修正：① 门槛人口 = venue 池自身（顶会同年 cohort；2024 中位 26 淘汰 LION 21/ContrastAlign 20/PointBeV 19，保留 RCBEVDet++ 32；2022 中位 ~119 保留 Sparse4D 209）② 窗口 300 让 BEVDet4D 类（rank 190-380）补位 | 校准后验证 run 162930 ✓：venue 通过 138/272；final 40（0 重复，2015-2025）；BEVDet4D #27 入场 ✓；非种子 24/40（配额 8 超额 3 倍）；promoted CORE 8/8 全入选；venue 补充入 final 10 篇。遗留 Sparse4D：过门槛 + CORE，但 2022 桶排 #8（前 4 入选）落选——V3.3.12 后 venue 补充 prox=0 只剩 0.25·cit 分（→ V3.3.14） |
| V3.3.14 | 2026-08 | 排名公式重构：**删除 PageRank**（雪球图近二值：992 节点 67% pr_norm=0，corr(引用数)=0.24，测的是扩张覆盖度不是重要性）；citation_density → **citation_rate**（log1p(引用数/年龄) 归一化，与 venue 门槛"同年份比较"同一原则，跨年份公平）；权重 γ 0.25→0.8、β 0.30→0.2（旧 prox 差距 0.15 分 > 536 引论文全部 citation 分 0.127 分） | 验证 run 215612 ✓：final 40（0 重复）；PETRv2 335 引 #27、BEVFormer v2 310 引入场 ✓（用户原始投诉的两篇）；2022 桶按引用排序：TransFusion 895/DeepFusion 546/MapTR 430/BEVerse 283（最低 283 引，旧版 M2BEV 246 引压过 MapTR 430）；Sparse4D 209 引 2022 桶 #6 落选——按同年份引用标准输给 BEVerse 283，设计行为 ✓；非种子 22/40。暴露：BEVDet4D 本次未入图（→ 遗留 #2 入场不稳定） |
| V3.3.15 | 2026-08 | 种子解析根因修复（用户报告 unresolved 中混入 3 篇不相关论文：Sora World Simulator 综述 / AV motion planner 数据需求 / ASAP benchmark）：① venue 补充携带完整 OA 记录（`_oa_id` + 元数据），Step 2 直通复用不再按标题重搜（此前丢弃 `_oa_id` 后重搜，arXiv-hosted 标题标点/前缀差异导致失败）；② `_openalex_search` type 过滤改为客户端白名单 `{article, conference-paper, preprint, review}`（OA type filter 不支持 OR 语法返回 HTTP 400；旧 `type:article` 同时排除 preprint 和 OA 给 CS 顶会的独立类型 conference-paper——StreamPETR 242 引正式版即此类型） | 验证 run 225520 ✓：294/294 解析、**0 unresolved**（用户投诉的 3 篇不相关论文不再出现）；final 40（0 重复）；StreamPETR 解析到 242 引 conference-paper 正式版并入选 ✓（修复前 unresolved）；BEVDet4D 536 引经 citation_expansion 图路径入场 ✓（非种子）；Sparse4D CORE 209 引 2022 桶 #10/11 落选——同年份引用竞争，设计行为（桶供给因解析修复变多） |
