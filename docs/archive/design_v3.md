# Paper Context Tool – Design V3 (Seed-Paper-Centric Structured Understanding)

## 1. 项目目标重定义

V2 的目标是：用户输入关键词，系统自动抓取相关论文并生成领域技术路线图。

V3 将核心使用场景转向：**用户提供一篇 pdf论文链接 + 自然语言描述，系统对其进行结构化深度理解，最终生成结构化 Markdown 报告。** 技术路线演进分析作为用户可选功能，由用户根据需求决定是否生成。

### 两种运行模式

| 模式 | 触发方式 | 包含内容 | 适用场景 |
|---|---|---|---|
| **基础分析** | 默认（或 `V3_ROUTE_ANALYSIS_ENABLED=0`） | 结构化理解 + 引用挖掘 + 关键论文分析 | 快速了解一篇论文的方法和贡献 |
| **完整分析** | `V3_ROUTE_ANALYSIS_ENABLED=1` 或 `--route` CLI 参数 | 基础分析 + 技术路线归纳 + 对比分析 | 需要理解论文在领域中的定位和技术演进脉络 |

### 核心能力

1. **结构化理解论文** — 不满足于摘要级别的总结，而是深入分析：
   - 方法架构（网络设计、模块组成、数据流）
   - 关键公式（数学定义、损失函数、优化目标）
   - 训练流程（数据准备、训练策略、超参数）
   - 推理流程（前向传播、后处理）
   - 实验结果（基准对比、消融实验、可视化）
2. **通过引用关系挖掘领域论文** — 双向追溯：
   - **后向**: 种子论文的参考文献 → 递归查找到经典论文
   - **前向**: 查找引用这些经典论文的新论文
   - **分类**: 区分赞同性引用、对比性引用、奠基性引用
3. **归纳主流技术路线** [可选] — 将关键论文按技术路线分组，形成领域全景图
4. **对比分析** [可选] — 种子论文与主流技术路线的设计差异
5. **生成结构化 Markdown** — 一篇完整的论文深度解读报告（章节根据模式动态调整）

---

## 2. 核心数据模型

```python
@dataclass
class Paper:
    """统一论文表示，覆盖种子论文、参考文献、领域论文等所有来源。"""
    id: str                              # 论文唯一标识（Semantic Scholar ID）
    arxiv_id: str | None                 # arXiv ID（如果有）
    title: str
    authors: list[str]
    year: int
    abstract: str
    full_text: str | None                # PDF 提取的全文（关键论文才有）
    citation_count: int
    url: str                             # 论文链接
    source: str                          # "arxiv" | "semantic_scholar" | "pdf_file" | "openalex"
    reference_ids: list[str]             # 本文引用的论文 ID 列表
    structured: StructuredUnderstanding | None  # 结构化理解结果
    user_description: str                # 用户提供的分析聚焦描述


@dataclass
class Reference:
    """引用关系，记录被引论文 ID 及周边信息。被引论文的完整数据通过
    CitationMiner._papers 字典按 paper_id 查询。"""
    paper_id: str                # 被引论文的 Semantic Scholar ID
    paper_title: str             # 被引论文标题（冗余，方便直接显示）
    context: str                 # 在原文中的引用上下文（周围句子）
    citation_type: CitationType  # 引用类型
    is_key_reference: bool       # 是否是关键引用（高被引 / 奠基性）


class CitationType(Enum):
    SUPPORTING = "supporting"       # 赞同/沿用：作者在此工作基础上推进
    CONTRASTING = "contrasting"     # 对比/反对：作者提出替代方案或指出不足
    FOUNDATIONAL = "foundational"   # 背景/奠基：该领域的经典基础工作
    RELATED = "related_work"        # 相关工作：提及但不直接比较
    NOT_CLASSIFIED = "not_classified"


@dataclass
class StructuredUnderstanding:
    """结构化理解 — V3 的核心数据模型，对任意论文可复用的分析结果。"""
    # ── 问题定义 ──
    problem: str                    # 要解决什么问题
    motivation: str                 # 为什么重要 / 现有方法有什么不足
    key_insight: str                # 核心洞察/思路

    # ── 方法架构 ──
    architecture_overview: str      # 整体架构描述（如："encoder-decoder with cross-attention"）
    components: list[Component]     # 核心模块/组件列表
    formulas: list[Formula]         # 关键公式
    architecture_figure: str | None # 对架构图（通常是 Figure 1）的详细解释

    # ── 训练流程 ──
    training_data: str              # 训练数据
    loss_functions: list[str]       # 损失函数
    optimizer: str                  # 优化器及超参数
    training_procedure: str         # 训练流程详细描述（包括 trick、分阶段训练等）

    # ── 推理流程 ──
    inference_procedure: str        # 推理过程
    post_processing: str | None     # 后处理（如 NMS、阈值过滤等）

    # ── 实验结果 ──
    main_results: list[Result]      # 主要实验结果（数据集、指标、数值）
    ablation_results: list[str]     # 消融实验发现
    qualitative_results: str | None # 定性分析/可视化结果

    # ── 贡献与局限 ──
    contributions: list[str]        # 主要贡献
    limitations: list[str]          # 局限性（作者自述或明显可见的）


@dataclass
class Component:
    """模型中的一个模块/组件。"""
    name: str                       # 组件名（如 "Spatial Transformer", "Feature Pyramid Network"）
    purpose: str                    # 功能描述
    details: str | None             # 实现细节（输入/输出维度、层配置等）
    referenced_figure: str | None   # 对应的图表编号（如 "Figure 2(a)"）


@dataclass
class Formula:
    """一个关键公式。"""
    name: str                       # 公式名（如 "Focal Loss", "IoU Loss"）
    latex: str | None               # LaTeX 表达式
    explanation: str                # 含义解释
    significance: str               # 为什么重要


@dataclass
class Result:
    """一项实验结果。"""
    dataset: str                    # 数据集名
    metric: str                     # 指标名
    value: str                      # 数值（保留为字符串以处理各种格式）
    comparison: str | None          # 与 baseline 的对比说明
```

---

## 3. 完整 Pipeline

```
用户输入 arXiv URL/PDF + 自然语言描述 [--route]
    │
    ├── Phase 1: 论文解析
    │   ├─ resolve_paper() — arXiv API 元数据 + Semantic Scholar 补充 + PDF 下载 + 文本提取
    │   └─ 输出: Paper 对象（含 full_text）
    │
    ├── Phase 1.5: 结构化理解（种子论文）
    │   ├─ analyze_paper_structure(seed_paper, llm_client) → StructuredUnderstanding
    │   └─ 输出: Paper.structured（架构、公式、训练/推理流程、实验结果等）
    │
    ├── Phase 2: 后向引用挖掘 [V3_CITATION_MINING_ENABLED=1]
    │   ├─ CitationMiner.mine_references() — Semantic Scholar API 递归引用查找
    │   ├─ CitationMiner.classify_references() — LLM 引用分类（supporting/contrasting/foundational）
    │   └─ 输出: 引用关系池 + 分类结果
    │
    ├── Phase 3: 前向引用挖掘 [V3_CITATION_MINING_ENABLED=1]
    │   ├─ CitationMiner.mine_citations() — Semantic Scholar API 查找引用种子论文的论文
    │   └─ 输出: 扩展论文池
    │
    ├── Phase 4: 关键论文深度分析 [V3_STRUCTURED_ANALYSIS_ENABLED=1]
    │   ├─ CitationMiner.get_key_papers() — 加权排序筛选 Top-N 论文
    │   ├─ 对每篇关键论文: PDF 下载 + 文本提取 + analyze_paper_structure()
    │   └─ 输出: 关键论文列表（含 structured）
    │
    ├── Phase 5: 技术路线分析 [可选，V3_ROUTE_ANALYSIS_ENABLED=1]
    │   ├─ analyze_routes() — LLM 按技术路线分组 + 识别主流路线
    │   ├─ compare_with_mainstream() — 种子论文 vs 主流路线对比
    │   └─ 输出: routes dict + comparison dict
    │
    └── Phase 6: 输出导出
        ├─ export_markdown() — 结构化 Markdown 报告（英文）
        ├─ translate_markdown_to_zh() — LLM 后置中译
        └─ 输出: paper_analysis.md + paper_analysis.zh.md + seed_paper.json + citation_graph.json
```

### 3.1 两种运行模式

**基础分析模式**（默认，`V3_ROUTE_ANALYSIS_ENABLED=0`）：适合快速了解一篇论文。Pipeline 执行 Phase 1-4 + Phase 6 markdown 导出。输出报告包含论文概览、问题定义、方法架构、实验结果、贡献与局限、参考文献分类。不包含领域技术路线和对比分析章节。

**完整分析模式**（`V3_ROUTE_ANALYSIS_ENABLED=1`）：适合需要理解论文在领域中定位的用户。在基础分析之上，增加 Phase 5 技术路线归纳和 Phase 6.1 对比分析。输出报告额外包含"领域技术路线"和"对比分析"两个章节。

用户根据自身需求通过环境变量选择模式，无需为不需要的技术路线分析等待额外的 LLM 调用时间。

### 3.2 自然语言描述的作用与传递

用户提供的自然语言描述（`Paper.user_description`）**贯穿整个 pipeline**，在每个 LLM 分析步骤中作为上下文注入：

```
用户描述示例:
  "我对这篇论文的 temporal attention 机制特别感兴趣，
   想了解它跟传统 self-attention 的差异"

注入位置:
  1. 结构化理解 Prompt → "分析时请特别关注 temporal attention 机制"
  2. 引用分类 Prompt → "重点关注跟 temporal modeling 相关的引用"
  3. 技术路线分析 Prompt → "以 temporal modeling 为切入点归纳分支"
  4. 对比分析 Prompt → "重点对比 temporal attention 设计"
```

**设计原则**: 用户描述作为**分析透镜**（analytical lens），聚焦分析维度而非过滤信息。即使某条技术路线不直接涉及用户兴趣点，仍然会被收录到报告中，只是在对比分析中会突出用户关注的角度。

当用户不提供描述时，所有 LLM 调用使用通用分析指令，不做方向性引导。

### 3.3 数据流图

```
arXiv URL
   │
   ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│  ResolvePaper │───►│ ExtractText+Refs │───►│ StructUnderstanding │
│  (arxiv API)  │    │  (PyMuPDF + LLM) │    │  (LLM 全文分析)     │
└──────────────┘    └──────────────────┘    └──────────────────────┘
                                                    │
                          ┌─────────────────────────┤
                          ▼                         ▼
              ┌───────────────────┐     ┌──────────────────────┐
              │ CitationMiner     │     │ Phase 4: 关键论文分析 │
              │ (SemanticScholar) │     │ (复用 StructUnd.)    │
              │ 后向 + 前向       │     └──────────────────────┘
              └───────────────────┘              │
                          │                      ▼
                          ▼              ┌──────────────────────┐
              ┌───────────────────┐     │ TechnRouteAnalyzer   │
              │ 关键论文候选池    │────►│ 技术路线聚类 + 归纳  │
              └───────────────────┘     └──────────────────────┘
                                                 │
                                                 ▼
                                     ┌──────────────────────┐
                                     │ ComparativeAnalysis + │
                                     │ Markdown Export      │
                                     └──────────────────────┘
```

---

## 4. 结构化理解模块 — 核心抽象

### 4.1 设计目标

`analyze_paper_structure(paper: Paper, llm_client) -> StructuredUnderstanding` 是 V3 **最核心的抽象**。它满足：

1. **论文无关性** — 输入任意论文的全文 + 元数据，输出相同的结构化结构
2. **深度解析** — 不只总结，而是按要求提取 architecture / training / inference 等细节
3. **可复用** — 种子论文和关键论文共用同一函数
4. **优雅降级** — 有全文用全文分析，无全文用 abstract + metadata 降级

### 4.2 Prompt 设计策略

由于需要从长文本中提取结构化信息，prompt 设计上采用 **分步指令 + 结构化输出模板**：

```
1. 首先通读全文，理解论文的核心问题和解决思路
2. 识别架构图（通常是 Figure 1），详细描述其内容和数据流
3. 提取所有带编号的公式，解释每个公式的含义
4. 描述模型的训练流程（数据、loss、优化策略）
5. 描述推理流程（前向传播和后处理）
6. 提取主要实验结果

输出格式为 JSON，必须包含以下字段：
{structured_understanding_schema}
```

### 4.3 长文本处理

当前采用直接截断策略：全文截取前 120K 字符（~30K tokens），保留 abstract 开头 + 正文内容。对于大多数会议/期刊论文（8-14 页）足够覆盖，但对于长综述或博士论文可能丢失后续章节内容。

**未来优化方向**：分块策略 — 将长论文按章节分块（Introduction / Method / Experiments），每块独立提交 LLM 分析，最后汇总。"

### 4.4 降级策略

| 条件 | 行为 |
|---|---|
| 全文可用 | 完整结构化分析 |
| 全文不可用 | 基于摘要 + 标题的分析（少量字段为空） |
| LLM 不可用 | 返回 None，调用方自行处理 |

---

## 5. 引用挖掘策略

### 5.1 数据源选择：Semantic Scholar API

相比 arXiv API 和 OpenAlex，Semantic Scholar 对引用挖掘最友好：

| 能力 | Semantic Scholar | OpenAlex |
|---|---|---|
| 获取参考文献列表 | ✅ `GET /paper/{id}/references` | ✅ `referenced_works` |
| 获取引用论文列表 | ✅ `GET /paper/{id}/citations` | ✅ `cited_by_api_url` |
| 引用上下文（周围文本） | ✅ 部分支持 | ❌ |
| 引用量 | ✅ | ✅ |
| arXiv ID 映射 | ✅ | ✅ |
| 免费额度 | ✅ 100 req/s | ✅ 10 req/s |

### 5.2 引用分类

每篇参考文献由 LLM 根据其在种子论文中的上下文（被引句子 + 周围段落）分类：

```json
{
  "index": 3,
  "title": "BEVFormer: ...",
  "citation_type": "supporting",
  "reason": "作者在方法部分直接沿用 BEVFormer 的 query 设计作为基础组件",
  "is_key": true
}
```

**引用类型定义**：

- **supporting**：作者明确使用/借鉴了该工作（"based on", "following", "extending"）
- **contrasting**：作者指出该工作的不足并提出替代方案（"however", "in contrast", "limitation"）
- **foundational**：该工作是领域公认的奠基性工作，在背景/相关工作部分介绍（通常引用量极高）
- **related**：仅在相关工作或介绍中提及，无直接比较

### 5.3 递归深度控制

```
Phase 2 递归展开：
  Level 0: 种子论文本身
  Level 1: 种子论文的参考文献（从 Semantic Scholar API 获取）
  Level 2: Level 1 中 Key References 的参考文献
  ...

  每层缩窄：
  Level 1 → 最多取 Top-15 篇关键引用
  Level 2 → 从 Level 1 中每篇论文再取 Top-5 篇引用，去重后最多 20 篇
  Level 3+ → 仅追踪极少数高被引论文
```

#### 实际可行性考量

**API 调用量估算**: Level 1 取 15 篇引用需要 1 次 API 调用获取引用列表 + 15 次 API 调用获取每篇引用详情。Level 2 对 15 篇各取 Top-5 引用，去重后约 20 篇，需要 15 次引用列表调用 + 20 次详情调用。总计：Level 1 + Level 2 ≈ 51 次 API 调用。

**延迟优化**: Semantic Scholar 100 req/s 额度足够，但串行调用延迟高。采用以下策略：
- 批量请求：使用 `fields` 参数在单次调用中获取所有需要的信息（citedPaper.title, citedPaper.authors, citedPaper.year, citedPaper.citationCount 等）
- 引用列表一次性获取：`GET /paper/{id}/references?limit=500` 可以在单次调用中获取全部引用，本地再做排序筛选
- 并发处理：用 `concurrent.futures.ThreadPoolExecutor` 并发获取多篇论文的详细信息

**Semantic Scholar 覆盖不足的降级**:
- 如果某篇参考文献在 Semantic Scholar 中没有数据（`status: 404`），不阻塞 pipeline，标记为 `source: "unavailable"`
- 如果 Level 1 中超过 50% 的引用查不到数据 → 改用 OpenAlex API 做为备用数据源
- arXiv ID 到 Semantic Scholar ID 的映射失败 → 尝试用标题搜索 `GET /paper/search?query=`

**递归终止条件**（满足任一即停止）：
1. 达到 max_depth 上限
2. 新发现的关键论文数 < 3 篇（说明这一层已经没有重要引用）
3. 所有新发现论文的引用量 < 50（说明已经超出了核心圈）
4. 出现循环引用（新论文已经在 all_papers 集合中）

### 5.4 关键论文筛选公式

最终论文池的排序综合考虑：

```
score = w1 * citation_count + w2 * recency_bonus + w3 * citation_type_bonus + w4 * ref_frequency

其中：
- citation_count: 绝对引用量（归一化）
- recency_bonus: 越新越高，或越经典越高（取决于任务阶段）
- citation_type_bonus: supporting/foundational 加分
- ref_frequency: 在多个不同论文的引用中都出现（说明是领域共识）
```

---

## 6. 技术路线分析与对比 [可选功能]

技术路线分析是 V3 的**可选增强功能**，由 `V3_ROUTE_ANALYSIS_ENABLED` 环境变量控制（默认关闭）。当用户需要理解论文在领域中的技术定位和演进脉络时，可手动开启。

启用前提：至少完成 Phase 1-4，且有 ≥ 2 篇已分析的关键论文。

### 6.1 按技术路线聚类（替代 V2 的向量聚类）

V2 使用 Embedding + UMAP + HDBSCAN，将主题相似的论文聚类。但同一主题下可能有完全不同的技术路线（例如 BEV 感知中 "LSS-based" vs "Transformer-based" vs "Depth-based"），embedding 无法有效区分。

V3 的技术路线分析分为两步：

**Step 1: LLM 提取技术特征**

对每篇论文的结构化理解，LLM 提取其技术特征向量（非数值向量，而是语义标签）：

```json
{
  "paper_id": "xxx",
  "title": "BEVFormer",
  "technical_tags": [
    "transformer-encoder",
    "spatial-cross-attention",
    "temporal-self-attention",
    "pre-defined-queries",
    "camera-only"
  ]
}
```

**Step 2: 按技术标签分组**

```python
# 方案 A: 直接用 LLM 对论文按技术路线分组
papers_by_approach = llm_group_by_technical_approach(all_papers)

# 方案 B: 技术标签 → 聚类
# 将 technical_tags 编码为稀疏向量 → 计算相似度 → 聚类

# 方案 C: 混合
# LLM 先做粗粒度分组，组内再用标签向量细粒度区分
```

推荐方案 A+C：先用 LLM 识别 3-6 个主要技术路线分支，然后对每个分支归纳其共同技术特征。

### 6.2 主流路线识别

```
主流路线判定标准：
1. 包含论文数量多
2. 总被引用量高
3. 时间跨度大（持续有 work 产出）
4. 包含种子论文中的关键引用
```

### 6.3 对比分析

对比分析的维度：

| 维度 | 种子论文 | 主流路线 A | 主流路线 B |
|---|---|---|---|
| Core Architecture | Transformer | CNN-based | Transformer |
| Input Representation | 3D voxel | 2D image | Point cloud |
| Training Data | nuScenes | KITTI | Waymo |
| Key Design Choice | Sparse attention | Dense convolution | ...

最终输出一段结构化的对比描述，用自然语言 + 表格呈现。

#### 对比分析的 baseline 选择

种子论文在领域中的位置决定了对比分析以什么作为 reference：

| 种子论文类型 | baseline 选择 | 对比目的 |
|---|---|---|
| **新方法提出**（如 DETR 之于 object detection） | 当前该分支的 SOTA 论文 | 证明新方法的优势 |
| **SOTA 推进**（在现有路线上改进） | 该分支的奠基论文 + 上一代 SOTA | 定位进步幅度 |
| **跨路线融合**（借鉴多个路线的设计） | 各被借鉴路线的主论文 | 解释融合带来了什么 |
| **New Problem / New Setting** | 最接近的相关方法（即使不完美匹配） | 说明为什么新设定需要新方法 |

baseline 的确定方式：
1. **自动推断**: LLM 在分析种子论文的结构化理解时，判断论文类型（新方法/SOTA推进/融合/新问题）
2. **用户指定**: 用户可以在自然语言描述中指定希望对比的论文或路线
3. **默认策略**: 选择被引用最多且 citation_type 为 `contrasting` 或 `supporting` 的论文所属路线作为 baseline

对比分析的输出会根据种子论文类型调整侧重点。例如对"跨路线融合"，重点分析种子论文从各路线"借了什么"以及"为什么这样组合"。

---

## 7. Markdown 输出格式

最终输出是一篇完整的结构化报告。章节内容根据运行模式和数据可用性动态调整：

- **基础分析模式**（routes=None, comparison=None）：§1-5 + §8（参考文献分类，有条件）
- **完整分析模式**（routes 和 comparison 有数据）：§1-7 + §8（参考文献分类，有条件）

各章节根据对应数据是否可用自动显隐：
- §6 领域技术路线 — 仅在 `routes` 不为 None 时渲染
- §7 对比分析 — 仅在 `comparison` 不为 None 时渲染
- §8 参考文献分类 — 仅在分类结果非空时渲染

### 7.1 输出结构

```markdown
# [论文标题] — 结构化解读

## 1. 论文概览
- 标题 / 作者 / 年份 / 引用数
- 一句话总结

## 2. 问题定义与动机
- 要解决的问题
- 为什么重要
- 现有方法的不足

## 3. 方法架构
### 3.1 整体架构
### 3.2 架构图详解（← 对 Figure 1 的详细解释）
### 3.3 核心组件
| 组件 | 功能 | 实现细节 | 对应图表 |
### 3.4 关键公式
| 公式名 | 表达式 | 含义 | 重要性 |
### 3.5 训练流程
- 数据准备
- 损失函数
- 优化策略
- 训练细节（trick、超参数）
### 3.6 推理流程
- 前向传播
- 后处理

## 4. 实验结果
### 4.1 主要结果
| 数据集 | 指标 | 本方法 | Baseline | 差距 |
### 4.2 消融实验
### 4.3 定性分析

## 5. 贡献与局限性
### 5.1 主要贡献
### 5.2 局限性

---

## 6. 领域技术路线 [仅完整分析模式]

### 6.1 技术路线全景
（用文字 + 表格描述当前领域的几个主要技术分支）

### 6.2 分支 A: [名称]
- 关键论文
- 共同技术特征
- 演进脉络

### 6.3 分支 B: [名称]
（同上）

---

## 7. 对比分析 [仅完整分析模式]
### 7.1 与主流路线的设计差异
### 7.2 独特贡献定位

---

## 8. 参考文献分类
| 引用论文 | 引用类型 | 重要性 | 说明 |
|---|---|---|---|
| BEVFormer | Supporting | ★★★ | 作为 backbone 组件 |
| Lift-Splat-Shoot | Foundational | ★★★ | 奠基性工作 |
| PointPillars | Contrasting | ★★ | 对比方法 |

---

**图表示例**（留空，由 LLM 生成伪 ASCII 图或文字描述）：
```

### 7.2 双语输出策略

保持 V2 的双语输出能力：英文 Markdown 首先生成，中文翻译通过 LLM 后处理实现。

| 章节 | 翻译策略 |
|---|---|
| 论文概览 / 结构化理解 | LLM 全文翻译 |
| 公式 / 技术术语 / 模型名 | **保留原文**，避免翻译导致歧义（如 "Focal Loss" 不翻译为 "焦点损失"） |
| 实验结果数据 | 数字不变，仅翻译指标描述 |
| 引用分类表 | 论文标题保留原文，说明性文字翻译 |

最终输出以下文件：

```
output/v3/YYYY-MM-DD_title-slug/
├── paper_analysis.md       # 英文版（完整结构化报告）
├── paper_analysis.zh.md    # 中文版（LLM 后置翻译）
├── seed_paper.json         # 种子论文完整元数据
└── citation_graph.json     # 引用关系图数据
```

翻译时保留 Markdown 格式、表格和链接，仅替换自然语言文字。

### 7.3 对比 V2 输出

V2 输出的是领域路线图（Field → Branch → Key Papers → Evolution）。V3 输出的是 **论文深度解读 + 领域背景**，两者侧重点不同。V3 报告包含 V2 的"领域技术路线"部分，但把它作为**理解种子论文的背景信息**，而非最终目的。

---

## 8. 与 V2 的关系

### 8.1 可复用模块

| V2 模块 | V3 复用情况 |
|---|---|
| `config.py` | 完全复用（环境变量 + 路径管理） |
| `llm_analyzer.py` | 复用 JSON 提取函数 + client builder + 重试逻辑，新增结构化分析函数 |
| `markdown_export.py` | 需要大幅扩展以支持新输出格式，或写新的 exporter |
| `llm_namer.py` | 部分复用（`build_llm_client`） |
| `citation_graph.py` | 需要重写（从 OpenAlex 转向 Semantic Scholar） |
| `crawler.py` | **不直接复用**（V3 不需要关键词搜索），但 arXiv PDF 下载逻辑可借鉴 |
| `embedding.py` | **不直接复用**（V3 不依赖向量聚类） |
| `cluster.py` | **不直接复用**（V3 采用 LLM 技术路线分组，非向量聚类） |
| `branch_discovery.py` | **不直接复用**（V3 的"技术路线"由 LLM 直接归纳） |
| `key_paper.py` | 可选复用作为降级策略 |
| `timeline.py` | 可选复用 |

### 8.2 共存策略

V2（`main.py`）和 V3（新入口）共存于同一仓库。V2 保持不变，V3 作为新的 pipeline 入口：

```
src/
  main.py              # V2 入口（关键词模式）
  run_v3.py            # V3 入口（arXiv 链接模式）
  paper.py             # [新] Paper / StructuredUnderstanding 数据类
  paper_resolver.py    # [新] arXiv 链接解析、元数据获取、PDF 下载
  text_extractor.py    # [新] PDF 文本提取 + 参考文献解析
  structured_analyzer.py # [新] 结构化理解引擎（核心）
  citation_miner.py    # [新] Semantic Scholar 引用挖掘
  route_analyzer.py    # [新] 技术路线归纳 + 对比分析
  markdown_exporter_v3.py # [新] V3 Markdown 输出
  config.py            # 复用
  llm_analyzer.py      # 复用 + 扩展
```

---

## 9. 项目结构（V3）

```
paper_context_tool/
├── src/
│   ├── main.py                    # V2 入口（不变）
│   ├── run_v3.py                  # [新] V3 入口
│   │
│   ├── paper.py                   # [新] 数据类：Paper, Reference, StructuredUnderstanding, ...
│   ├── paper_resolver.py          # [新] arXiv URL 解析 + 元数据获取 + PDF 下载
│   ├── text_extractor.py          # [新] PDF 文本提取 + 参考文献解析
│   ├── structured_analyzer.py     # [新] 结构化理解引擎（核心抽象）
│   ├── citation_miner.py          # [新] 引用挖掘（Semantic Scholar API）
│   ├── route_analyzer.py          # [新] 技术路线归纳 + 对比分析
│   ├── markdown_exporter_v3.py    # [新] V3 Markdown 输出
│   │
│   ├── config.py                  # 复用（扩展 V3 配置）
│   ├── llm_analyzer.py            # 复用（JSON 提取 + client builder）
│   ├── key_paper.py               # 可选复用（降级）
│   ├── timeline.py                # 可选复用
│   ├── ...
│
├── docs/
│   ├── design_v2.md
│   ├── design_v3.md               # [新] 本文档
│
├── tests/
│   ├── test_structured_analyzer.py # [新]
│   ├── test_citation_miner.py      # [新]
│   ├── test_paper_resolver.py      # [新]
│   ├── test_text_extractor.py      # [新]
│   ├── test_route_analyzer.py      # [新]
│   ├── conftest.py                 # 复用 + 扩展
│
├── data/
│   └── paper_cache/               # [新] 论文 PDF / 文本缓存
│
├── output/
│   └── v3/                        # [新] V3 输出目录
│       └── YYYY-MM-DD_title-slug/
│           ├── paper_analysis.md       # 结构化分析报告（英文）
│           ├── paper_analysis.zh.md    # 结构化分析报告（中文）
│           ├── seed_paper.json         # 种子论文元数据（JSON）
│           └── citation_graph.json     # 引用关系数据（JSON）
│
└── requirements.txt               # 扩展依赖
```

### 新增依赖

```
# requirements.txt 新增
PyMuPDF                  # PDF 文本提取
semantic-scholar-api     # Semantic Scholar API（或直接使用 requests + REST API）
```

---

## 10. 配置（环境变量，V3 新增）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SS_API_KEY` | `""` | Semantic Scholar API Key（可选，提高限流） |
| `REFERENCE_MAX_DEPTH` | `2` | 引用递归展开最大深度 |
| `REFERENCE_TOP_K_LEVEL1` | `15` | Level 1 最大引用数 |
| `REFERENCE_TOP_K_LEVEL2` | `20` | Level 2 最大引用数 |
| `KEY_PAPERS_TOTAL` | `30` | 最终参与分析的关键论文总数 |
| `TEXT_CHUNK_SIZE` | `4000` | 全文分块大小（tokens） |
| `TEXT_CHUNK_OVERLAP` | `200` | 分块重叠（tokens） |
| `V3_OUTPUT_DIR` | `output/v3` | V3 输出目录 |
| `PAPER_CACHE_DIR` | `data/paper_cache` | PDF / 文本缓存目录 |
| `V3_STRUCTURED_ANALYSIS_ENABLED` | `1` | 开关：LLM 结构化分析 |
| `V3_CITATION_MINING_ENABLED` | `1` | 开关：引用挖掘（Semantic Scholar） |
| `V3_ROUTE_ANALYSIS_ENABLED` | `0` | **开关：技术路线归纳 + 对比分析（用户可选，默认关闭）** |
| ... | (其余 V2 配置不变) | |

---

## 11. 成本估算（相对于 V2）

### 基础分析模式（默认，`V3_ROUTE_ANALYSIS_ENABLED=0`）

| 步骤 | Token 估算 | 费用（DeepSeek chat） |
|---|---|---|
| 结构化理解（种子论文，~5000 tokens 输入） | ~7K | ~$0.00014 |
| 引用分类（~30 篇引用 × ~200 tokens 输入） | ~12K | ~$0.00024 |
| 结构化理解（~10 篇关键论文 × ~5000 tokens） | ~70K | ~$0.0014 |
| **总计（基础模式）** | **~89K** | **~$0.0018** |

### 完整分析模式（`V3_ROUTE_ANALYSIS_ENABLED=1`）

| 步骤 | Token 估算 | 费用（DeepSeek chat） |
|---|---|---|
| 基础模式总计 | ~89K | ~$0.0018 |
| 技术路线分析 | ~5K | ~$0.0001 |
| 对比分析 | ~3K | ~$0.00006 |
| **总计（完整模式）** | **~97K** | **~$0.002** |

每次运行约 $0.0018-$0.002，比 V2 高一个数量级但依然极低。如果使用 GPT-4，成本会显著增加（约 $0.5-1.0）。

### PDF 处理成本

下载和文本提取免费。如果使用外部 PDF 解析服务（如 Nougat、Grobid），会有额外费用或部署成本。

---

## 12. 优雅降级

| 步骤 | 正常 | LLM 不可用 | API 不可用 |
|---|---|---|---|
| arXiv URL 解析 | ✅ | ✅（不依赖 LLM） | ❌ 退出 |
| PDF 下载 | ✅ | ✅ | ❌ 提示用户手动下载 |
| 文本提取 | ✅ | ✅ | ✅（PyMuPDF 本地运行） |
| 结构化理解 | LLM 全文分析 | 返回基本结构（仅摘要） | 返回基本结构 |
| 引用挖掘（Semantic Scholar） | ✅ | ✅（不依赖 LLM） | ❌ 使用 OpenAlex 降级 |
| 引用分类 | LLM 分类 | 全部标记为 not_classified | 全部标记为 not_classified |
| 递归展开 | ✅ | ✅（不依赖 LLM） | 有限展开 |
| 关键论文分析 | 全文结构化分析 | 摘要级分析 | 跳过 |
| 技术路线归纳 | 用户可选（`V3_ROUTE_ANALYSIS_ENABLED=1` 时启用）。LLM 分析 | 两步降级：① LLM 提取 technical_tags 后，用 Jaccard 相似度做 manual grouping（无需额外 LLM 调用）② 完全无 LLM 时，按年份分组 + 按 technical keywords 相同度合并（基于 paper.keywords / 标题 TF-IDF） | 跳过 |
| 对比分析 | 用户可选（依赖 Phase 5 完成）。LLM 生成对比表格 + 文本描述 | 仅输出技术标签对比矩阵（种子论文标签 vs 每组的共有标签），无自然语言描述 | 跳过 |
| Markdown 输出 | 完整报告（基础模式）或含路线分析的完整报告（完整模式） | 简化报告 | 简化报告 |

**降级原则**（与 V2 一致）：每一步独立降级，单点故障不中断 pipeline。技术路线分析本身作为可选功能，关闭时不影响基础报告质量。

---

## 13. 实现路线图

### Phase 1 — 基础能力（数据层）
1. 实现 `paper.py` 数据类
2. 实现 `paper_resolver.py`（arXiv URL → Paper 对象）
3. 实现 `text_extractor.py`（PDF → 全文文本）
4. 测试：单篇论文的完整解析链路

### Phase 2 — 结构化理解（核心抽象）
1. 实现 `structured_analyzer.py`（LLM 驱动的结构化分析）
2. 分块策略 + 提示词设计
3. 测试：多篇不同论文的结构化理解

### Phase 3 — 引用挖掘
1. 实现 `citation_miner.py`（Semantic Scholar API）
2. 后向引用 + 引用分类
3. 前向引用
4. 递归展开
5. 测试：一个已知领域的引用关系网

### Phase 4 — 技术路线 + 对比
1. 实现 `route_analyzer.py`
2. 技术路线分类
3. 对比分析
4. 测试：已知领域的路线归纳

### Phase 5 — 输出
1. 实现 `markdown_exporter_v3.py`
2. `run_v3.py` 入口整合
3. 测试：端到端流程

---

## 14. 未解决的问题 / 待讨论

1. **PDF 公式提取**：PyMuPDF 提取的文本中公式通常是乱码。是否需要用 LaTeX 源码替代 PDF？
   - 替代方案：arXiv 提供 LaTeX 源代码下载，解析 `.tex` 文件可获得干净公式
   - 代价：LaTeX 编译环境 / 解析复杂度增加
2. **引用上下文获取**：Semantic Scholar 的引用上下文覆盖率不高（~50%）。对无上下文的引用，如何分类？
3. **非 arXiv 论文**：参考文献中大量非 arXiv 论文（会议、期刊），如何获取全文？
   - 方案：不获取全文，仅基于摘要分析
4. **长论文分块**：超过 10 页的论文（如 survey）可能超过 LLM 一次性处理长度，分块策略的准确性如何保证？
5. **V2 和 V3 的维护关系**：V2 是否继续维护？还是逐步迁移到 V3？
