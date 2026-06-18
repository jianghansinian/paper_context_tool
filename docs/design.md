# Paper Tool — 系统设计文档

## 1. 系统总览

Paper Tool 由两个独立但互相调用的子系统组成：

```
V3: Single Paper Understanding          V4: Research Narrative Engine
┌─────────────────────────────┐         ┌──────────────────────────────┐
│ 输入：一篇论文（arXiv URL/PDF）│         │ 输入：一个领域 / 一篇种子论文    │
│ 输出：结构化深度分析报告      │         │ 输出：领域技术发展叙事          │
│                             │         │                              │
│ • 方法架构与公式              │    ───→ │ • 每个分支包含关键论文          │
│ • 训练/推理流程               │  Paper  │ • 论文关联 Claim 锚点          │
│ • 实验结果与消融              │  ←───  │ • Paradigm Shift 检测         │
│ • 贡献与局限                  │         │ • 演化故事（Conflict→Breakthrough）│
│ • 领域定位（可选）             │         │                              │
└─────────────────────────────┘         └──────────────────────────────┘
```

**核心原则**：V3 回答"这篇论文做了什么"，V4 回答"这个领域为什么发展成今天这样"。两者共享 Paper + StructuredUnderstanding 数据模型，V4 消费 V3 的分析结果生成叙事。

### 1.1 两个用户旅程

**旅程 A：从论文到领域**
```
用户提供论文 → V3 深度解析 → 用户读完后自然追问"这篇论文在领域中的位置？"
→ 调用 V4，以该论文为种子，展示领域演化全景
→ Narrative 中高亮标注该论文的位置
```

**旅程 B：从领域到论文**
```
用户搜索领域/问题 → V4 领域叙事 → 每个分支节点附带关键论文
→ 用户对某个节点感兴趣 → 调用 V3 深度解析该论文
```

### 1.2 数据流（单向依赖）

```
Paper（原始论文元数据 + 全文）
  ↓
V3 analyze_paper_structure() → StructuredUnderstanding
  ↓                                    ↓
  ↓                               V4 消费多个 V3 分析结果
  ↓                                    ↓
  ↓                          claim_extractor → Claim
  ↓                          claim_relation_builder → ClaimRelation
  ↓                          research_question_detector → ResearchQuestion  ← V6 新增
  ↓                          tension_detector → Tension（从属于 RQ）
  ↓                          paradigm_shift_detector → ParadigmShift
  ↓                          narrative_builder → ResearchNarrative
  ↓                                    ↓
  └────────────────────────────────────┤
                                       ↓
                              Markdown Report（含 V3 深度 + V4 叙事）
```

V3 是 V4 的生产者，V4 是 V3 的导航层。依赖单向，不形成循环。

---

## 2. 核心数据模型

### 2.1 V3 现有模型（保留，简要列出）

```python
@dataclass
class Paper:
    id: str                              # Semantic Scholar ID 或 "arxiv:XXXX"
    arxiv_id: Optional[str]
    title: str
    authors: list[str]
    year: int
    abstract: str
    full_text: Optional[str]             # PDF 提取全文
    citation_count: int
    url: str
    source: str                          # "arxiv" | "semantic_scholar" | "pdf_file"
    reference_ids: list[str]
    structured: Optional[StructuredUnderstanding]
    user_description: str

@dataclass
class StructuredUnderstanding:
    # 问题定义
    problem: str
    motivation: str
    key_insight: str
    field_evolution: Optional[str]
    core_question: Optional[str]
    related_work_context: Optional[str]

    # 方法架构
    architecture_overview: str
    architecture_figure: Optional[str]
    components: list[Component]          # {name, purpose, details, referenced_figure}
    formulas: list[Formula]              # {name, latex, explanation, significance}
    design_rationale: Optional[str]
    intuitive_analogy: Optional[str]

    # 训练/推理
    training_data: str
    data_engineering: Optional[str]
    training_stages: list[dict]
    loss_functions: list[str]
    optimizer: str
    training_procedure: str
    inference_procedure: str
    post_processing: Optional[str]
    deployment_architecture: Optional[str]

    # 实验结果
    evaluation_setup: Optional[str]
    main_results: list[Result]           # {dataset, metric, value, comparison}
    ablation_results: list[str]
    qualitative_results: Optional[str]
    industry_comparison: list[dict]

    # 贡献与局限
    contributions: list[str]
    limitations: list[str]
    synthesis: Optional[str]
```

### 2.2 V4 新增模型

**设计原则**：V8 建模的是"领域如何通过解决一个个矛盾而演进"。叙事的自然单位是 **Phase（时间阶段）**——Phase 不是独立实体，而是 **Tension 时间聚类后涌现的顶层矛盾**。

V4→V5→V6→V8 演化：
```
V4: Branch → V5: Tension → V6: ResearchQuestion → V8: Phase（Tension聚类涌现）
     （聚类）      （矛盾）        （问题）              （时间因果链）
```

**V6/V7 的教训**：用 RQ 做章节标题破坏了时间因果链。每个 RQ 章节跨越相同的时间段（2020-2024），读者读完一章又跳回去。正确的叙事单元是**时间阶段**——每个阶段有一个核心矛盾，阶段的遗留问题自然引出下一阶段。

**Phase 的本质**：Phase 不新增实体——它是 Tension 时间聚类的结果。某段时间内，整个领域围着一个核心矛盾转。

```python
@dataclass
class ResearchQuestion:
    """一个研究问题 — V8 中是 Phase 的内容，不是章节标题。

    RQ 是领域曾争论过的问题。它们比 Tension 更稳定，但不适合做叙事章节标题
    （会破坏时间因果链）。在 V8 中，RQ 是 Phase 内的"情节"——每个 Phase 有一个
    核心 RQ，Phase 内的论文围绕它在辩论。
    """
    question: str                        # 完整问题文本，以 "?" 结尾
    short_name: str                      # 简短标签，如 "Depth Necessity"
    description: str                     # 1-2 句背景
    level: str                           # field | paradigm | engineering
    status: str                          # direction_clear | direction_forming | open
    positions: list[dict]                # [{"paper": "...", "position": "...", "evidence": "..."}]
    introduced_by: list[str]             # 首次提出此问题的论文标题
    tensions: list[Tension] = []         # 从属的 Tension
    direction: Optional[Direction] = None  # V7 新增：证据指向的结论

@dataclass
class Direction:
    """一个研究方向的结论 — V7 新增，从属于 RQ。

    与 ParadigmShift 的区别：Direction 是 per-RQ 的（"社区偏向隐式几何"），
    ParadigmShift 是跨 Phase 的（"Dense→Sparse 改变了表示哲学"）。
    """
    statement: str                       # e.g. "Implicit geometry learning is sufficient"
    support_papers: list[str]            # 支持此方向的论文
    opposing_papers: list[str]           # 反对或复杂化此方向的论文
    confidence: str = "medium"           # high | medium | low
    evidence_summary: str = ""           # 1-2 句关键证据摘要

@dataclass
class Claim:
    """一篇论文的可证伪断言 — V4 的建模原子。"""
    paper_id: str                        # 来源论文
    paper_title: str                     # 冗余，方便显示
    year: int
    statement: str                       # 核心断言（可证伪的判断，非方法描述）
    evidence: str                        # 支撑证据（实验结果、消融等）
    problem_addressed: str               # 这个 Claim 解决什么问题
    claim_type: str                      # "improves" | "extends" | "replaces" | "combines" | "introduces"
    claim_level: str = "methodological"  # "paradigm" | "methodological" | "engineering"（V7 新增）

@dataclass
class ClaimRelation:
    """两个 Claim 之间的关系 — 演化图的边。

    V5.1: 单次 LLM 调用直接分类，不需要单独的 lineage gatekeeper。
    "parallel" 只是 relation 的一个可能值 — 当两篇论文服务于不同的下游任务时。
    """
    source_paper: str                    # 论文标题
    target_paper: str                    # 论文标题
    source_claim: str                    # Claim 文本
    target_claim: str                    # Claim 文本
    relation: str                        # attack|replace|improve|extend|support|parallel
    explanation: str                     # 关系解释
    source_year: int
    target_year: int

@dataclass
class Tension:
    """研究张力 — V8 中 Stage 1 检测所有细粒度矛盾，Stage 2 合并为 Phase。

    一个 Tension 是领域内的一个矛盾——两个阵营对同一问题给出对立回答。
    V8 中 Tension 不限制数量（通常 8-12 个），在 Stage 2 按时间+主题合并为 2-4 个 Phase。
    Tension 保留细粒度矛盾信息，Phase 是叙事章节标题。

    状态语义：
    - direction_clear: 社区在特定范围内形成了主导方向（不是"已解决"，新证据可能重新打开）
    - direction_forming: 竞争方案存在，但某个方向正在获得 traction
    - open: 活跃争论中，无明确方向
    """
    tension: str                         # 短标签（如 "Dense vs Sparse Representation Efficiency"）
    description: str                     # 详细描述
    introduced_by: list[str]             # 哪个/哪些论文首次暴露了这个矛盾
    resolved_by: list[str]               # 哪个/哪些论文推进或倾向了某个方向
    status: str                          # direction_clear | direction_forming | open
    dimension: str                       # representation|geometry|system|evaluation
    domain_scope: str = ""               # 适用域（如 "in detection/tracking on nuScenes"）

@dataclass
class Phase:
    """一个时间阶段 — V8 的叙事章节单元，从 Tension 时间聚类涌现。

    Phase 不是独立实体——它是多个 Tension 按时间+主题合并后的顶层矛盾。
    Phase 间有因果链: Phase N 的 unresolved_problem → Phase N+1 的 core_contradiction。
    """
    name: str                            # 问题驱动名称，如 "How to Build a 3D View from 2D Images?"
    time_range: str                      # 时间范围，如 "2020-2022"
    core_contradiction: str              # 核心矛盾（1 句话，<100 字符）
    key_papers: list[str]                # 关键论文（2-5 篇）
    core_debate: str                     # 核心辩论（这个阶段领域在争论什么）
    unresolved_problem: str              # 未解决问题（→ 成为下一阶段 motivation）
    tensions: list[Tension]              # 属于此 Phase 的 Tension

@dataclass
class ParadigmShift:
    """范式转移 — 领域共识的根本改变。
    
    严格区分范式转移（领域共识改变）和技术演进（方法改进但共识不变）。
    标准：旧范式下的研究者会认为新范式"难以置信"或"不可理喻"。
    全场最多 3 个范式转移。
    """
    shift_name: str                      # 短标签（如 "Dense BEV → Sparse Representation"）
    description: str                     # 2-3 句：改变了什么，为什么重要
    old_paradigm: str                    # 旧共识
    new_paradigm: str                    # 新共识
    catalyst_papers: list[str]           # 触发/结晶此转移的论文
    magnitude: str                       # paradigm_shift | optimization | convergence
    level: str                           # research_question | method | evaluation
    dimension: str                       # representation|geometry|system|evaluation
    year_range: str                      # e.g. "2022-2024"

@dataclass
class NarrativeSection:
    """叙事的一个章节 — V8 每个 Phase 独占一节。

    演化：
    - V5.1: 按 Tension.dimension（4 个硬编码类别）分组 → 论文跨节重复，维度标签空洞
    - V5.2: 每个 Tension = 一个 Section → 7-8 个节，Tension 重叠严重
    - V6/V7: 每个 ResearchQuestion = 一个 Section → 3-5 个节，问题驱动
        问题：RQ 章节跨相同时间段，破坏时间因果链
    - V8: 每个 Phase = 一个 Section → 2-4 个节，时间因果链

    叙事模型：教授讲课 — "上次我们看到... 这次讲..."
    Phase 是章节，RQ 是 Phase 内的情节。每个章节内嵌叙事+演化图+主张表+方向判断。
    """
    title: str                           # Phase 名称，如 "How to Build a 3D View from 2D Images? (2020-2022)"
    phase: Phase                         # 本章节对应的 Phase
    narrative: str                       # DeepSeek 风格叙事文本（短段落+粗体+分点）
    claims: list[Claim]                  # Phase 时间范围内的 claims
    claim_relations: list[ClaimRelation] # Phase 内的 relations
    paradigm_shifts: list[ParadigmShift] # 涉及此 Phase 的范式转移
    direction: Optional[Direction] = None  # Phase 特定的方向判断（year-filtered）

@dataclass
class ResearchNarrative:
    """V8.1 输出：一个领域的完整技术发展故事。

    V8.1: idea-centric 叙事。论文退居脚注，思想（Phase/Debate/Turning Point）
    成为叙事主角。Direction 精简为 3 行结论。证据不重复三次。
    """
    field_name: str
    seed_paper_id: Optional[str]
    overview: str                        # 领域全景 — 1 短段（原 3 段压缩）
    sections: list[NarrativeSection]     # 按 Phase 分节
    phases: list[Phase]
    research_questions: list[ResearchQuestion]
    paradigm_shifts: list[ParadigmShift] # 0-5 条，渲染为一句结论
    tensions: list[Tension]
    claims: list[Claim]
    claim_relations: list[ClaimRelation]
    open_questions: list[str]            # V8.1 新增: 0-3 条开放问题
    reading_list: list[dict]             # V8.1 新增: 按 Phase 分组的阅读列表
    synthesis: str                       # 趋势总结（Evidence-Backed 合并了 Speculative）
```

**已移除的模型：**
- `Branch` — 分组论文的职责被 NarrativeSection 替代，且 Branch 存在三个不可修复的问题：聚类不稳定、截断跨 phase 因果链、论文归属排他与实际演化不符
- `EvolutionEdge` — 设计文档残留，从未实现，被 ClaimRelation 覆盖

### 2.3 Claim vs Solution 的区分标准

这是 V4 最关键的抽象。Prompt 设计必须教会 LLM 区分：

| | Solution（方法描述） | Claim（核心断言） |
|---|---|---|
| 定义 | 论文做了什么 | 论文断言什么是对的/更好的 |
| 特征 | 描述性的，可复现的 | 判断性的，可证伪的 |
| 例子 | "We use V-JEPA for video pre-training" | "V-JEPA pretraining can outperform perception-heavy pipelines by 3 PDMS" |
| 例子 | "Sparse Query mechanism" | "Dense BEV representation is unnecessary; sparse representation preserves performance at lower cost" |
| 验证方式 | 照着做能复现 | 对照实验能推翻 |

**判定测试**：如果一个句子可以被"不对，实验证明相反"反驳，它就是 Claim。如果只能被"你没写清楚实现"质疑，它就是 Solution。

---

## 3. V3：单篇论文结构化分析

### 3.1 定位

V3 是系统的基础能力。输入一篇论文，输出完整的结构化深度分析。

### 3.2 Pipeline

```
Phase 1: 论文解析
  ├─ arXiv API 元数据 + Semantic Scholar 补充
  ├─ PDF 下载（重试 + SS fallback）+ PyMuPDF 文本提取
  └─ 输出: Paper 对象（含 full_text）

Phase 1.2: 领域 + 论文类型检测
  ├─ detect_domain(paper, llm) → DomainProfile
  └─ detect_paper_type(paper, domain, llm) → PaperTypeProfile（即 schema）

Phase 1.5: 结构化理解（核心）
  ├─ analyze_paper_structure(paper, llm, profile, domain)
  ├─ Schema → Prompt 生成 → LLM 分析 → dict → StructuredUnderstanding
  └─ 输出: Paper.structured

Phase 2-3: 引用挖掘 [可选]
  ├─ Backward: Semantic Scholar 递归引用查找
  ├─ Forward: 引用种子论文的论文
  └─ LLM 引用分类 (supporting/contrasting/foundational/related)

Phase 4: 关键论文分析 [可选]
  ├─ 加权排序筛选 Top-N 论文
  ├─ 并行 PDF 下载 + 结构化分析
  └─ 输出: 关键论文列表（含 structured）

Phase 5: 技术路线分析 [可选，--route]
  ├─ analyze_routes() → 技术分支分组 + 主流识别
  └─ compare_with_mainstream() → 对比分析

Phase 6: 导出
  ├─ Markdown（英文）→ LLM 中译
  └─ seed_paper.json + citation_graph.json
```

### 3.3 Schema 驱动架构（已部分实现）

分析维度和输出结构由 `DomainProfile → PaperTypeProfile` 定义。Schema 包含：
- `fields: list[FieldDef]` — 定义提取哪些字段、每个字段的类型、prompt、是否必填
- `sections: list[SectionDef]` — 定义输出章节结构、各章节包含哪些字段

当前已实现 `ai_ml` 领域的 schema，其他领域（biology, materials_science）已在 `design_schema_driven.md` 中预设计。

### 3.4 输出格式

```markdown
# 结构化理解：[论文标题]

## 1. 论文概述 — meta + key_insight
## 2. 问题定义与动机 — field_evolution + problem + motivation + core_question + related_work_context
## 3. 方法架构 — overview + figure + components + design_rationale + training + inference
## 4. 实验结果 — evaluation_setup + main_results + ablation + qualitative + industry_comparison
## 5. 贡献与局限性 — contributions + limitations + synthesis
## 6. 在领域演化中的位置 [当 V4 可用时]
## 7. 参考文献分类 [有条件]
```

---

## 4. V4：领域技术叙事引擎

### 4.1 第一性原理
**技术发展史的本质是阶段（Phase）的因果推进——每个阶段有一个核心矛盾，阶段的遗留问题自然引出下一阶段。**

V4→V5→V6→V8 的认知演化：

```
V4: Paper-Centric → 错误
    Paper A → Paper B → Paper C
    问题：默认时间顺序 = 因果顺序

V5: Tension-Centric → 不够稳定
    Tension 1 → Paper Response → Tension 2 → ...
    问题：Tension 容易被论文表述层面过拟合
         7-8 个 Tension 中有一半是工程噪音或彼此重叠

V6/V7: Question-Centric → 稳定但不顺
    ResearchQuestion → Positions → Answer direction
    优势：问题是知识骨架，3-5 个问题自然收束
    问题：RQ 做章节标题破坏时间因果链，每章跨越相同时间段（2020-2024）
         读者读完一章又跳回去，无法感知技术"往前推进"

V8: Phase-Centric → 时间因果链
    Phase 1 (2020-2022 core contradiction) → Phase 2 (2022-2023 next contradiction) → Phase 3 (2023-2024)
    优势：每个 Phase 是一个时间段，Phase 间有因果链（Phase N 的遗留问题 → Phase N+1）
         读者跟着时间往前走，不跳回
         与 DeepSeek 的 4 讲结构一致：按时间阶段组织，线性推进
```

**V6/V7 的核心教训**：用 RQ 做章节标题导致每章都是"从 2020 到 2024 重新走一遍"。DeepSeek 的 4 讲结构证明了正确做法是按时间阶段组织——每讲有一个核心矛盾，讲完进入下一阶段。"Dense era → Sparse era → End-to-end era"的叙事节奏是读者最能感知趋势的。

**Phase 的本质**：Phase 不新增实体——它是 Tension 时间聚类后涌现的顶层矛盾。某段时间内，整个领域围着一个核心矛盾转。

**两阶段检测流程**：
1. **Stage 1: 检测所有 Tension**（8-12 个细粒度，不限制数量）— 输入 Claim + ClaimRelation，输出所有能识别到的矛盾
2. **Stage 2: 合并为 Phase**（LLM 按时间+主题二次合并）— 输入所有 Tension，按时间重叠和主题相似度聚类，输出 2-4 个 Phase

**Phase 的因果链约束**：
- 每个 Phase 有一个 `unresolved_problem` 字段（这个阶段没解决的问题）
- Phase N 的 `unresolved_problem` → Phase N+1 的核心矛盾
- 这形成了"但这又产生了新问题..."的因果推进

**V8 概念分工**：
- **Tension** — 检测所有矛盾（Stage 1，8-12 个细粒度，不做数量限制）
- **Phase** — Tension 时间聚类涌现的顶层矛盾（Stage 2，2-4 个）← **V8 章节标题**
- **ResearchQuestion** — Phase 内的"情节"（领域围绕这个 Phase 争论的具体问题）← 不再是章节标题
- **Claim/ClaimRelation** — 定义事实和证据关系（不变）
- **ParadigmShift** — 跨 Phase 的领域共识改变（不变）
- **Direction** — per-RQ 的证据指向（不变，从属于 RQ）

**V8 建模示例**（问题驱动命名 + 时间约束）：
```
Phase 1: "How to Build a 3D View from 2D Images?" (2020-2022)
  Core contradiction: Camera-to-BEV needs depth but dense projection is expensive
  Key papers: LSS, BEVDet, BEVDepth, BEVFormer, BEVFormer v2
  Core debate: Is explicit depth supervision necessary?
  Unresolved: Dense BEV grids waste computation on empty space
  约束: 禁止引用 2023+ 论文（SparseBEV 等尚未发表）

Phase 2: "Can Sparse Representations Match Dense Accuracy?" (2022-2023)
  Core contradiction: Sparse methods reduce cost but need temporal fusion
  Key papers: Sparse4D, SparseBEV, Sparse4D v2, BEVDet4D
  Core debate: Can sparse queries fully replace dense grids?
  Unresolved: Sparse methods lack unified architecture for end-to-end tasks

Phase 3: "What Is the Optimal Representation for End-to-End Driving?" (2023-2024)
  Core contradiction: Task unification demands one representation but tasks need specialization
  Key papers: UniAD, VAD, SparseDrive
  Core debate: Should planning use rasterized, vectorized, or sparse representations?
  Unresolved: No single representation optimally handles all tasks
```
### 4.2 Pipeline

```
输入：领域描述 / 种子论文
  ↓
Phase A: 论文检索 (paper_retriever)
  ├─ 阶段1: Citation Expansion（backward + forward，50-100篇，高精度）
  ├─ 阶段2: Problem-based Semantic Search（用提取的 problem 做查询，补充遗漏）
  └─ 目标候选池: 80-120篇

Phase B: 论文理解 (paper_understanding)
  ├─ 复用 V3 的 analyze_paper_structure()
  ├─ 种子论文: 完整 full-text 分析
  ├─ Top-20候选: 有 full-text 则完整分析，否则 abstract 降级
  └─ 其余: abstract + metadata 轻量分析

Phase C: Claim 提取 (claim_extractor) ← 最关键模块
  ├─ 输入: Paper.title + abstract + StructuredUnderstanding（如有 full-text）
  ├─ LLM 提取: {statement, evidence, problem_addressed, claim_type}
  ├─ Prompt 设计: few-shot 正反例教 LLM 区分 Claim vs Solution
  └─ 输出: list[Claim]

Phase D: Claim 关系构建 (claim_relation_builder)
  ├─ V5.2: 先按 downstream task 分组（detection/planning/tracking/...），
  │   组内按时序建链（LLM 分类），跨 task 相邻对直接标记 parallel
  │   防止时间相邻但任务不同的论文被错误赋予因果语义
  ├─ 单次 LLM 调用直接分类: attack|replace|improve|extend|support|parallel
  ├─ "parallel" 仅当两篇论文服务于不同下游任务（detection vs planning）
  │   同一任务的不同方法（dense vs sparse detection）→ replace/attack，不是 parallel
  ├─ O(N) 而非 O(N²)
  └─ 输出: list[ClaimRelation]

Phase E: 两阶段 Tension→Phase 检测 (tension_detector) ← V8 重构
  ├─ Stage 1: detect_all_tensions(claims, relations)
  │   ├─ 检测所有研究张力，不限制数量（通常 8-12 个细粒度 tension）
  │   ├─ 每个 Tension: 矛盾描述、引入者、解决者、时间范围、所属维度
  │   └─ 输出: list[Tension]（全部，不做截断）
  ├─ Stage 2: merge_tensions_into_phases(tensions, claims)
  │   ├─ LLM 按时间+主题将 Tension 合并为 2-4 个 Phase
  │   ├─ 每个 Phase: 时间范围、核心矛盾、关键论文、核心辩论、未解决问题
  │   ├─ 因果链约束: Phase N 的未解决问题 → Phase N+1 的核心矛盾
  │   ├─ 内容均衡约束: 每个 Phase 2-5 篇关键论文，2 篇也可接受（若形成连贯叙事弧线）
  │   ├─ 命名规范: 问题驱动标题（"How/Can/Should...?"），不用静态标签
  │   ├─ 长度约束: core_contradiction/unresolved_problem 控制在 1 句以内（<100 字符）
  │   └─ 输出: list[Phase]（2-4 个，按时间排序）
  └─ 输出: list[Tension] + list[Phase]

Phase E.5: 研究问题检测 (research_question_detector) ← V8 保留，退居 Phase 内容
  ├─ 输入: Claim（全部论文的断言）+ Phase 上下文
  ├─ LLM 识别 3-5 个核心研究问题（ResearchQuestion）
  ├─ 每个 RQ 标记: question 文本、short_name、level（field/paradigm/engineering）、
  │   status、papers→positions 映射、introduced_by
  ├─ RQ 嵌套 Tension 和 Direction（V7 的嵌套结构保留）
  ├─ RQ 不再是章节标题——它是 Phase 内的"情节"，不同 Phase 可以涉及同一 RQ 的不同侧面
  ├─ 自动过滤: 只有 field/paradigm 级别的 RQ 才会出现在叙事中
  └─ 输出: list[ResearchQuestion]

Phase F: 范式转移检测 (paradigm_shift_detector)
  ├─ 输入: Claim + ClaimRelation + Phase + Tension
  ├─ 严格标准: 领域共识改变才算范式转移，技术演进不算
  ├─ 跨 Phase 检测: 关注 Phase 边界处的共识变化
  ├─ LLM 识别 2-3 个根本性转变
  └─ 输出: list[ParadigmShift]

Phase G: 叙事生成 (narrative_builder) ← V8.1 idea-centric + GPT 结构
  ├─ 输入: Claim + ClaimRelation + Phase + ResearchQuestion + ParadigmShift
  ├─ V8.1: Idea-centric 叙事 — 论文退居脚注，思想成主角
  │   section title = Phase 名称（问题驱动，如 "How to Build a 3D View from 2D Images? (2020-2022)"）
  ├─ TIME DISCIPLINE（关键约束，同 V8）:
  │   ├─ 每个 Phase 只能引用该时间范围内或更早的论文
  │   ├─ section_claims/section_relations 按 phase_max_year 过滤
  │   ├─ Mermaid 图: 只保留双方都在 phase 内的 relation
  │   ├─ Direction: year-filtered + phase-specific keyword matching
  │   └─ 排除后续 Phase 的 key_papers（boundary guard）
  ├─ Phase 内叙事结构（idea-centric）:
  │   1. CORE QUESTION — 本 Phase 的核心问题，粗体独立句
  │   2. SETUP (1-2 句) — 简要承接上一阶段遗留问题
  │   3. CORE DISCOVERY (2-4 句) — 这个阶段的 IDEA 是什么（非论文列表）
  │      - 先讲核心发现/转折，再举例论文（论文是"比如"而非"然后"）
  │      - 论文名粗体，括号年份，一句贡献，不展开全文
  │      - 格式: "X showed Y (Paper, YEAR). Z later demonstrated W (Paper, YEAR)."
  │   4. TURNING POINT (1 句) — 什么证据推动方向
  │   5. TAKEAWAY — 一句粗体金句
  │   6. UNSOLVED — 遗留问题（→ 下一阶段 hook）
  ├─ 格式规则:
  │   ├─ 段落: 2-4 句 MAX，每个逻辑转折处分段
  │   ├─ 粗体: 核心概念、论文名（首次出现）、关键数字
  │   ├─ 不逐篇展开论文贡献/局限 bullet points — 那是 paper summary 的写法
  │   ├─ 章节标题: 问题驱动命名
  │   └─ 避免重复同一证据（Narrative 讲过的数字 Direction 不再重复）
  ├─ Direction 精简为 3 行:
  │   1 行结论 + 1 行置信度 + 1 行 Why（论文名列表，不展开）
  ├─ Overview: 1 短段，不拆 3 段
  ├─ Synthesis 输出 open_questions (0-3) + reading_list
  └─ 输出: ResearchNarrative

Phase H: 导出
  ├─ Markdown（英文）→ LLM 中译
  └─ JSON 结构化数据（供程序化消费）
```

**已移除的 Phase：**
- ~~Phase D: 分支发现 (branch_discovery)~~ — Branch 已从数据模型中移除。Tension 维度分组替代了论文聚类分组。

### 4.3 模块设计

#### 4.3.1 paper_retriever.py

**目标**：构建候选论文池，高召回但噪声可控。

**两阶段检索**：
1. Citation Expansion（backward + forward），目标 50-100 篇，高精度
2. Problem-based Semantic Search，用提取到的 problem statement 做 query，补充遗漏

最终候选池 80-120 篇。

**数据源**：Semantic Scholar API（引用关系 + 元数据）、arXiv API（PDF）

#### 4.3.2 claim_extractor.py

**目标**：从论文中提取可证伪的 Claim，而非方法描述。

**输入**：
- Paper 元数据 + abstract
- StructuredUnderstanding（如有 full-text，优先使用 problem/motivation/architecture_overview/results）

**Prompt 核心设计**：
```
你正在从一篇 {domain_name} 领域论文中提取其核心研究主张（Claim）。

Claim 不是"论文做了什么"（那是 Solution），而是"论文断言什么是对的/更好的"。

区分标准：
- Solution: "We use V-JEPA for pre-training" → 不要提取
- Claim: "V-JEPA pretraining outperforms perception-heavy pipelines" → 提取这个

正确 Claim 示例：
- "Dense BEV representation is unnecessary; sparse queries preserve performance at lower cost"
- "Joint embedding prediction aligns better with planning than pixel reconstruction"

错误 Claim 示例（这些是 Solution，不是 Claim）：
- "We propose a three-module framework"
- "The model uses deformable attention"
- "Training is done in two stages"

对每篇论文，提取 4-6 个核心 Claim。每个 Claim 必须是可证伪的判断。
```

**输出**：`list[Claim]`

#### 4.3.3 claim_relation_builder.py

**目标**：构建 Claim 之间的 pairwise 关系，用于发现演化路径中的转折点。

**V5.1 单次分类设计**：一次 LLM 调用直接判断关系类型，不需要单独的 lineage gatekeeper。

关系类型：
- `attack` — B 直接否定 A 的前提假设
- `replace` — B 认为 A 的整个范式已过时（同一任务，根本不同的方法哲学）
- `improve` — B 接受 A 的核心方法但修复了具体限制
- `extend` — B 将 A 的洞察应用到新领域
- `support` — B 从不同角度为 A 提供了独立证据
- `parallel` — B 服务于不同下游任务（detection vs planning），两篇论文不交互

**关键判断标准**：基于 downstream task 而非论文自身的问题描述。
- 同一任务（都做 detection）但方法根本不同（dense vs sparse）→ `replace`
- 同一任务，方法改进 → `improve`
- 不同任务（detection vs planning）→ `parallel`

**边构建策略（V5.2）**：
- 先按 downstream task 分组论文（`_guess_downstream_task()` 启发式：detection/planning/tracking/prediction/mapping）
- 组内按时序相邻建边（LLM 分类，O(N)）
- 跨 task 的相邻论文对直接标记 parallel（不调 LLM，避免伪因果关系）
- 这限制了跨间隔的 replace/attack 检测，但 paradigm shift detector 在宏观层面补足

**输出**：`list[ClaimRelation]`

#### 4.3.4 tension_detector.py
**目标**：两阶段检测——先识别所有细粒度矛盾，再按时间+主题合并为 Phase。

**设计理由**：V6/V7 限制 Tension 数量（3-8 个）导致信息丢失。V8 不限制 Stage 1 的数量，让 LLM 自由识别所有矛盾。Stage 2 的合并保证了叙事章节数可控。

**Stage 1: detect_all_tensions()**

输入：`list[Claim]` + `list[ClaimRelation]`

Prompt 核心：
```
识别该领域演化过程中的所有研究张力，不限制数量。一个 Tension 是：
- 一种矛盾（accuracy vs efficiency）
- 一个被挑战的前提假设（"explicit depth is necessary" → "no, attention can learn it"）
- 一个形成了两个对立阵营的方法论选择（dense vs sparse）

对每个 Tension:
- 短标签 + 详细描述（矛盾的本质）
- 引入者（哪些论文首次暴露了此矛盾）
- 解决者（哪些论文推进或倾向了某个方向）
- 时间范围（矛盾活跃的时间段，如 "2021-2023"）
- 状态（direction_clear | direction_forming | open）
- 维度（representation | geometry | system | evaluation）
- 适用域（domain_scope）

输出所有能识别到的 Tension，通常 8-12 个。不要合并，保留细粒度。
```

**Stage 2: merge_tensions_into_phases()**

输入：`list[Tension]`（Stage 1 的全部输出）+ `list[Claim]`（提供论文元数据）

Prompt 核心：
```
将以下研究张力按时间和主题合并为 2-4 个 Phase（技术发展阶段）。

合并原则：
1. 时间重叠的 Tension 优先合并（同一时期的矛盾属于同一 Phase）
2. 主题相似的 Tension 优先合并（如"depth supervision"和"depth quality"→同一 Phase）
3. 每个 Phase 必须有一个 CORE CONTRADICTION（最核心的那个矛盾）
4. Phase 间必须有因果链：Phase N 的 unresolved_problem 自然引出 Phase N+1

对每个 Phase：
- name: 阶段名称，问题驱动（如 "How to Build a 3D View from 2D Images?"），不用静态标签
- time_range: 时间范围（如 "2020-2022"）
- core_contradiction: 核心矛盾（1 句话，<100 字符）
- key_papers: 关键论文列表（3-5 篇，不足则与相邻 Phase 合并）
- core_debate: 核心辩论（这个阶段领域在争论什么）
- unresolved_problem: 未解决的问题（1 句话，→ 成为下一阶段的 motivation）
- tensions: 属于此 Phase 的 Tension（从输入中选择）
```

**输出**：`list[Tension]` + `list[Phase]`
#### 4.3.4.5 research_question_detector.py ← V6 新增
**目标**：从论文池中提取核心研究问题——V8 中退居 Phase 内容，不再是章节标题。

**V8 角色变化**：RQ 仍然是知识骨架（领域争论的问题比具体矛盾更稳定），但不再驱动章节结构。在 V8 中，RQ 是 Phase 内的"情节"——每个 Phase 可能涉及 1-2 个核心 RQ，不同 Phase 可以涉及同一 RQ 的不同侧面。

**与 tension_detector 的关系**（V8）：
- Stage 1 tension detector 先运行，识别所有细粒度矛盾
- Stage 2 phase merger 将 tensions 聚类为 Phase
- RQ detector 在 Phase 确定后运行，为每个 Phase 补充"这个阶段领域在争论什么问题"
- RQ 嵌套 Tension 和 Direction（V7 结构保留），但不决定章节标题

**Prompt 核心**（与 V6/V7 基本一致，增加 Phase 上下文）：
```
你是一位教授，正在设计一门关于 {field_name} 技术演化的研究生课程。

领域已被分为以下阶段（Phase）：
{phases_text}

对每个 Phase，识别该阶段领域争论的 1-2 个核心研究问题。

识别的是 QUESTION，不是 CONTRADICTION：
  ✓ "Is explicit depth supervision necessary?"（一个问题，各方有不同回答）
  ✗ "Dense vs Sparse representation"（一个矛盾，没有提出问题）

层级（level）：
  - "field": 定义整个研究领域
  - "paradigm": 领域内的重大方法论争论
  - "engineering": 工程约束（不会成为独立叙事内容）

对每个问题，识别：
  - 问题文本（以 ? 结尾的完整句子）
  - 简短名称（3-5 个单词）
  - 1-2 句背景
  - 层级和状态
  - 哪些论文持什么立场（position + evidence）
  - 哪些论文首次提出了这个问题
  - 从属的 Tension（1-3 个）
  - Direction（证据指向的结论）
```

**输出**：`list[ResearchQuestion]`
#### 4.3.5 paradigm_shift_detector.py

**目标**：检测领域共识的根本性改变。

**与 Tension 的关系**：Tension 是"当前存在什么矛盾"，ParadigmShift 是"矛盾是否彻底改变了领域共识"。不是所有 Tension 都导致 ParadigmShift。

**严格区分范式转移 vs 技术演进**：
- 技术演进（不算范式转移）：single-frame→temporal fusion、CNN→Transformer backbone、O(T)→O(1) fusion
- 范式转移：dense BEV→sparse queries、modular pipeline→end-to-end planning

**Litmus Test**：旧范式下的研究者会认为新范式"难以置信"或"不可理喻"→ 范式转移。觉得是"自然改进"→ 技术演进。

**输出**：全场最多 3 个 `ParadigmShift`，每个标记 dimension、magnitude、level。

#### 4.3.6 narrative_builder.py ← 产品核心
**目标**：生成领域技术发展故事——Phase 驱动的因果叙事。

**V8 重大变更**：按 Phase（而非 ResearchQuestion）组织 NarrativeSection。章节 = 时间阶段，Phase 间有因果链。RQ 退居 Phase 内的"情节"。

**四段式生成**：

1. **Field Overview**（开场白 — 预告课程将要穿越的阶段）：
   ```
   "Welcome to the seminar. Over the next several weeks, we're going to trace
   how BEV perception evolved through three distinct eras. First, the Dense era
   (2020-2022), where the field struggled with a fundamental question: how do we
   project 2D camera features into 3D space? Then, the Sparse revolution
   (2022-2023), where researchers realized that most of that dense grid was empty.
   Finally, the End-to-End era (2023-2024), where the question shifted from
   'how do we perceive?' to 'how do we drive?'"
   ```
   基于所有 Phase + ParadigmShift，预告课程将要穿越的阶段。
   展示阶段之间的因果链（Phase 1 的遗留 → Phase 2 的核心矛盾）。

2. **Per-Phase Narrative**（每个 Phase 独立叙事）：
   ```
   V8: 按 Phase（时间阶段）生成 NarrativeSection，而非按 RQ。

   教授讲课结构（保留 V5 的叙事弧线）：
   "Last week we ended with a problem: [Phase N-1 的 unresolved_problem]"
   1. WHY THIS PHASE HAPPENED — 承接上一阶段的遗留问题，设置动机
   2. THE CORE DEBATE — 这个阶段的核心矛盾
      - 各方论文的立场和证据
      - Tension 的戏剧化呈现（V5 弧线: 问题 → Paper A → Tension → Paper B）
      - "Two competing answers emerged..."
   3. THE TURNING POINT — 什么论文/证据推动了阶段转换
      - 关键 breakthrough paper 的角色
      - 引用具体指标作为证据
   4. UNRESOLVED — 这个阶段留下了什么问题
      - 自然引出下一阶段（"but this created a new problem..."）

   论文是回答阶段矛盾的证人，不是故事的主角。
   RQ 作为"情节"嵌入：每个 Phase 内可能涉及 1-2 个 RQ 的辩论。
   ```

3. **Synthesis**（结课总结）：
   ```
   分两部分:
   - Evidence-Backed Conclusions: 2-3 个由论文直接支撑的结论
   - Speculative Directions: 1-2 个推测方向，明确标注为推测
   ```

**叙事风格指南**（在 system prompt 中编码）：
- "Last week we ended with a problem: ..." — 每章用遗留问题开场
- "Two competing answers emerged..." — 构建辩论结构
- "The evidence increasingly favored..." — 证据驱动的结论（V7 Direction）
- "But this created a new problem: ..." — Phase 因果链 hook
- 使用具体指标作为证据："SparseBEV achieved 67.5 NDS, surpassing BEVDepth's 60.9"
- 校准语言：基于有限论文集合，使用 hedged 语言
- Phase 名称应该 memorable："Dense Era" → "Sparse Era" → "End-to-End Era"

**输出**：`ResearchNarrative`（sections 按 Phase 组织，而非 RQ）
#### 4.3.7 markdown_report_builder.py
**目标**：生成最终 Narrative 报告。

**输出结构**（V8.1: GPT 建议结构）：
```markdown
# [领域名称] — 技术发展叙事

## 1. Field Overview
（1 短段，结构清晰、换行合理、小标题）

## 2. Major Paradigm Shifts（0-5 条）
（每条一句结论，无 Mermaid，无分维度展开）
- Dense BEV Grid → Sparse Queries (2022-2024)
- Explicit Depth Supervision → Learned Geometry via Attention (2022)
- Modular Pipeline → End-to-End Planning-Oriented System (2022-2024)

## 3. Phase Evolution

### 3.1 Phase 1: "How to Build a 3D View from 2D Images?" (2020-2022)
> **核心矛盾**: ... | **核心辩论**: ...

（idea-centric 叙事: 核心发现 → 举例 → 转折 → Takeaway → Unsolved）

#### 思想演化图（Mermaid）

#### 关键论文与核心主张（Claims 表格）

### 3.2 Phase 2: ...
...

## 4. Open Questions（0-3 条）
- How to handle long-tail scenarios in sparse end-to-end systems?
- Is there a hybrid representation that optimally balances dense/sparse?

## 5. Reading List
（按 Phase 分组的论文列表，含标题、年份、一句贡献）
```
**已移除的内容（V8.1）**:
- Direction 独立 block — 证据已融入 Narrative
- 核心研究张力 表格 — 过于详细，Tension 信息已在 Phase 叙事中体现
- Evidence-Backed/Speculative 分节 — 合并为 Open Questions
- 演化路径（边+解释）— 重复 Mermaid 图的信息
- 范式演化 Mermaid 全景图 + 按维度展开 — 压缩为一句结论列表
---

## 5. V3 与 V4 的交互

### 5.1 V3 调用 V4

当 V3 完成种子论文分析后，用户可以请求查看该论文在领域中的位置：

```python
# run_v3.py 中
if user_wants_evolution:
    narrative = build_narrative(
        seed_paper=seed_paper,
        seed_structured=seed_paper.structured,
        paper_pool=candidate_papers,  # 来自 Phase 2-4
    )
```

V3 输出中增加 Section 6 "在领域演化中的位置"，包含：
- 该论文所属分支
- 该分支内的前序/后续 Claim
- "查看完整领域演化全景"的入口

### 5.2 V4 调用 V3

V4 叙事中每个关键论文节点可触发 V3 深度分析：

```python
# narrative 输出中
for paper in section.papers:
    report.add_link(
        text=f"→ 深度解析 {paper.title}",
        action=f"v3.analyze({paper.arxiv_id})"
    )
```

### 5.3 接口约定

两个子系统通过以下接口交互：

```python
# V3 提供：
def analyze_paper_structure(paper: Paper, llm_client, profile, domain_name) -> StructuredUnderstanding

# V4 提供：
def build_narrative(seed_paper, seed_structured, paper_pool, llm_client) -> ResearchNarrative
def locate_paper_in_narrative(paper_id: str, narrative: ResearchNarrative) -> PaperPosition
```

---

## 6. 配置

### 6.1 V4 新增环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `V4_ENABLED` | `0` | V4 Narrative Engine 是否启用 |
| `V4_CANDIDATE_POOL_SIZE` | `100` | 候选论文池目标大小 |
| `V4_CLAIM_EXTRACTION_ENABLED` | `1` | Claim 提取开关 |
| `V4_TENSION_DETECTION_ENABLED` | `1` | Tension 检测开关 |
| `V4_NARRATIVE_ENABLED` | `1` | 叙事生成开关 |
| `V4_PARADIGM_DETECTION_ENABLED` | `0` | 范式检测开关（实验性功能） |

### 6.2 现有 V3 配置（保持不变）

参见 CLAUDE.md 中 V3 Configuration 节。

---

## 7. 实验计划（MVP）

### 7.1 目标

验证核心假设：Phase（Tension 时间聚类）驱动的叙事比 RQ 驱动的叙事更易读、更能让读者感知技术趋势。

### 7.2 当前状态（已完成）

**V7 已验证（RQ 嵌套 Tension + Direction）**：
- `claim_extractor.py` — Claim 提取（4-6 claims/paper，标记 claim_level）
- `claim_relation_builder.py` — ClaimRelation 单次分类（attack/replace/improve/extend/support/parallel）
- `research_question_detector.py` — RQ 检测（嵌套 Tension + Direction，单次 LLM 调用）
- `tension_detector.py` — Tension 检测（单阶段，从属于 RQ）
- `paradigm_shift_detector.py` — ParadigmShift 检测（≤3 shifts）
- `narrative_builder.py` — 叙事生成（V7: RQ-driven + Tension escalation + Direction）
- V7 输出: `output/v7_mvp_001/bev_evolution.md`

**V7 质量评估**：
- RQ 重叠减少（4 paradigm RQs vs V6 的 7）
- 结构化 Direction 提供证据支撑
- Tension escalation 叙事有戏剧弧线
- **核心问题**：RQ 做章节标题导致"读起来非常不顺畅"，每章覆盖相同 2020-2024 时间段，读者无法感知技术"线性往前推进"

### 7.3 V8 已完成

**V8 核心实现**：
1. ✅ Phase 数据模型（paper.py）— name, time_range, core_contradiction, key_papers, core_debate, unresolved_problem, tensions
2. ✅ 两阶段 tension_detector — `detect_all_tensions()` (8-12 细粒度) → `merge_tensions_into_phases()` (2-4 Phase)
3. ✅ Phase-based narrative_builder — 因果链叙事，TIME DISCIPLINE 约束
4. ✅ 统一 Phase 章节渲染 — 叙事 + Direction + Mermaid + Claims 表 + 遗留问题
5. ✅ 问题驱动 Phase 命名 — "How/Can/Should...?" 格式
6. ✅ 内容均衡约束 — 每 Phase 3-5 篇论文，不足则合并

**V8.1 优化（对比 DeepSeek，17 项修复）**：
7. ✅ TIME DISCIPLINE — section_claims/relations/Direction 按 phase year 过滤，禁止未来论文泄漏
8. ✅ Phase 特定 Direction — `_aggregate_direction()` year-filtered，无 fallback
9. ✅ DeepSeek 风格叙事格式 — 短段落(2-4句)、粗体关键词、分点论文分析、问题开篇、金句收尾
10. ✅ 统一章节结构 — 移除独立的"演化细节"section，Mermaid+Claims 嵌入 Phase 章节内
11. ✅ 全景拆分为 3 短段、表格长内容不再硬截断

**V8 输出质量评估**：
- Phase 时间纯度: Phase 1 (2020-2022) 不再引用 SparseBEV/SparseDrive
- Direction 阶段特定: Phase 1 无 supporting papers（正确，稀疏方案尚未出现），置信度 Medium（非 High）
- Phase 间因果链: "Dense grids waste computation → Can sparse match? → What representation for end-to-end?"
- 与 DeepSeek 4 讲结构一致: 按时间阶段组织，线性推进，不跳回

### 7.4 待完成（远期）

**中期**：
1. 实现 `paper_retriever.py` — 两阶段自动检索
2. 端到端测试（从种子论文到完整叙事）

**远期**：
3. V3/V4 集成（V3 输出增加"在领域演化中的位置"section）
4. 跨领域叙事（如 "CV 对 Autonomous Driving 的影响"）
---

## 8. 迁移路径

### Phase A：V4 核心模块（不依赖自动检索）
1. 实现 `claim_extractor.py` — Claim 提取（含 claim_level）
2. 实现 `claim_relation_builder.py` — ClaimRelation 单次分类
3. 实现 `tension_detector.py` — 两阶段：detect_all + merge_into_phases
4. 实现 `research_question_detector.py` — RQ 检测（V7 嵌套结构，V8 作为 Phase 内容）
5. 实现 `paradigm_shift_detector.py` — 范式转移检测
6. 实现 `narrative_builder.py` — Phase-based 叙事生成（因果链）
7. 手工准备 BEV 12 篇论文数据，运行验证

### Phase B：V4 自动化
7. 实现 `paper_retriever.py` — 两阶段自动检索
8. 端到端测试（从种子论文到完整叙事）

### Phase C：V3/V4 集成
9. V3 输出增加"在领域演化中的位置"section
10. V4 输出增加"深度解析此论文"入口
11. 共享数据模型稳定化（ClaimRelation, Tension, ParadigmShift, NarrativeSection dataclass）

### Phase D：增强功能
12. 跨领域 Narrative（如 "Computer Vision 对 Autonomous Driving 的影响"）
13. 交互式叙事探索（可点击的演化图节点 → 展开论文详情）

---

## 9. 文件布局（目标状态）

```
src/
  # V3 — 单篇论文分析
  run_v3.py                    # V3 入口
  paper.py                     # 数据模型：Paper, StructuredUnderstanding, Claim, ClaimRelation, Tension, Phase, ParadigmShift, NarrativeSection, ResearchNarrative
  paper_resolver.py            # arXiv/PDF 解析
  text_extractor.py            # PyMuPDF 文本提取
  structured_analyzer.py       # 结构化分析引擎（Schema → Prompt → LLM → dict）
  citation_miner.py            # Semantic Scholar 引用挖掘
  route_analyzer.py            # 技术路线归纳 + 对比分析（V3 旧版，将被 V4 替代）
  markdown_exporter_v3.py      # V3 Markdown 导出
  llm_analyzer.py              # LLM 工具函数（JSON 解析、client builder）
  config.py                    # 配置

  # V4 — 领域叙事引擎
  run_v4.py                       # V4 入口
  paper_retriever.py              # 两阶段论文检索（待实现）
  claim_extractor.py              # Claim 提取（核心）
  claim_relation_builder.py       # ClaimRelation 单次分类
  research_question_detector.py   # RQ 检测 — V8 Phase 内"情节"，不再是章节标题
  tension_detector.py             # 两阶段: detect_all_tensions → merge_tensions_into_phases
  paradigm_shift_detector.py      # 范式转移检测（跨 Phase，≤3 个）
  narrative_builder.py            # 叙事生成（V8: Phase-based + DeepSeek 格式 + TIME DISCIPLINE）
  markdown_exporter_v4.py         # V4 Markdown 导出

  # 领域 Schema
  domains/
    __init__.py                # 注册中心
    base.py                    # FieldDef, SectionDef, PaperTypeProfile, DomainProfile
    ai_ml.py                   # AI/ML 领域
    biology.py                 # 生物学领域（预设计）
    materials_science.py       # 材料学领域（预设计）

  # 共享
  paper_type_detector.py       # 论文类型检测
  domain_detector.py           # 领域检测

docs/
  design.md                    # [本文档] 系统整体设计
  archive/
    design_v3.md               # V3 原始设计（归档）
    design_schema_driven.md    # Schema 驱动架构（归档）
    design_research_evolution*.md  # V4 迭代讨论（归档）

tests/
  test_claim_extractor.py
  test_claim_relation_builder.py
  test_tension_detector.py
  test_narrative_builder.py
  test_paradigm_shift_detector.py
  ...
```

---

## 10. 设计原则总结

1. **V3 深度，V4 广度** — 两个子系统独立但互补
2. **Claim 是演化建模的原子** — Problem 定义战场，Claim 定义立场
3. **Phase 是叙事章节标题** — 时间阶段 + 因果链是读者最能感知趋势的结构；RQ 是 Phase 内的"情节"，不是章节
4. **两阶段 Tension→Phase** — 先无限制检测所有矛盾（细粒度），再按时间+主题合并为 Phase（章节数可控）
5. **因果链是叙事的脊梁** — Phase N 的 unresolved_problem → Phase N+1 的 core_contradiction；读者跟着时间走，不跳回
6. **叙事是产品的最终价值载体** — 不是图、不是表、不是聚类，是故事；教授讲课是最好的叙事模型
7. **先验证最高风险模块** — MVP 先做 Claim + Tension/Phase + Narrative，再做检索和聚类
8. **Schema 驱动 V3** — 论文类型差异通过 FieldDef/SectionDef 表达，不做 if-else 分支
9. **LLM 不可用时 graceful degrade** — 返回基本结构/跳过，不中断 pipeline
10. **认知建模优先于工程优化** — 正确的抽象（Phase > RQ > Tension > Branch）比正确的代码更重要
11. **TIME DISCIPLINE** — 每个 Phase 只能引用该时间范围内或更早的论文；claims/relations/Direction 全部按 phase_max_year 过滤，Direction 无 fallback（宁可缺失也不泄漏未来论文）
12. **DeepSeek 叙事格式** — 短段落（2-4 句）、粗体关键词/论文名、分点论文分析（贡献/局限）、问题驱动标题、核心问题开篇、金句收尾；读者应能快速扫描而不被大段文字淹没
