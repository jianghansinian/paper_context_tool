# Phase Detection — 设计文档 (v4.5)

## 1. 问题定义

**输入**：一个领域内 N 篇论文的 raw text（title, year, month, abstract, introduction, results textual conclusions, conclusion）

**输出**：按 Phase 分组的论文 + Phase 间的 transitions + 每篇论文的 key claims + 每个 Phase 内的 tensions

**核心约束**：不做代码级后处理（no merge, no split, no re-assignment）。

## 2. 核心理念：Stage 由论文的构建机制和目的定义

**Stage 不是由我们预设的维度定义的。** prompt 不告诉 LLM "什么算 stage boundary"。prompt 只给分析框架——三个轴，让 LLM 从论文中读出什么变了。

### 2.1 三轴分析框架

所有研究领域共享这三个分析轴。LLM 用它来读论文，而不是用它来套答案：

| 轴 | 作用 | LLM 要回答的问题 |
|----|------|-----------------|
| **Axis 1: 构建机制** (主分类) | 决定 stage 归属 | 论文用什么机制构建核心技术产物？相同机制 = 同 stage；机制本质不同 = 新 stage |
| **Axis 2: 机制改进** (组内排序) | 不决定边界 | 论文如何在同机制内改进/扩展？仅影响 stage 内论文的先后顺序 |
| **Axis 3: 构建 vs 应用** (任务范围) | 触发边界 | 论文在 BUILD 这个技术产物，还是把它当组件 APPLY 到一个本质不同的更大系统？ |

**这三个轴不是 checklist。** Axis 1 是主分类依据，Axis 3 是补充（区分 build vs apply）。Axis 2 只决定排序，不产生边界。LLM 从论文中读出构建机制和目的，不是从 prompt 中套答案。

### 2.2 核心约束

- **不预设 stage 数量**：数据决定
- **不预设构建机制类型**：论文自己揭示
- **不预设领域知识**：prompt 不含任何领域特定的术语或示例
- **不做代码级后处理**：no merge, no split, no re-assignment

### 2.3 Stage size

- 通常每个 stage 至少 2 篇论文，但 genuinely foundational 的单篇论文可以单独成段
- 不要把概念上独立的 singleton 强行合并到相邻 stage

### 2.4 Stage vs Paradigm

- **Stage** 由构建机制或目的的本质变化定义
- **Paradigm** 是同一 stage 内不同的具体实现路径
- 同一 Paradigm 可以跨 Stage，同一 Stage 可以含多个 Paradigm

## 3. Stage Boundary 判定逻辑

### 3.1 分析方法（给 LLM 的，不是给 LLM 套的）

LLM 读完全部论文后，用两个 test 判断边界：

**Construction Test**（主测试）：
> B 组论文是否使用了与 A 组**本质不同**的构建机制来构建核心技术产物？
> A 组研究者看到 B 组的机制，会说"这不是我们方法的改进——这是完全不同的做法"吗？

**Purpose Test**（补充测试）：
> A 组是否在 **BUILD** 这个技术产物，而 B 组在把它作为组件 **APPLY** 到一个本质不同的更大系统？
> A 组作者会说"我们造了工具；他们在用我们的工具解决一个我们没打算解决的问题"吗？

**任一 test 回答 YES → stage boundary。**

### 3.2 Litmus Test

> Pick a representative paper from Stage N. Read it carefully. Now read a paper from Stage N+1.
>
> 1. Do they use FUNDAMENTALLY DIFFERENT mechanisms to construct the core artifact?
> 2. Is the earlier paper BUILDING the artifact while the later paper APPLIES it for a different purpose?
>
> If YES to either → stage boundary.
> If NO to both but different specific methods → same stage, competing paradigm.

### 3.3 Concurrent ≠ Same Stage

> 时间重叠不是合并 stage 的理由。如果两组论文使用不同的构建机制或有不同的目的，它们属于不同 stage。Stage 排序按逻辑依赖关系，不按时间。

### 3.4 What is NOT a Stage Boundary

- 同一构建机制下的改进或组件增强 → stage 内的 progress
- 同一机制下的新方法 → stage 内的 competing paradigm
- 同一指标的性能提升 → stage 内的 progress

区分 "构建机制变化" 和 "机制改进"：
- 机制变化 = 用什么方法构建核心技术产物发生了本质改变。这是 stage boundary。
- 机制改进 = 同一构建方法下的优化、增加数据源、改进训练。这是 stage 内的 progress。

## 5. 方案 B：True One-Shot（主路径）

### 6.1 Pipeline

```
all_papers_text → analyze_field_one_shot() → {
    phases[],
    shifts[],
    claims[],
    tensions[]
}
    │
    └─→ generate_evolution_md() → 结构化叙事文档
          ├─ 1 次 Field Overview
          ├─ 每 phase 1 次叙事（共 N 次）
          └─ 1 次 Synthesis
```

总计 **N + 3 次 LLM 调用**（分析 1 次 + 叙事 N+2 次，N = phase 数）。结构化叙事规格见 §6.6。

### 6.2 输入

每篇论文：title + year-month + abstract + introduction + results textual conclusions (tables removed) + conclusion。

### 6.3 任务步骤

```
STEP 1 — Identify STAGES by grouping papers that share the same construction mechanism
        for the core technical artifact.
        - For each paper: what is the core artifact, HOW is it constructed, and
          is the purpose BUILD or APPLY?
        - Group by construction mechanism (Axis 1). Split by purpose if needed (Axis 3).
        - Apply Construction Test and Purpose Test at each candidate boundary.
STEP 2 — Describe STAGE TRANSITIONS (shifts): what mechanism/purpose changed, catalyst papers,
        trigger
STEP 3 — Extract CLAIMS: falsifiable judgments with evidence
STEP 4 — Identify TENSIONS: debates between competing refinements of the SAME construction
        mechanism within a stage
```

### 6.4 约束

- 每篇论文恰好属于一个 stage
- 通常每个 stage 至少 2 篇论文，但 genuinely foundational 的单篇可以单独成段
- Stage 数量 = Shift 数量 + 1
- 不预设 stage/shift 的具体数量

### 6.5 输出格式

```json
{
  "phases": [
    {
      "index": 0,
      "name": "Phase name",
      "dominant_question": "The core question the community was asking",
      "core_tension": "The central contradiction driving work in this phase",
      "papers": ["Paper Title 1", "Paper Title 2"],
      "year_range": "YYYY-MM—YYYY-MM"
    }
  ],
  "shifts": [
    {
      "shift_name": "Short label",
      "from_phase": 0,
      "to_phase": 1,
      "old_question": "What the community was asking before",
      "new_question": "What the community started asking instead",
      "catalyst_papers": ["Paper That Triggered This Shift"],
      "trigger": "What caused this transition"
    }
  ],
  "claims": [
    {
      "paper": "Paper Title",
      "statement": "A falsifiable judgment about what is true",
      "evidence": "Supporting evidence",
      "claim_level": "paradigm | methodological | engineering"
    }
  ],
  "tensions": [
    {
      "phase": 0,
      "name": "Short label",
      "description": "What is the disagreement about?",
      "positions": [
        {"paper": "Paper A", "position": "Position", "evidence": "..."},
        {"paper": "Paper B", "position": "Opposing position", "evidence": "..."}
      ]
    }
  ]
}
```

### 6.6 叙事文档生成（结构化，v4.5）

one-shot 结果由 `src/one_shot_narrative.py::generate_evolution_md()` 分步生成结构化文档（实现自 2026-07 实验脚本 `experiments/generate_evolution_md.py` 转正）：

```
one_shot 结果（phases/shifts/claims/tensions）
  ├─ Step 1: Field Overview（1 次 LLM 调用）→ 领域全景段落
  ├─ Step 2: Per-Phase Narrative（每 phase 1 次 LLM 调用，共 N 次）
  │     输入: 该 phase 的 papers/claims/tensions + 上一 phase 的遗留问题
  │     输出: {narrative（含固定子标题）, mermaid, key_insight, unresolved}
  ├─ Step 3: Synthesis（1 次 LLM 调用）→ 综合段 + 开放问题 + 推荐阅读
  └─ Step 4: 代码拼装（_assemble_markdown）→ 6 节 markdown
```

**文档结构（6 节）**：

1. **领域全景** — overview 段落 + phase 总览表（Phase/Time/Key Papers）
2. **范式转移** — shifts 列表
3. **阶段演化** — 每 phase 一小节：核心矛盾 blockquote（+ 承接上一阶段）→ 叙事正文（子标题：背景/核心发现/转折点/关键认知/遗留问题）→ mermaid 关系图 → 关键论文与核心主张表（论文/年份/主张/证据）
4. **开放问题** — 领域未解决的问题
5. **推荐阅读** — 按 phase 分组的最重要论文
6. **领域趋势与展望** — 综合段落

**每 phase 叙事 prompt 规则**（完整 prompt 见 `src/one_shot_narrative.py`）：

- Idea-centric：先写洞察，论文是证据不是主角
- 时间纪律：只引用本 phase 时间范围内或更早的论文，不提未来论文
- 固定子标题必须逐字输出（背景/核心发现/转折点/关键认知/遗留问题）
- mermaid：`flowchart LR`，边类型 改进/并行/替代
- 数字约束遵循附录 A.1（不硬编码数量/字数）

**历史**：v4.4 的散文叙事（`generate_narrative_from_one_shot()`，1 次 LLM 调用产出 600-1200 词无结构散文）未经结构评审且与 V8 时代文档形态脱节，被 v4.5 结构化文档取代（2026-08-17）。

## 6. 方案 A：多步提取（扩展路径）

当论文数 > 50 或 one-shot 输出质量不足时，退回方案 A。

### 7.1 Pipeline

```
papers
  │
  ├─ Step 0: Claim + Paper Profile 提取（per paper, N 次 LLM 调用）
  │     输入: 每篇论文的 abstract + intro + conclusion
  │     输出: claims[] + paper_profile{core_question, core_belief, responds_to, key_result}
  │
  ├─ Step 0.5: Relation + Tension 检测（1-2 次 LLM 调用）
  │     输入: paper_profiles[]
  │     输出: relations[] + tensions[]
  │
  ├─ Step 1: Phase Detection（1 次 LLM 调用）
  │     输入: paper_profiles + relations + tensions（按时间排序）
  │     任务: 划分 timeline → phases + shifts + paper 分配
  │     输出: phases[] + shifts[]
  │
  └─ 总计: N + 3 次 LLM 调用
```

### 6.2 方案 A vs B

| 维度 | 方案 B（主路径） | 方案 A（扩展路径） |
|------|-----------------|-------------------|
| LLM 调用 | **2** | N + 3 |
| 适用规模 | ≤ 50 篇 | 无上限 |
| Paper Profile | One-shot 内部 | 独立提取，质量可控 |
| 可调试性 | 中（profile 是中间产物） | 高（每步可独立检查） |
| 信息完整性 | 高（LLM 直接读原文） | 中（每步看前序输出） |

## 7. 真值参考（BEV 12 篇论文）

### 7.1 Claude 5-Stage Ground Truth（推荐基准）

Claude（DeepSeek v4）对 12 篇 BEV 论文的分析（仅基于对论文技术方案的知识，无外部检索）：

| Stage | Papers | 构建机制 | Dominant Question |
|-------|--------|---------|-------------------|
| 1. 视图变换奠基 | LSS | 显式深度估计 + 投影 | How to lift 2D perspective to 3D BEV? |
| 2. Dense BEV 工程化 | BEVDet, BEVDepth, BEVDet4D | 显式深度 + 检测 pipeline | How to build a complete detection framework on LSS? |
| 3. 注意力范式转移 | BEVFormer, BEVFormerV2 | 隐式 Cross-Attention（不估计深度） | Can we skip explicit depth and learn BEV via attention? |
| 4. Sparse 范式 | Sparse4D, Sparse4Dv2, SparseBEV | Sparse queries 替代 dense grid | Can sparse queries replace dense BEV grids? |
| 5. 端到端 Planning | UniAD, VAD, SparseDrive | BEV 作为中间表示，输出规划轨迹 | How to make perception serve planning end-to-end? |

5 个 stage，4 个 shift：
- **Shift 1**: 显式深度投影 → 工程化检测框架（机制相同，从 feasibility 到 production）
- **Shift 2**: 显式几何 → 隐式注意力（构建机制本质改变）
- **Shift 3**: Dense grid → Sparse queries（构建机制本质改变）
- **Shift 4**: 感知 → 端到端 Planning（BUILD → APPLY，目的改变）

### 7.2 原始 Ground Truth（v4 时期，偏保守）

| Phase | Papers | Dominant Question |
|-------|--------|-------------------|
| Dense BEV | LSS, BEVDet, BEVDet4D, BEVDepth, BEVFormer, BEVFormerV2 | How to build a reliable 3D view from 2D images? |
| Sparse Detection | Sparse4D, Sparse4Dv2, SparseBEV | Can sparse queries replace dense grids for 3D detection? |
| End-to-End Planning | UniAD, VAD, SparseDrive | How to make perception serve planning safety? |

## 8. 当前状态

### 8.1 v4.4 结果（2026-07-01，3-axis 框架，domain-neutral prompt，5-run）

| 指标 | 结果 |
|------|:----:|
| Stage 数 | **4 (5/5)** |
| Unique partitions | **1** |
| Pairwise Jaccard | **1.00 (全部)** |
| Shift 数 | **3 (5/5)** |
| 覆盖率 | **12/12** |
| 领域特定词汇 | **0** |

**5 跑完全一致的分组**：

| Stage | Papers | 构建机制 |
|-------|--------|---------|
| Depth-based BEV | LSS, BEVDet, BEVDepth | 显式深度估计 + 投影 |
| Attention-based BEV + Temporal | BEVFormer, BEVDet4D, BEVFormer v2 | 隐式注意力 + 时序融合 |
| Sparse Detection | Sparse4D, Sparse4Dv2, SparseBEV | Sparse queries |
| End-to-End Driving | UniAD, VAD, SparseDrive | 感知→规划 (BUILD→APPLY) |

**Shift 名跨跑一致性**：3 个 shift 主题在所有 5 跑中完全一致：
1. Depth-based → Attention-based BEV Construction
2. Dense BEV → Sparse Queries
3. Perception → End-to-End Driving

**与 Claude 5-stage GT 的差异**：
- LSS 未单独成段（与 BEVDet/BEVDepth 使用相同构建机制，合理）
- BEVDet4D 被归入 attention 段而非 depth 段（模型将 temporal fusion 与 attention 机制关联）

### 8.2 演进历史

| Version | 核心框架 | 3-phase 命中 | Stability | 主要问题 |
|---------|---------|:-----------:|:---------:|---------|
| v3 | Phase by question | 3/5 (60%) | 中 | Sparse+E2E 偶尔 lump |
| v4 | Paper Profile embedded | 2/5 (40%) | 低 | Profile 稀释注意力 |
| v4.1 | 无 Profile, phase by question | 3/5 (60%) | 中 | Dense+Sparse 9篇 lump 新失败模式 |
| v4.2 | Research Regime (3 dims) | 1/5 (20%) | 低 | Constraints 维度跨领域差 |
| v4.3 | 4 Universal Dimensions | 0/5 (0%) | 高 (Jaccard 1.00) | 总是 lump 到 2-phase |
| **v4.4** | **3-Axis (construction + purpose)** | N/A (4-stage) | **高 (Jaccard 1.00)** | Sparse/E2E 始终分离; BEVDet4D 归类偏差 |
| **v4.5** | 3-Axis 不变 | 不变 | 不变 | 叙事输出从散文改为结构化 6 节文档（§6.6，N+3 次 LLM 调用） |

### 8.3 已尝试 & 放弃

- **Paper Profile 嵌入 one-shot（v4）**：prompt 变长 + 输出 token 增加稀释了模型注意力 → 已回退
- **Research Regime + 信号 checklist（v4.1-v4.2）**：包含领域特定信号（data regime, constraints）→ 跨领域泛化差
- **4 Universal Dimensions（v4.3）**：太抽象，"shared assumptions" 概念让模型无法区分 Sparse/E2E → 改为 3-axis
- **Phase 术语**：暗示长周期 → 改为 Stage（逻辑步骤）

## 9. 实现路径

1. ~~更新 `one_shot_analyzer.py` prompt：添加 min-2-papers 约束~~（已完成）
2. ~~更新 `_parse_one_shot_response()`：兼容 phases/stages~~（已完成）
3. ~~添加 >50 篇警告~~（已完成）
4. ~~5-run 稳定性测试：v4.1 恢复到 v3 水平~~（已完成）
5. ~~移除 Paper Profile~~（已完成）
6. ~~Phase → Stage 术语转换~~（已完成）
7. ~~3-axis 框架：construction mechanism + purpose~~（已完成）
8. ~~清除所有领域特定示例~~（已完成）
9. ~~v4.4 5-run 验证：Jaccard 1.00, 4 stages 稳定~~（已完成）
10. ~~v4.5 结构化叙事转正：experiments/generate_evolution_md.py → src/one_shot_narrative.py，接入 run_v4~~（已完成，2026-08-17）
11. 待做：BEVDet4D 归类偏差（temporal fusion 被关联到 attention 而非 depth）

## 10. 文件清单

| 文件 | 说明 |
|------|------|
| `src/one_shot_analyzer.py` | Scheme B 实现：one-shot LLM call（§6.2-6.5）+ 转换 helpers |
| `src/one_shot_narrative.py` | Scheme B 结构化叙事（§6.6）：`generate_evolution_md()` |
| `src/text_extractor.py` | Section 提取 + `assemble_paper_text_for_one_shot()` |
| `docs/design_stage_boundary.md` | 本文档 |
| `CLAUDE.md` | Project workflow, commands, testing patterns |

---

## 附录 A：Prompt 设计规则

本项目的所有 LLM prompt 遵循两条硬规则。违反其中任何一条都会导致 prompt 在跨领域、跨规模时静默失效。

### A.1 不做数值约束（No Numeric Constraints）

**绝不**在 prompt 中硬编码"输出 N 个 X"的数字。包括：

- "输出 2-3 个 paradigm shift" / "划分 3-5 个 phase"
- "提取 1-3 条 claim" / "keep 2-3 turning points"

原因：3 个 shift 对 12 篇 4 年的 BEV 论文可能合适，对 50 篇 20 年的领域就是灾难。

**应该用什么？** 语义约束：paradigm shift 改变 WHAT 被解决，method improvement 改变 HOW。用 litmus test 代替数字：研究者会认为这是同一问题吗？

### A.2 不做领域特定假设（No Domain-Specific Assumptions）

**绝不**在 prompt 中编码任何特定学科的知识。包括：

- 领域特定的判断维度："representation 变了"、"output format 变了"
- 领域特定的示例："dense grid → sparse queries"
- 领域特定的术语：这些词来自 CV，对生物学/经济学/NLP 无意义

原因：一个对 CV 有效的 prompt 可能对分子生物学或经济学静默失效——不是因为推理框架错了，而是因为 prompt 词汇不翻译。

**应该用什么？** 普适的研究概念：research question、core tension、method assumption、result。LLM 从论文中推断这个领域的哪些概念重要，prompt 只提供分析方法（如何思考），不提供答案词汇（找什么）。

### A.3 示例是脚手架，不是模板

如果 prompt 中给出示例，必须明确说"这只是一个例子，当前研究的领域可能结构完全不同"。否则 LLM 会把示例当成模板去匹配，而非当成推理方式的说明。

### A.4 自检

写 prompt 时问自己：**"这个 prompt 对数学逻辑学研究者有意义吗？对药物发现？对编程语言？"** 如果答案是否，说明 prompt 包含了不应该存在的领域特定假设。
