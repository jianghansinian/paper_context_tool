# Phase Detection — Design Document

## 1. 问题定义

**输入**：一个领域内 N 篇论文的 raw text（title, year, month, abstract, introduction, results textual conclusions, conclusion）

**输出**：按 Phase 分组的论文 + Phase 间的 transitions + 每篇论文的 key claims + 每个 Phase 内的 tensions

**核心约束**：一次 LLM 调用完成，不做 post-processing（no merge, no split, no re-assignment）。

## 2. 核心理念：Phase 由 Question 定义，不由 Method 定义

从用户提供的框架（2026-06-24 conversation）：

```
Field → Research Tensions → Phase → Core Questions → Paradigms → Papers
```

- **Phase** 回答："社区在解决什么问题？"
- **Paradigm** 回答："社区用什么方法解决？"

同一 Paradigm（Transformer）可以跨多个 Phase。同一 Phase（端到端驾驶）可以包含多个 Paradigm。

## 3. Phase 边界的 5 个信号

用户提供的框架：

| # | 信号 | 含义 |
|---|---|---|
| 1 | 主导问题变化 | 社区在问什么变了？ |
| 2 | 评价指标变化 | "什么是好"变了 |
| 3 | 论文结构变化 | 新 pipeline，新模块组合 |
| 4 | 数据集变化 | 新数据集允许/要求新问题 |
| 5 | 工业界关注点变化 | 招聘、投资方向 |

其中 **信号 1（主导问题）是最强的**。信号 2 是"什么是好"的显式编码。

## 4. Phase Boundary 的判定逻辑

### 4.1 Litmus Test

> Pick a representative paper from Phase N. Would its authors consider the main contribution of a paper from Phase N+1 to be answering a FUNDAMENTALLY DIFFERENT QUESTION than theirs?
>
> YES → phase boundary. NO → same phase, internal tension.

### 4.2 Concurrent ≠ Same Phase

> If two groups of papers from overlapping time periods ask fundamentally different questions, they belong to DIFFERENT phases.
> Phase ordering follows LOGICAL DEPENDENCY — which question had to be addressed first for the other to become meaningful?

### 4.3 三种 Phase 边界类型

| 类型 | 触发条件 | 评价指标变化 | 示例 |
|---|---|---|---|
| **Representation philosophy change** | 社区对"什么算有效解"的核心假设变了 | 可能不变 | Dense grid 必须 → Sparse queries 足够 |
| **Task goal change** | 社区追求的目标本身变了 | **变** | 优化 detection → 优化 planning |
| **Constraint regime change** | 社区的操作约束变了 | 可能变 | 任意算力 → 必须实时 |

三种类型都是合法的 phase boundary。只认 task goal change 会 lump Dense+Sparse（指标相同）。只认 representation change 会 lump Sparse+Planning（都用 sparse queries）。**必须同时支持多种类型。**

### 4.4 已探索的方案

| 方案 | 机制 | 结果 |
|---|---|---|
| Direct shift detection | LLM 从 claims+tensions 直接找 shift | 5 shifts，方向对但过多，paper 分配破碎 |
| Narrative-first | LLM 先写叙事再提取 shift | 3 shifts 稳定，但语义一致性差（同样 3 个，每次不同） |
| Scheme B v1 (output format) | Stage 由 "output format" 定义 | 概念错误：生物学没有"output format" |
| Scheme B v2 (question-based) | Phase 由 dominant question 定义 | 概念正确，phase 数量 2-4，paper 分组 Jaccard=1.00 |
| Scheme B v3 (+ concurrent rule) | v2 + "time overlap not a reason to merge" | 改善到 2-3 phases，3/5 命中 3-phase，paper 分组 Jaccard=1.00 |

## 5. 当前 Prompt 结构（v3）

```
SYSTEM: 研究史学家，Phase ≠ Paradigm
  ├─ HOW TO IDENTIFY PHASE BOUNDARIES (5 signals)
  ├─ LITMUS TEST
  ├─ CONCURRENT BUT DIFFERENT = DIFFERENT PHASES
  ├─ WHAT IS NOT A PHASE BOUNDARY
  └─ WHAT TO OUTPUT

USER: Papers + Task
  ├─ STEP 1: Identify PHASES (5 signals checklist)
  ├─ STEP 2: Describe PHASE TRANSITIONS
  ├─ STEP 3: Extract CLAIMS
  ├─ STEP 4: Identify TENSIONS
  ├─ OUTPUT FORMAT (generic JSON example)
  └─ QUALITY CHECKS
```

## 6. 当前状态与待解决问题

### 6.1 已解决

- [x] 概念正确：Phase 由 question 定义，不由 artifact 定义
- [x] 无领域偏见：prompt 不含 BEV-specific 示例
- [x] 无数量限制：不预设 phase/shift 数量
- [x] Paper 分组稳定：同粒度的 phase 之间 Jaccard = 1.00
- [x] Planning 组始终识别（即使 lump 时也在 narrative 中区分）

### 6.2 待解决：Split vs Lump 判断

**现象**：5 次运行中，3 次拆成 3 phases（正确），2 次 lump 成 2 phases。

**根因**：Dense→Sparse（表示哲学变化）和 Sparse→Planning（任务目标变化）是两种不同类型的 phase boundary。当两者都出现在 2022-2023 年时，model 有时判断它们是"同一批问题的两个方向"而非"两个不同的问题"。

**已尝试**："concurrent but different = different phases" 规则 — 从 1-4 phases 收窄到 2-3，但未完全解决。

### 6.3 待讨论：四种 Phase Boundary 类型

Dense→Sparse 和 Sparse→Planning 的本质区别：

```
Dense→Sparse:   "我们要不要 grid？"（对同一任务用什么表示）
Sparse→Planning: "我们要不要开车？"（任务本身是不是换了）
```

区分这两种变化需要一个不与领域绑定的判定框架。以下是可能需要 review 的方向：

1. **按 primary claim 判断**：paper self-claims 是 solving different problem 还是 better approach for same problem？

2. **按 downstream dependency 判断**：Phase N+1 的工作是否逻辑上依赖 Phase N 的产出？如果 B 的输入是 A 的输出，它们就是不同 phase。

3. **按"可替代性"判断**：如果把论文 X 从 timeline 中删掉，论文 Y 的 motivation 是否仍然成立？如果 Y 必须在 X 存在的前提下才有意义 → 不同 phase。

4. **按 task structure 判断**：如果两批论文的 pipeline/output 拓扑结构不同（不同的模块、不同的上下游），它们大概率在解决不同的问题。

## 7. 文件清单

| 文件 | 说明 |
|---|---|
| `src/one_shot_analyzer.py` | Scheme B 实现：one-shot LLM call + narrative generation + 转换 helpers |
| `src/text_extractor.py` | 新增 `extract_results_section()`, `extract_introduction_section()`, `extract_conclusion_section()`, `assemble_paper_text_for_one_shot()` |
| `tmp/one_shot_test.py` | 单次运行测试 |
| `tmp/one_shot_stability.py` | 5-run 稳定性测试 |
| `CLAUDE.md` | 新增 "No Domain-Specific Assumptions" 原则 |
