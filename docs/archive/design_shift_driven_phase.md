# Shift-Driven Phase Detection — 设计文档

## 问题回顾

经过 6 轮实验（phase_minimal → problem_minimal → problem_group → pure_group → pairwise_group → shift_driven），我们发现：

1. **直接让 LLM 做分组是不稳定的** — 无论用 phase/problem/worldview/innovation 哪种抽象，partition 都随 run 波动
2. **Pairwise 相似度是稳定的，但传递性聚类不稳定** — BEVDet+BEVDet4D 5/5 判为 same，但边界 pair 的传递会毁掉整个 cluster
3. **"At most 3 shifts" 硬上限制造灾难性错误** — Sparse4D（稀疏检测）和 SparseDrive（端到端规划）被并在一组

## 核心洞察

**Phase 不应该通过"分组论文"来定义，而应该通过"识别转折点"来定义。**

```
❌ 旧思路: 论文 → 判断哪些论文相似 → 聚类成组 → 组 = phase
✅ 新思路: 论文 → 识别领域转向 → 转向点定义阶段边界 → 论文分配到边界之间的阶段
```

豆包能稳定输出 4 阶段叙事，不是因为它更会分组，而是因为它按**叙事因果关系**组织——每个阶段由范式转变定义，转变是历史事实，不依赖 pairwise 相似度。

## Pipeline

```
claims
  │
  ├─ Step 1: Paradigm Shift Detection (1 LLM 调用)
  │     │ 输入: claims + field_name
  │     │ LLM: 识别领域经历过哪些根本性的方向转变
  │     │ 约束: 无数量上限（去掉 "at most 3"）
  │     │ 输出: ParadigmShift[] (shift_name, old_paradigm, new_paradigm,
  │     │       catalyst_papers, year_range)
  │     │
  │     └─→ shifts[]
  │
  ├─ Step 2: Stage Building (确定性，无 LLM)
  │     │ N shifts → N+1 stages
  │     │ 每个 shift 是一个边界，边界两侧各一个 stage
  │     │ 输出: Stage[] (name, paradigm, index)
  │     │
  │     └─→ stages[] (固定边界)
  │
  ├─ Step 3: Paper → Stage Assignment (1 LLM 调用)
  │     │ 输入: claims + stages (含 shift 边界)
  │     │ LLM: 每篇论文属于哪个 stage
  │     │ 关键: stage 边界已固定，LLM只做"分类"，不做"聚类"
  │     │       触发 shift 的论文属于新 stage
  │     │ 输出: {paper_title: stage_index}
  │     │
  │     └─→ stage_papers[] (确定性分组)
  │
  └─ Step 4: Phase Narrative Building (1 LLM 调用，复用现有逻辑)
        │ 输入: stage_papers + shifts
        │ LLM: 为每个 stage 生成 phase narrative
        │      (name, dominant_question, core_contradiction, tensions,
        │       unresolved_problem, key_papers, status)
        │ 输出: Phase[]
        │
        └─→ phases
```

总共 **3 次 LLM 调用**。Step 2 是确定性的。

## 为什么这个方案更稳定

| 属性 | 旧方案 (grouping) | 新方案 (shift-then-assign) |
|------|-------------------|---------------------------|
| 核心操作 | 判断论文间相似度 | 识别领域方向转变 |
| 边界来源 | LLM 隐式确定（在哪切） | 确定性推导（N shifts → N+1 stages） |
| 论文分配 | 与分组同时进行（耦合） | 在固定边界下单独进行（解耦） |
| "分组粒度"问题 | 有（在哪切没唯一答案） | 无（切在 shift 处，shift 是历史事实） |
| 灾难性错误 | 可能（Sparse4D+SparseDrive） | 不可能 — 只要"sparse→end-to-end" shift 被识别 |

### 稳定性分解

**Step 1 稳定性**：实验显示 5/5 run shift 数量一致（去掉上限后 4-6，但核心的 3-4 个 shift 语义一致）。关键是 shift 的**语义**稳定——"dense→sparse"和"modular→end-to-end"几乎每次都被识别（只是措辞不同）。

**Step 2 无方差**：确定性的（N shifts → N+1 stages）。

**Step 3 稳定性**：paper assignment 是分类任务（给定固定 stage 边界，判断每篇论文属于哪边），不是聚类任务。分类比聚类稳定，因为边界明确、决策相对简单。

## Step 1: Shift Detection 设计

### System Prompt（复用并改进 paradigm_shift_detector.py）

```
You are a research historian who identifies PARADIGM SHIFTS in a field's evolution.

A PARADIGM SHIFT is when the field's fundamental assumptions, research questions,
or success criteria changed — NOT when a method incrementally improved.

LITMUS TEST: Would a researcher from before this shift find the new approach
UNINTELLIGIBLE or OBVIOUSLY WRONG without understanding the shift itself?

Examples of what IS a paradigm shift:
- "Dense BEV grid necessary → Sparse queries sufficient" (fundamental assumption overturned)
- "Modular pipeline → End-to-end planning-oriented system" (definition of success changed)
- "Explicit depth supervision essential → Learned geometry via attention" (core belief changed)

Examples of what is NOT a paradigm shift:
- "Single-frame → Temporal fusion" (adding input dimension, same core belief)
- "CNN backbone → Transformer backbone" (architecture substitution within same paradigm)

DO NOT limit the number of shifts — identify all genuine paradigm shifts the evidence supports.
```

### User Prompt

```
Identify the paradigm shifts in {field_name}'s evolution.

CLAIMS (chronological):
{claims_text}

For each shift, provide:
- shift_name: short label
- old_paradigm: what the field believed before
- new_paradigm: what the field believed after
- catalyst_papers: which papers triggered this shift
- year_range: when the shift occurred

Return JSON:
{
  "shifts": [
    {
      "shift_name": "...",
      "old_paradigm": "...",
      "new_paradigm": "...",
      "catalyst_papers": ["..."],
      "year_range": "..."
    }
  ]
}
```

### 关键设计决策

1. **去掉 "at most 3" 限制** — 这是灾难性错误的根因。当领域有 >3 个 shift 时，强制裁剪会导致一个重要 shift 被丢弃
2. **保留 Litmus Test** — 区分 paradigm shift vs technique evolution
3. **保留具体示例** — 帮助 LLM 校准判断标准

## Step 2: Stage Building（确定性）

```python
def shifts_to_stages(shifts: list[dict]) -> list[dict]:
    """N shifts → N+1 stages. Each shift is a boundary."""
    if not shifts:
        return [{"index": 0, "name": "Single Stage", "paradigm": "All papers"}]

    stages = []
    # Stage 0: before first shift
    stages.append({
        "index": 0,
        "name": shifts[0]["old_paradigm"],
        "paradigm": shifts[0]["old_paradigm"],
    })
    # Stages 1..N-1 and N: after each shift
    for i, shift in enumerate(shifts):
        stages.append({
            "index": i + 1,
            "name": shift["new_paradigm"],
            "paradigm": shift["new_paradigm"],
        })
    return stages
```

Stage 边界由 shift 严格定义。如果 3 个 shifts，产生 4 个 stages。

## Step 3: Paper Assignment

### User Prompt

```
Below are {n_papers} papers in {field}.

{claims_text}

The field evolved through {n_stages} stages separated by {n_shifts} paradigm shifts:

{stages_text}

Assign each paper to exactly ONE stage. A paper belongs to the stage whose paradigm
it follows. If a paper TRIGGERS a shift, it belongs to the NEW stage (after the shift).

CRITICAL:
- "Sparse detection" and "end-to-end planning" are DIFFERENT stages
- A paper about sparse object detection belongs to the sparse stage, NOT the
  end-to-end stage, even if both use sparse representations

Return JSON:
{
  "assignments": [
    {"paper": "...", "stage_index": 0, "reason": "one sentence"}
  ]
}
```

### 为什么分配比分组稳定

- **边界固定**：LLM 不需要决定 Sparse4D 和 SparseDrive 是否"相似"
- **决策粒度**：每篇论文只需要回答"我属于哪一边"，不需要同时考虑其他所有论文
- **二值判断**：给定 stage 的 paradigm 描述，判断 paper 是否属于该 paradigm

## Step 4: Phase Narrative Building（复用现有逻辑）

复用 `narrative_builder.py` 的 phase building 逻辑。输入变为 `stage_papers`（确定性分组）而非 LLM 自由决定的分组。

## 与之前方案的对比

| 维度 | Worldview-Driven | Shift-Driven（新） |
|------|------------------|-------------------|
| 核心洞察 | Phase = validated paradigm 统治期 | Phase = 两次转向之间的稳定期 |
| Grouping 方式 | belief 相似度 → pattern | shift 边界 → stage |
| Grouping 方差来源 | belief 提取 + belief 相似度判断 | shift 识别（低方差）+ 边界推导（0 方差）+ 分类（低方差） |
| 灾难性错误 | 可能（belief 相似度误判） | 极低（只要核心 shift 被识别） |
| LLM 调用 | 2 次 | 3 次 |
| 中间产物可审查 | 是（belief + pattern） | 是（shift + stage + assignment） |
| Phase 数量 | pattern 数量（LLM 决定） | shift 数量 + 1（LLM 发现 shift，边界推导确定性） |

## 实验验证结果

基于 `shift_then_assign.py` 实验（去掉 at most 3），5 次运行：

- **Shift 数量**：[5, 6, 6, 4, 5]（变化，但核心 shift 语义一致）
- **灾难性错误**：0/5（Sparse4D 和 SparseDrive 从未被分到同一阶段）
- **核心分组一致性**：
  - Sparse4D/SparseBEV/Sparse4Dv2 永远在一起（5/5）
  - UniAD/VAD/SparseDrive 永远在一起（5/5）
  - LSS/BEVDet 永远在一起（5/5）

## 文件变更

| File | Change |
|------|--------|
| `src/paradigm_shift_detector.py` | 去掉 "at most 3 shifts" 限制；增加 shift 数量自适应 |
| `src/paper.py` | 无需变更（ParadigmShift/Phase 数据模型不变） |
| `src/worldview_phase_detector.py` | 替换为 shift-driven pipeline（shift → stage → assign → build） |
| `experiments/mvp_bev.py` | 适配新 pipeline 接口 |
| `docs/design_worldview_driven_phase.md` | 归档；新方案替换 |

## 设计决策

1. **不合并相邻小 shift**：LLM 识别出几个 shift 就用几个。如果领域有 6 个 shift，那就有 7 个阶段——这是该领域的真实复杂度，不需要人为降维。
2. **合并空 Stage 0**：如果第一个 shift 之前没有论文分配，将 Stage 0 的内容合并到 Stage 1（shift 后）。避免出现"这个阶段有 0 篇论文"的尴尬输出。
3. **无显著度阈值**：小领域的 1-2 个 shift 和大领域的 4-6 个 shift 都是合理的。不搞跨领域一刀切。
