# Worldview-Driven Phase 设计（V2）

## 核心洞察：Phase 不可直接检测

之前所有方案（Tension-Driven, Problem-Driven, 当前的 Label-Based）有一个共同的错误：**试图直接"切分" phase**。

```
❌ Tension-Driven:  tension → LLM 聚类 → phase
❌ Problem-Driven:   problem → LLM 聚类 → phase
❌ Label-Based:      claims → LLM 直接分配 phase_label
```

但 Phase 本身不是可检测的实体。它是 **Paradigm Shift 的结果**——社区接受新世界观之后进入的稳定期。Phase 的边界不由时间范围或论文分组决定，而由 worldview 何时切换决定。

```
✅ Worldview-Driven:

claims → worldview extraction (每篇论文的信念)
      → belief grouping → Innovation Patterns (同一技术路线的论文群体)
      → validation detection (哪个 pattern 被社区接受了？)
      → phase = validated paradigm 的统治期
```

这对应了演化模型：

```
Tension (旧 paradigm 的极限)
    ↓
Exploration Wave (多个 worldview 同时尝试)
    ↓
Dominant Innovation Pattern (某个 worldview 展现出优势)
    ↓
Validation (社区接受，大量 follow-up)
    ↓
Paradigm Shift (世界观切换完成)
    ↓
Phase N+1 (新 worldview 的统治期)
```

---

## 核心概念

### Worldview = 论文对"问题该怎么解"的核心信念

不是 problem（解决什么），不是 method（用什么技术），而是**对问题结构的信念**。

```
论文         Problem                 Worldview (核心信念)
─────────────────────────────────────────────────────────
LSS          如何从相机构建BEV？     "BEV 是投影问题，隐式深度分布就够了"
BEVDepth     如何让BEV更准？         "BEV 是几何精度问题，显式深度监督必须"
BEVFormer    如何不用深度做BEV？     "BEV 是特征学习问题，attention 可以学几何"
Sparse4D     如何高效做3D检测？      "世界是 object 的集合，不需要 BEV grid"
UniAD        如何统一自动驾驶？      "规划是 organizing principle，感知应服务规划"
```

同一 worldview 的论文 = 共享同一套关于"问题该怎么解"的信念。
不同 worldview 的论文 = 即使 problem 相同，信念也不同。

**Litmus test**：如果一句话可以被 "We propose a method for X" 替代而不丢失洞察，这是 method description，不是 belief。每个 belief 必须有一个合逻辑的反面信念。

### Innovation Pattern = 同一技术路线的论文群体

共享同一个 worldview 的论文形成一个 Innovation Pattern。例如：
- **Depth-based BEV**：LSS, BEVDet, BEVDepth — 都相信"深度是构建 BEV 的关键"
- **Attention-based BEV**：BEVFormer, BEVFormerV2 — 都相信"几何可以通过 attention 隐式学习"
- **Sparse Detection**：Sparse4D, Sparse4Dv2, SparseBEV — 都相信"世界是 object 的集合"

### Validation = 社区接受

不是所有 Innovation Pattern 都成为 Phase。只有被社区验证接受的 pattern 才定义一个 phase。

Validation 信号：
- **内部 follow-up 链**：pattern 内部有 improve/extend 关系 → 社区持续投入
- **跨 pattern 互动**：后续 pattern 引用/替代此 pattern → 该 pattern 被认真对待
- **规模**：pattern ≥ 2 papers → 不是孤案
- **时间跨度**：pattern 活跃期 ≥ 6 个月 → 不是昙花一现

### Phase = Validated Paradigm 的统治期

Phase 边界 = worldview 切换点。Phase 内部 = 同一 worldview 下的竞争性工作（不同的具体方法，但共享同一个核心信念）。

---

## Pipeline

```
claims
  │
  ├─ Step 1: Worldview Extraction + Grouping (LLM — 1 次调用)
  │     │
  │     │ 输入: claim.statement + claim.problem_addressed (所有论文)
  │     │
  │     │ LLM 做两件事（合并为一次调用，可以自校准）：
  │     │   (a) 提取每篇论文的 core belief
  │     │       约束: falsifiable, 有合逻辑的反面信念, 非 method description
  │     │   (b) 按 shared belief 分组 → Innovation Patterns
  │     │       约束: 每个 pattern ≥ 2 papers, 独特的 belief → "unclustered"
  │     │
  │     │ 输出: PaperBelief[] + InnovationPattern[] + unclustered[]
  │     │
  │     └─→ patterns[] (论文 → belief → pattern)
  │
  ├─ Step 2: Validation Detection (纯 heuristics，无 LLM)
  │     │
  │     │ 对每个 pattern 评估：
  │     │   signal_1: size ≥ 2                 → "size>=2"
  │     │   signal_2: 内部 improve/extend 链     → "internal_followup_chain"
  │     │   signal_3: 跨 pattern replace/attack  → "cross_pattern_engagement"
  │     │   signal_4: 时间跨度 ≥ 6 个月          → "active_Nmonths"
  │     │
  │     │ validated = signals ≥ 2
  │     │
  │     └─→ validated_patterns[] + unvalidated_patterns[]
  │
  └─ Step 3: Phase Building (LLM — 1 次调用)
        │
        │ 输入: validated_patterns[] + unvalidated_patterns[]
        │
        │ LLM: 对每个 validated pattern 构建 Phase:
        │   - name (问句形式)
        │   - dominant_question (核心研究问题)
        │   - core_contradiction (核心矛盾)
        │   - core_debate (领域争论什么)
        │   - internal_tensions (内部竞争立场)
        │   - unresolved_problem (→ 下一 phase 的动机)
        │   - status (direction_clear | direction_forming | open)
        │
        │ 对 unvalidated patterns: 标注为 "旁支探索"
        │
        │ 输出: Phase[] + exploration_waves[]
        │
        └─→ phases
```

总共 **2 次 LLM 调用**。Grouping 和 Validation 不由 LLM 做——grouping 合并到 Step 1（LLM 以 belief 为依据分组），validation 是纯 heuristics。

---

## Step 1 Prompt 设计

### System Prompt

```text
You are a research historian. Your task is to:
1. Extract each paper's CORE BELIEF — its conviction about HOW this class of problem
   should be solved.
2. Group papers that SHARE THE SAME BELIEF into Innovation Patterns.

A core belief is NOT what problem the paper solves, nor what method it uses.
It IS the author's conviction about the STRUCTURE of the problem — what kind of
problem IS this, and therefore what kind of solution MUST work.

LITMUS TEST: If your sentence could be replaced by "We propose a method for X"
without losing insight, it's a METHOD DESCRIPTION, not a belief. Rewrite it until
it captures the CONVICTION behind the method.

CRITICAL: Every belief must have a plausible OPPOSITE belief. If you can't imagine
someone holding the opposite conviction, you haven't found the belief yet.

Examples of GOOD beliefs (used in worldview_experiment.py, validated across 5 runs):
  "Depth determines BEV quality; explicit supervision is essential, not optional"
    → Opposite belief: "Depth can be learned implicitly; explicit supervision is unnecessary"
  "The world is object instances; dense BEV grids are a wasteful intermediate artifact"
    → Opposite belief: "Dense BEV grids are necessary for complete scene coverage"
  "Planning is the organizing principle; perception must serve planning, not lead it"
    → Opposite belief: "Perception is independent; planning consumes its output"

Examples of BAD beliefs (method description, not conviction):
  "Temporal fusion from a single frame improves detection with minimal changes"
    → Better: "Temporal cues are simple to exploit; you don't need complex architectures"
  "Recurrent temporal fusion reduces complexity from O(T) to O(1)"
    → Better: "Temporal fusion should be persistent and recurrent, not multi-frame stacking"

When grouping: papers that share the SAME conviction belong to the SAME pattern,
even if they propose different specific techniques. Two papers with DIFFERENT
convictions belong to DIFFERENT patterns, even if they address similar problems.

Return ONLY a JSON object. No other text.
```

### User Prompt

```text
Extract the CORE BELIEF for each paper, then group papers by shared belief into
Innovation Patterns.

PAPERS AND THEIR CLAIMS:
{claims_text}

A pattern must have at least 2 papers. If a paper's belief is unique (no other
paper shares it), flag it as "unclustered" rather than creating a single-paper pattern.

Return JSON:
{
  "papers": [
    {"paper": "full title", "belief": "one-sentence core conviction", "year": 2020}
  ],
  "patterns": [
    {
      "pattern_name": "short label for this technical route",
      "shared_belief": "the belief all papers in this pattern share",
      "papers": ["title1", "title2"],
      "dominant_paper": "the paper that best crystallizes this belief"
    }
  ],
  "unclustered": ["paper titles whose beliefs are unique, if any"]
}
```

### 设计决策

1. **Good/Bad belief 示例**：保留了worldview_experiment.py 中验证过的示例。与之前被删除的 "BEV 3-phase 示例" 不同——这些示例教 LLM "什么是 belief vs method description"，不暗示 phase 数量。示例本身是领域相关的（BEV），但它们约束的是 belief 的**质量标准**，不是分组结构。如果这构成了 hidden bias，可以替换为抽象示例或跨领域示例。

2. **Combined extraction + grouping**：LLM 同时输出 beliefs 和 patterns，可以自校准——如果两个论文的 belief 措辞相似但被分到不同 pattern，LLM 可以在输出前调整。

3. **"Unclustered" 机制**：替代 singleton merge。与其事后合并单篇 phase，不如让 LLM 主动标记独特信念，避免强行分组。

---

## Step 2: Validation Heuristics

```python
def detect_validation(
    patterns: list[InnovationPattern],
    relations: list[ClaimRelation],
) -> list[InnovationPattern]:
    """Heuristic validation — no LLM. Deterministic."""
    for pattern in patterns:
        signals = []
        paper_set = set(pattern.papers)

        # Signal 1: Size ≥ 2
        if len(pattern.papers) >= 2:
            signals.append("size>=2")

        # Signal 2: Internal follow-up (improve/extend within pattern)
        if any(r.relation in ("improve", "extend")
               and r.source_paper in paper_set
               and r.target_paper in paper_set
               for r in relations):
            signals.append("internal_followup_chain")

        # Signal 3: Cross-pattern engagement (replace/attack across patterns)
        if any(r.relation in ("replace", "attack")
               and ((r.source_paper in paper_set) != (r.target_paper in paper_set))
               for r in relations):
            signals.append("cross_pattern_engagement")

        # Signal 4: Time span ≥ 6 months
        span = (pattern.year_end - pattern.year_start) * 12 + \
               (pattern.month_end - pattern.month_start)
        if span >= 6:
            signals.append(f"active_{span}months")

        pattern.validation_signals = signals
        pattern.validated = len(signals) >= 2

    return patterns
```

阈值 `≥ 2 signals` 是保守的——大多数 legitimate patterns 自然满足 size + followup 两个信号。

### 讨论点

- **Signal 4（时间跨度）可能过于机械**：一个 2024-12 的 paper 和 2025-01 的 paper 组成 pattern，跨度只有 1 个月，但可能是 genuine paradigm。是否需要调整？
- **"Unclustered" 论文如何贡献 validation 信号**：如果一篇 unclustered 论文被后续 pattern 大量引用，它可能是一个重要的 precursor。当前 heuristics 不处理这种情况。

---

## Step 3: Phase Building Prompt

基于当前 `_PHASE_BUILDING_PROMPT` 修改，区别是输入从 "paradigm groups" 变为 "validated + unvalidated patterns"：

```text
Build phases from Innovation Patterns in the field of {field_name}.

VALIDATED PATTERNS (community-accepted paradigms — each becomes a phase):
{validated_text}

UNVALIDATED EXPLORATIONS (explored but didn't become dominant — context for narrative):
{unvalidated_text}

For each VALIDATED pattern, output a Phase with:
- name: question-based phase name
- dominant_question: Core research question driving this phase
- core_contradiction: The central tension/contradiction within this phase (1 sentence)
- core_debate: What was the field debating? (1 sentence)
- key_papers: List of paper titles in this phase (all papers from the pattern)
- time_range: e.g. "2020-08—2022-11"
- internal_tensions: debates between competing positions within the phase
- unresolved_problem: The problem that remains → seeds the next phase (1 sentence)
- status: "direction_clear" | "direction_forming" | "open"

PHASE COUNT MUST MATCH VALIDATED PATTERN COUNT: {n_validated}
Each paper MUST appear in exactly one phase (COVERAGE: {n_papers} papers total)
unresolved_problem should logically motivate the NEXT phase

For unvalidated patterns, output "exploration_waves" — one sentence each on what was
tried and why it didn't become dominant.

Return JSON:
{
  "phases": [...],
  "exploration_waves": [
    {"pattern_name": "...", "description": "one sentence"}
  ]
}
```

---

## 数据模型

### 新增 InnovationPattern

```python
@dataclass
class InnovationPattern:
    """A group of papers sharing the same core belief — a technical route."""
    pattern_name: str            # Short label
    shared_belief: str           # The belief ALL papers in this pattern share
    papers: list[str]            # Paper titles (sorted by time)
    time_range: str              # e.g. "2020-08—2022-06"
    year_start: int = 0
    month_start: int = 0
    year_end: int = 0
    month_end: int = 0
    dominant_paper: str = ""     # The paper that best crystallizes this belief
    validated: bool = False
    validation_signals: list[str] = field(default_factory=list)
```

### 保留 PaperBelief

不变（paper_title, belief, year, month）。每篇论文的 worldview 提取结果，可独立审查。

### Phase 不变

现有 Phase dataclass 接口不变，Phase Building step 的输出格式不变。

---

## 稳定性分析

### 为什么这个方案比 Label-Based 更稳定

| 维度 | Label-Based（当前） | Worldview-Driven（新） |
|------|---------------------|------------------------|
| LLM 任务 | 直接分配 phase_label（自由度高） | 提取 belief（严格约束） + 按 belief 分组（低自由度） |
| 中间产物 | 无（paper → phase_label，不透明） | 有（paper → belief → pattern → validated → phase） |
| Grouping | LLM 隐式决定 | LLM 以 belief 文本为依据分组（可审查） |
| Validation | 不存在 | 确定性 heuristics |
| 稳定性锚点 | "core question changes"（抽象） | belief 约束（falsifiable, opposite test） |
| Phase 数量 | LLM 自由决定 | Pattern 数量由 LLM 发现 + Validation 过滤 |

### 已知的方差来源

1. **Belief 措辞变化**（低影响）：5-run 实验已证实无意义漂移，但措辞差异可能影响 grouping
2. **Grouping 粒度**（中影响）：LLM 可能把 "temporal fusion" 和 "depth supervision" 拆成两个 pattern 或合并为一个——这是合理的史学歧义
3. **Validation heuristics**（无方差）：确定性，无 LLM 随机性

---

## 与之前方案的对比

| 维度 | Tension-Driven | Problem-Driven | Label-Based | Worldview-Driven |
|------|---------------|----------------|-------------|------------------|
| 分组依据 | LLM 发明的 tension | problem 语义相似度 | LLM 直接分 phase | belief 相似度 → pattern |
| LLM 调用 | 2 次 | 1 次 | 2 次 | 2 次 |
| 中间产物可审查 | 否 | 否 | 否 | 是（belief + pattern） |
| Validation | 无 | 无 | 无 | 有（heuristics） |
| Phase 定义 | tension 聚类 | problem 聚类 | LLM 分配 | validated paradigm 统治期 |
| 适应性 | 低 | 中 | 中 | 高（pattern 数量自适应领域） |

---

## 待讨论

1. **Good/Bad belief 示例是否构成 hidden hardcode？** 示例来自 BEV，但它们约束的是 belief 质量而非 grouping 结构。如果认为这是 bias，可以改为抽象示例或随机轮换示例。

2. **Validation heuristics 的阈值**：`≥ 2 signals` 对大多数领域合理，但极端情况（如只有 3 篇论文的领域，可能 0 个 pattern 被 validated）需要验证。

3. **"Unclustered" 论文的处理**：当前设计中 unclustered 论文不参与 phase building。但它们可能是重要的 precursor 或 outlier。是否应该将它们注入到 narrative 中作为"独立探索"？

4. **跨 pattern 时间重叠**：两个 validated patterns 在时间上可能重叠（如 attention-based BEV 和 depth-based BEV 在 2022 年同时活跃）。Phase 如何排序？当前按 earliest paper 时间排序，但可能需要考虑"主导 paradigm"的概念——同一时间只有一个 dominant paradigm。
