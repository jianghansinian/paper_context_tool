# Problem-Driven Phase 设计

## 动机

当前 tension-driven phase 划分有两层不稳定性：

1. `detect_all_tensions` 产出的 8-12 个 tension 是 LLM 生成的中间抽象，每次 run 不同
2. `merge_tensions_into_phases` 对这些不稳定的 tension 再聚类，方差放大

结果：phase 数量、论文归属、时间范围都不稳定。GPT 的做法更稳定——按"论文在解决什么问题"直接分组，不需要中间抽象。

## 核心思路

**让 phase 从 problem 的自然分组中涌现，而不是从 LLM 发明的 tension 中聚类。**

```
当前:  claims → tensions (LLM抽象) → phase (LLM聚类抽象)  ← 两层不稳定
改为:  claims → problem clusters → phase (LLM命名+提取tension) ← 单层，problem 是显式字段
```

## 架构

```
Step 1: Problem Clustering
  claim.problem_addressed → LLM 按 problem 语义分组 → 4-6 个 problem cluster
  输入：每篇论文的 (title, year, month, problem_addressed[])
  输出：每个 cluster = {papers[], core_problem, time_range}

Step 2: Phase 命名 + 内部 tension
  每个 cluster → LLM 提炼 phase name, core_question, 内部 tensions
  1 次 LLM 调用，批量处理所有 cluster

Step 3: Causal Chain（自动）
  Phase N 的遗留问题 = Phase N+1 的种子 problem
  不需要 LLM 推断——从 problem 演化自然形成

Step 4: 其他模块不变
  claim_relations → mermaid 图（按 phase 过滤）
  paradigm_shifts → 跨 phase 注释
  narrative_builder → 按 phase 生成叙事（输入改为 phase + 内部 tensions）
```

## 数据流

```
claims (all_claims)
  │
  ├─→ build_paper_chain_relations (不变)
  │
  ├─→ detect_problem_based_phases 【新】
  │     │
  │     │ 输入: papers[title, year, month, problem_addressed]
  │     │ LLM: 按 problem 语义分组
  │     │ 输出: Phase[] (含 name, question, papers, time_range,
  │     │                 internal_tensions, unresolved_problem)
  │     │
  │     └─→ phases
  │
  ├─→ detect_research_questions (不变，但 RQ 映射到 phase)
  │
  ├─→ detect_paradigm_shifts (不变)
  │
  └─→ build_narrative (输入改为 problem-driven phases)
        │
        └─→ markdown export
```

## Phase 数据结构（复用现有）

```python
@dataclass
class Phase:
    name: str                    # "Industrializing BEV Perception"
    dominant_question: str       # "How to make BEV fast enough for deployment?"
    time_range: str              # "2020-08—2022-06"
    key_papers: list[str]        # 按 problem 语义自动归入
    tensions: list[Tension]      # LLM 在 phase 内检测
    core_contradiction: str      # 集群内核心矛盾
    core_debate: str             # 核心辩论
    unresolved_problem: str      # → 下一 phase 的种子
    status: str                  # "direction_clear" | "direction_forming" | "open"
```

## Prompt 设计

### 单次 LLM 调用：problem → phases

核心约束：
- 分组依据是 **problem 语义相似度**，不是方法相似度
- "用深度监督做 BEV" 和 "用 attention 隐式学深度做 BEV" 解决同一个 problem（如何构建准确 BEV），归入同一 phase
- "引入时序信息" 是不同 problem，另立 phase
- 每个 phase 2-6 篇论文
- 每篇论文只能属于一个 phase（但可以出现在多个 tension 的 evidence 中）

输出结构：
```json
{
  "phases": [
    {
      "name": "How to Build BEV from Multi-Camera Images?",
      "dominant_question": "Can a 3D bird's-eye-view be reliably constructed from 2D images alone?",
      "time_range": "2020-08—2022-06",
      "key_papers": ["LSS (2020-08)", "BEVDet (2021-12)", "BEVDepth (2022-06)"],
      "core_contradiction": "BEV construction needs depth but explicit depth labels constrain architecture choices",
      "core_debate": "Should depth be learned implicitly or supervised explicitly?",
      "tensions": [
        {
          "tension": "Implicit depth is simpler but less reliable",
          "introduced_by": ["LSS"],
          "resolved_by": ["BEVDepth"],
          "status": "direction_clear",
          "dimension": "geometry",
          "domain_scope": "camera-only BEV perception on nuScenes"
        }
      ],
      "unresolved_problem": "Dense BEV grids waste computation on empty space",
      "status": "direction_clear"
    }
  ]
}
```

### 关键约束（prompt rules）

1. **按 problem 分组，不按 method 分组**。解法相反但问题相同 → 同一 phase。
2. **时间范围由群内论文的最早和最晚时间决定**。
3. **每篇论文只进入一个 phase**（以其主要解决的 problem 为准）。
4. **time_range 必须包含所有 key_papers 的发表时间**。
5. **unresolved_problem 是 concrete 的技术问题**，不是 vague 的方向描述。应该能直接作为下一 phase 的 problem 陈述。
6. **4-6 个 phase**。太少丢失粒度，太多碎片化。

## Phase 间因果链（自动）

```
Phase 1 unresolved: "Dense BEV grids waste computation on empty space"
  → Phase 2 problem: "How to represent the scene sparsely without losing accuracy?"

Phase 2 unresolved: "Sparse representations lack instance-level structure for planning"
  → Phase 3 problem: "How to unify perception, prediction, and planning?"
```

叙事生成时，Phase N 的开头用上一 phase 的 `unresolved_problem` 作为 "承接上一阶段"。

## GPT 对比

| | GPT 做法 | 我们的设计 |
|---|---|---|
| 分组依据 | 人工判断（写作者的经验） | `problem_addressed` 语义聚类 + LLM |
| 稳定性来源 | 人的认知模型 | claim 的显式字段，不依赖中间抽象 |
| Phase 粒度 | 6 个 | 4-6 个（可控） |
| 内部张力 | 无 | 有（LLM 在 phase 内检测） |
| 证据 | 无 | 有（claims table） |
| 关系图 | ASCII 示意图 | mermaid 演化图 |

## 待讨论

1. **Problem 聚类方式**：纯 LLM（一次调用）vs embedding + 启发式（降低成本但增加复杂度）？建议先用纯 LLM，简单且与现有架构一致。
2. **Phase 数量**：4-6 还是让 LLM 自己决定？建议设 min=3 max=6，让 LLM 在这个范围内选择最自然的切分。
3. **现有 tension_detector.py 的处理**：`detect_all_tensions` 不再需要（tension 改为 phase 内部生成），`merge_tensions_into_phases` 被新函数替代。保留文件但废弃这两个函数。
4. **research_question_detector.py**：RQ 仍然有价值（跨 phase 的 field-level question），但不再是 narrative section 的 organizing entity。可以保留在领域全景部分。
