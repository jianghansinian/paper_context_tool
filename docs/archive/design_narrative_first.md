# Narrative-First Phase Detection — Design Document

## 问题诊断

当前 pipeline：`claims → [shift detection] → stages → phases → narrative`

5-run 稳定性测试结果：
- **Phase 数量**：3-5（3 是正确答案，但 2/5 跑膨胀到 5）
- **Shift 命名**：5 跑产生 16 个不同 shift 名，没有一个 shift 在 5 跑中一致出现
- **Paper 分组**：Phase 1/2/3 的核心论文分配非常稳定（4/5 或 5/5）
- **根因**：shift detection 的 CoT 在 LLM 内部发生，不被保存或复用。同一个 LLM 既要写故事又要检测 shift，导致 shift 数量不稳定

## 核心假设

**如果先让 LLM 写出一个完整、连贯的演化叙事（narrative），再从这个叙事中提取 paradigm shift，那么 shift 的数量和命名会更稳定。**

理由：
1. 写叙事是 LLM 的强项——它自然会把离散事件组织成有因果的线性故事
2. 叙事中的"转折点"天然对应 paradigm shift，且已被叙事文本锚定，无法随意增删
3. 从叙事中提取 shift 是一个约束更强的任务——LLM 不能凭空发明叙事中不存在的 shift
4. 叙事本身是一个可审查的中间产物——如果叙事质量差，可以不改 shift detection 而是改进叙事生成

## 新 Pipeline

```
claims + relations + tensions
  │
  ├─ Step 1: Generate Field Narrative (1 LLM call)  ← NEW
  │     Input: claims (chronological), relations, tensions, RQs
  │     Output: A narrative text (800-1500 words) covering:
  │       - The evolution story from earliest to latest paper
  │       - TURNING POINTS explicitly marked, with paper names
  │       - Causal chains: why each shift happened
  │
  ├─ Step 2: Extract Shifts from Narrative (1 LLM call)  ← MODIFIED
  │     Input: Narrative text + claims (for paper name matching)
  │     Output: Structured paradigm shifts (same JSON format)
  │     Key: Shifts MUST be grounded in the narrative's 
  │           turning points. The LLM reads the story and 
  │           formalizes the shifts, rather than detecting 
  │           them from raw data.
  │
  └─ Step 3: Build Phases from Shifts (deterministic, unchanged)
        N shifts → N+1 stages → assign papers → build phases
```

### 与当前 pipeline 的对比

| 维度 | 当前 | 新方案 |
|------|------|--------|
| Shift 检测 | 从 claims + tensions 直接检测 | 从叙事文本中提取 |
| CoT 可见性 | LLM 内部，不可审查 | 叙事是独立的中间产物 |
| 信息丰富度 | claims 是离散点 | 叙事包含因果链 + 上下文 |
| 稳定性来源 | prompt 约束 | 叙事文本锚定 + prompt 约束 |
| LLM 调用数 | 3（shift + assign + phase） | 4（narrative + shift + assign + phase） |
| 额外成本 | — | +1 LLM call，约 2000 tokens 输出 |

## 数据模型

无需新增数据模型。Narrative 是纯文本（str），shifts 和 phases 复用现有格式。

## Prompt 设计

### Step 1: 生成演化叙事 (`_NARRATIVE_FIRST_SYSTEM` / `_NARRATIVE_FIRST_PROMPT`)

这个 prompt 要求 LLM 写出整个领域的演化史，并在文本中明确标注转折点。

**System prompt 核心约束：**
- 你是研究史学家，写一个领域的演化故事
- 故事必须基于提供的 claims、tensions、relations，不能编造
- 明确标注每个 TURNING POINT（用 `[TURNING POINT]` 标记）
- 每个 turning point 必须命名涉及的 papers
- 区分 paradigm shift（信念改变）和 technique improvement（方法改进）

**User prompt 输入：**
- claims（按时间排序，含 claim_level）
- tensions（RQ-nested，含 introduced_by/resolved_by）
- relations（ATTACK/REPLACE/PARALLEL）
- RQs（核心研究问题，含 direction）

**输出：** 纯文本叙事，800-1500 词，包含 `[TURNING POINT]` 标记

### Step 2: 从叙事中提取 Shifts (`_NARRATIVE_SHIFT_SYSTEM` / `_NARRATIVE_SHIFT_PROMPT`)

这个 prompt 将叙事中的 turning points 形式化为结构化的 paradigm shifts。

**System prompt 核心约束：**
- 从下面的叙事中提取 paradigm shifts
- 每个 shift 必须对应叙事中的一个 turning point
- 不要添加叙事中不存在的 shift
- 如果叙事中某个 turning point 是 technique evolution 而非 paradigm shift，跳过它

**User prompt 输入：**
- 完整的叙事文本
- claims 列表（仅用于 catalyst paper 的 exact title matching）

**输出：** JSON `{"shifts": [...]}` —— 与当前 `_SHIFT_PROMPT` 相同的输出格式

## 实现计划

### 文件变更

| 文件 | 变更 |
|------|------|
| `src/worldview_phase_detector.py` | 新增 `_generate_field_narrative()` 和 `_extract_shifts_from_narrative()`；修改 `detect_worldview_phases()` 可选走 narrative-first 路径 |
| `experiments/mvp_bev.py` | 添加 `--narrative-first` flag 切换新旧路径 |

### 具体步骤

1. **新增 `_generate_field_narrative()`**
   - 输入：claims, relations, tensions, rqs, field_name, client
   - 输出：str（叙事文本）
   - 构建 prompt：claims 按时间排列，tensions 标注 introduced_by/resolved_by，relations 标注 ATTACK/REPLACE
   - Temperature=0.3, max_tokens=3072

2. **新增 `_extract_shifts_from_narrative()`**
   - 输入：narrative_text, claims, field_name, client
   - 输出：list[dict]（与当前 `_detect_shifts` 相同的 shifts 格式）
   - 构建 prompt：叙事全文 + claims 列表
   - Temperature=0.1, max_tokens=2048

3. **修改 `detect_worldview_phases()`**
   - 新增参数 `narrative_first: bool = False`
   - 当 `narrative_first=True` 时：
     - 调用 `_generate_field_narrative()` 生成叙事
     - 调用 `_extract_shifts_from_narrative()` 从叙事中提取 shifts
     - 其余步骤（stages, assignment, phase building）不变
   - 叙事文本保存到返回值中

4. **修改 `mvp_bev.py`**
   - 添加 `--narrative-first` 命令行参数
   - 默认保持当前行为（不改变现有流程）

## 验证方案

```bash
# 单次运行验证
python experiments/mvp_bev.py --narrative-first

# 5-run 稳定性测试（与之前相同的脚本）
python /tmp/stability_test_nf.py
```

检查点：
1. 叙事文本质量：是否有清晰的因果链？turning points 是否合理？
2. Shift 数量：在 2-3 之间（对应 Dense→Sparse→Planning）？
3. Shift 命名一致性：5 跑中同样的 shift 是否用相同/相似的名称？
4. Phase 数量：稳定在 3？
5. Paper 分组：保持当前的稳定性（不退化）

## 为什么这次可能不同

之前的 CoT prompt 也要求 LLM "先写故事再提取 shift"，但：
- 故事在同一个 LLM 调用中生成，LLM 没有机会"重读"自己的故事
- 故事不被保存，无法审查或改进
- Shift 提取和故事生成共享同一个 context window，可能互相干扰

新方案把这两个任务拆分到两次 LLM 调用中：
- 第一次：专注写故事，有完整的 context window
- 第二次：只读故事，提取 shift，任务简单且约束强

叙事作为中间产物，如果 shift 仍不稳定，我们可以直接检查叙事质量，定位问题是"故事写得不好"还是"shift 提取不够准"。
