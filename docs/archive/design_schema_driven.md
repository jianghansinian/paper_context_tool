# V3 架构增强设计：Schema 驱动的结构化分析

> **状态：设计提案** — 本文档描述的是 V3 的架构演进方向，**当前代码尚未实施**。
> 当前 V3 的实际实现以 `design_v3.md` 为准。
> 本文档的设计目标是在不破坏现有功能的前提下，逐步将硬编码的分析维度迁移为 schema 驱动的可插拔架构。
>
> 迁移路径见[第 12 节](#12-迁移路径)，分为 Phase A-D 四个阶段。

## 1. 问题分析

V3 当前的结构化分析（`design_v3.md` 描述的实现）在三个组件之间存在**刚性耦合**：

```
structured_analyzer.py          markdown_exporter_v3.py     paper.py
┌─────────────────────┐         ┌──────────────────────┐    ┌──────────────────┐
│ 固定 prompt           │         │ 固定 8 段模板          │    │ 固定数据类        │
│ (字符串内嵌 JSON      │ ──────► │ (硬编码 §1-§8         │    │ StructuredUnder-  │
│  schema，硬编码)       │  data   │  渲染顺序)             │    │ standing 字段)    │
└─────────────────────┘         └──────────────────────┘    └──────────────────┘
```

三个组件各自独立维护，没有统一的"分析维度"定义。

**具体问题**：

| # | 问题 | 影响 |
|---|---|---|
| 1 | **一刀切的 prompt** — 实验论文、综述、理论论文、立场论文共用同一套 JSON schema | LLM 被迫为不适用的字段编造内容（例如让综述论文提取 "training_procedure"），输出质量下降 |
| 2 | **Prompt 与输出脱节** — prompt 中的 JSON schema 和 markdown 模板分开维护，没有形式化关联 | 增加一个字段需要改 3 处以上（prompt 字符串、`StructuredUnderstanding` 数据类、`export_markdown` 渲染逻辑），容易遗漏 |
| 3 | **领域锁定** — 所有 prompt 和模板都假设是 AI/ML 实验型论文 | 扩展到生物、社会科学等领域需要 fork 整个 pipeline |
| 4 | **空章节污染输出** — 缺失数据仍然渲染章节标题和"（无数据）"占位符 | 非标准论文的阅读体验差 |

**根因**：分析 schema 是隐式的——散落在 prompt 字符串、dataclass 字段和 markdown 渲染代码中。没有单一真相源。

---

## 2. 设计目标

1. **Schema 作为单一真相源** — 一份定义同时驱动 prompt 生成和 markdown 渲染
2. **论文类型感知** — 不同论文类型（实验型、理论型、综述、立场论文）使用不同的分析 schema
3. **领域可插拔** — AI/ML 作为首个领域；生物、材料科学等可作为新领域 profile 加入
4. **优雅的空值处理** — 无数据的章节静默跳过，不渲染空占位符
5. **向后兼容** — 现有 `StructuredUnderstanding` 数据类成为多个 schema 之一；已有数据继续可用

---

## 3. 核心概念

```
                    ┌──────────────────────────────────┐
                    │         分析 SCHEMA                │
                    │  (每个领域下每种论文类型的          │
                    │   单一真相源)                      │
                    │                                    │
                    │  • 提取哪些字段                    │
                    │  • 每个字段的类型                  │
                    │  • 如何向 LLM 提问                 │
                    │  • 如何在 markdown 中渲染          │
                    │  • 属于哪个章节                    │
                    └──────────────────────────────────┘
                           │                  │
                           ▼                  ▼
              ┌──────────────────┐  ┌──────────────────┐
              │ Prompt 生成器     │  │ Markdown 渲染器   │
              │                  │  │                  │
              │ Schema → LLM     │  │ Schema + 数据 →  │
              │ prompt（含结构化  │  │ markdown 章节    │
              │ JSON 输出格式）   │  │ （条件渲染）      │
              └──────────────────┘  └──────────────────┘
```

**核心洞察**：prompt 和输出是同一份 schema 的两个视图。在 schema 中增加一个字段，LLM prompt 和 markdown 输出自动同步。

---

## 4. 领域 Profile 体系

### 4.1 层级结构

```
DomainProfile ("ai_ml")
  ├── PaperTypeProfile ("experimental" — 实验型)
  │     ├── 字段: problem, motivation, ... architecture_overview,
  │     │        components, formulas, training_procedure,
  │     │        inference_procedure, main_results, ...
  │     └── 章节: §1 概览, §2 问题, §3 方法, §4 实验, ...
  │
  ├── PaperTypeProfile ("theoretical" — 理论型)
  │     ├── 字段: problem, motivation, ... theoretical_framework,
  │     │        definitions, theorems, proof_sketch,
  │     │        theoretical_results, ...
  │     └── 章节: §1 概览, §2 问题, §3 理论框架, §4 结论, ...
  │
  ├── PaperTypeProfile ("survey" — 综述型)
  │     ├── 字段: problem, scope, taxonomy_overview,
  │     │        taxonomy_categories, trends, open_challenges, ...
  │     └── 章节: §1 概览, §2 范围, §3 分类体系, §4 趋势, ...
  │
  └── PaperTypeProfile ("position" — 立场型)
        ├── 字段: problem, position_statement, arguments,
        │        supporting_evidence, counter_arguments, ...
        └── 章节: ...
```

每个领域 profile 放在 `src/domains/` 目录下的独立文件中。

### 4.2 PaperTypeProfile 构成

一种论文类型**完全由其字段定义和章节定义来描述**，没有其他硬编码：

```python
# 概念示意 — 非最终 API
PaperTypeProfile(
    type_name="experimental",
    description="具有方法、训练、实验、消融的实证论文",

    # LLM 被要求提取的字段
    fields=[
        FieldDef(name="problem", kind="text", required=True,
                 label_en="Problem", label_zh="问题定义",
                 prompt="本文解决什么问题？"),
        FieldDef(name="motivation", kind="text", required=True,
                 label_en="Motivation", label_zh="动机",
                 prompt="为什么这个问题重要？现有工作有哪些不足？"),
        FieldDef(name="key_insight", kind="text", required=True,
                 label_en="Key Insight", label_zh="核心思路",
                 prompt="本文的核心思路/关键洞察是什么？"),
        FieldDef(name="components", kind="component_table", required=False,
                 label_en="Core Components", label_zh="核心组件",
                 prompt="列出所有主要架构组件及其功能、实现细节、对应图表",
                 columns=[...]),
        # ... 更多字段
    ],

    # Markdown 输出的章节结构
    sections=[
        SectionDef(name="overview", level=1,
                   title_en="Paper Overview", title_zh="论文概览",
                   fields=["title", "authors", "key_insight"]),
        SectionDef(name="problem", level=1,
                   title_en="Problem Definition", title_zh="问题定义与动机",
                   fields=["problem", "motivation"]),
        # ... 更多章节
    ],
)
```

### 4.3 字段类型（Field Kind）

| Kind | JSON 类型 | Markdown 渲染方式 | 适用场景 |
|---|---|---|---|
| `text` | `string \| null` | 段落文本 | `architecture_overview` |
| `list[str]` | `["..."]` | 项目符号列表 | `contributions`, `limitations` |
| `component_table` | `[{"name","purpose","details","fig"}]` | Markdown 表格 | `components` |
| `formula_table` | `[{"name","latex","explanation","significance"}]` | 含 LaTeX 的表格 | `formulas` |
| `result_table` | `[{"dataset","metric","value","comparison"}]` | Markdown 表格 | `main_results` |
| `key_value_table` | `[{"key","value"}]` | 两列表格 | `definitions`（理论论文） |
| `structured_list` | `[{"title","description","papers"}]` | 嵌套列表 | `taxonomy_categories`（综述） |

---

## 5. 论文类型检测

检测发生在 Phase 1（论文解析+文本提取）**之后**，此时 PDF 全文已提取完毕。应充分利用已获取的内容来提高检测准确性。

### 5.1 输入内容

| 内容片段 | 来源 | 字符数 | 检测价值 |
|---|---|---|---|
| 标题 | metadata | ~100 | 高 — 常包含方法名、任务名 |
| 完整摘要 | metadata/PDF | ~1500 | 高 — 浓缩了论文的全部要素 |
| 引言前段 | PDF | ~2000 | 高 — 详述问题背景和方法思路 |
| 章节标题列表 | PDF | ~300 | 中 — "Experiments"/"Proof"/"Related Work" 是强信号 |
| 结论首段 | PDF | ~800 | 中 — 常以 "In this paper, we propose/present/introduce..." 开头，直接暴露贡献类型 |

**总计输入**：约 5000 字符（~1200 tokens），成本可忽略。

### 5.2 提取逻辑

```python
def build_detection_input(paper: Paper) -> str:
    """从已解析的论文中组装类型检测所需的输入文本。"""
    text = paper.full_text or ""
    abstract = paper.abstract or ""

    # 1. 引言：abstract 之后的前 2000 字符
    intro_start = text.find(abstract) + len(abstract) if abstract in text else 0
    intro = text[intro_start:intro_start + 2000] if intro_start < len(text) else ""

    # 2. 章节标题：匹配 "1. Introduction", "2. Method" 等模式
    import re
    headings = re.findall(r'(?m)^\s*(?:\d+\.?\s+)?([A-Z][a-zA-Z\s]{2,50})$', text)
    headings_str = ", ".join(headings[:20])

    # 3. 结论首段：找 "Conclusion" 章节后的第一段
    conclusion_start = _find_section_start(text, ["conclusion", "concluding remarks", "discussion"])
    conclusion = text[conclusion_start:conclusion_start + 800] if conclusion_start else ""

    return f"""TITLE: {paper.title}
ABSTRACT: {abstract}
INTRODUCTION: {intro}
SECTIONS: {headings_str}
CONCLUSION: {conclusion}"""
```

### 5.3 检测 prompt（领域感知）

```
你正在对一篇 {domain_name} 领域的论文进行类型分类。
候选类型:
{paper_type_descriptions}

请根据以下论文内容判断其类型:
{detection_input}

返回 JSON: {{"paper_type": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
```

### 5.4 降级策略

| 场景 | 处理 |
|---|---|
| confidence ≥ 0.7 | 使用检测结果 |
| 0.5 ≤ confidence < 0.7 | 使用检测结果，但在报告中标注低置信度 |
| confidence < 0.5 或解析失败 | 使用 `domain.default_paper_type` |
| 论文无全文（仅有 metadata） | 仅用标题+摘要检测，confidence 阈值放宽到 0.4 |

### 5.5 成本

每次检测约 1200 input + 80 output tokens。按 DeepSeek 计价约 ¥0.00003。

---

## 6. 从 Schema 生成 Prompt

### 6.1 生成算法

```python
def generate_analysis_prompt(paper, paper_type_profile, domain_name):
    """完全从 schema 定义构建 LLM prompt"""

    # 1. 构建 JSON schema 部分
    field_specs = []
    for f in paper_type_profile.fields:
        spec = f'  "{f.name}": {_describe_json_type(f)}'
        if not f.required:
            spec += '  // 可选字段，不适用时用 null'
        spec += f'\n    // {f.prompt}'
        field_specs.append(spec)

    json_schema = "{\n" + ",\n".join(field_specs) + "\n}"

    # 2. 构建指令部分
    required_fields = [f for f in paper_type_profile.fields if f.required]
    optional_fields = [f for f in paper_type_profile.fields if not f.required]

    instructions = f"""
    你正在分析一篇 {domain_name} 领域的 {paper_type_profile.type_name} 类型论文。

    分析步骤:
    {chr(10).join(f'{i+1}. {f.prompt}' for i, f in enumerate(paper_type_profile.fields))}

    关键要求:
    - 必填字段: {', '.join(f.name for f in required_fields)}
    - 可选字段 ({', '.join(f.name for f in optional_fields)}):
      信息不存在时使用 null（不是空字符串）
    - 只返回 JSON 对象，不要其他文字。
    """

    return f"{instructions}\n\n返回 JSON:\n{json_schema}\n\n论文正文:\n{text}"
```

### 6.2 具体示例：理论型论文的 Schema

为理论型论文生成的 prompt 会要求提取：

```json
{
  "problem": "string",
  "motivation": "string",
  "key_insight": "string",
  "theoretical_framework": "string",       // 替代 architecture_overview
  "key_definitions": [                     // 替代 components 表
    {"term": "...", "definition": "...", "significance": "..."}
  ],
  "theorems": [                            // 替代 formulas
    {"name": "...", "statement": "...", "proof_sketch": "...", "significance": "..."}
  ],
  "assumptions": ["..."],                  // 理论论文特有
  "theoretical_results": [                 // 替代 main_results
    {"type": "bound|guarantee|complexity", "statement": "...", "conditions": "..."}
  ],
  "contributions": ["..."],
  "limitations": ["..."],
  "open_problems": ["..."]                 // 理论论文特有
}
```

注意：没有 `training_data`、`loss_functions`、`optimizer`、`inference_procedure`、`post_processing`、`ablation_results`、`qualitative_results`——因为 "theoretical" 的 schema 根本不会定义这些字段。

---

## 7. 从 Schema 生成 Markdown

### 7.1 渲染算法

```python
def export_markdown(paper, structured_data, paper_type_profile, routes, comparison, references, lang="en"):
    """完全从章节定义生成 markdown"""

    lines = []

    # 标题
    lines.append(f"# {_section_title('title', lang)}: {paper.title}")

    for section in paper_type_profile.sections:
        # 1. 检查章节级条件
        if section.condition and not _eval_condition(section, paper, structured_data):
            continue

        # 2. 收集本节各字段的渲染内容
        rendered_fields = []
        for field_name in section.fields:
            field = paper_type_profile.get_field(field_name)
            value = structured_data.get(field_name)

            # 可选字段为空 → 静默跳过
            if _is_empty(value) and not field.required:
                continue
            # 必填字段为空 → 渲染占位符
            if _is_empty(value) and field.required:
                rendered_fields.append(_render_missing(field, lang))
                continue

            rendered_fields.append(_render_field(value, field, lang))

        # 3. 全部字段为空 → 跳过整个章节（不输出标题）
        if not rendered_fields:
            continue

        # 4. 输出章节标题 + 内容
        heading = "#" * section.level
        lines.append(f"{heading} {_section_title(section, lang)}")
        lines.extend(rendered_fields)

        # 5. 递归处理子章节
        for sub in section.subsections:
            ...  # 同样逻辑

    return "\n".join(lines)
```

### 7.2 与当前方案的关键差异

| 当前 (`export_markdown`) | 新方案 |
|---|---|
| 章节标题始终渲染 | 章节内所有字段为空 → 整节跳过 |
| "（无数据）"占位符 | 可选字段静默跳过 |
| 每种字段类型的渲染逻辑内联在函数中 | `_render_field()` 按 `field.kind` 分发 |
| 章节顺序硬编码 | 章节顺序来自 `PaperTypeProfile.sections` |
| 增加字段 = 改 3+ 处 | 增加字段 = 在 profile 中加一条 `FieldDef` |

### 7.3 条件章节

章节可配置 `condition` 表达式：

```python
SectionDef(
    name="experiments",
    condition="paper_type == 'experimental'",   # 仅实验型论文显示
    ...
)

SectionDef(
    name="theoretical_results",
    condition="has_field('theoretical_results')",  # 仅 schema 包含此字段时显示
    ...
)
```

这意味着综述论文的 markdown 根本不会出现"方法架构"或"实验结果"章节——它们不在综述的 schema 章节列表中。

---

## 8. 数据模型变更

### 8.1 Schema 定义类（新增）

```python
# src/domains/base.py

@dataclass
class ColumnDef:
    """表格列定义"""
    name: str
    label_en: str
    label_zh: str

@dataclass
class FieldDef:
    """单个分析字段的定义"""
    name: str
    kind: str                   # "text" | "list[str]" | "component_table" | ...
    label_en: str
    label_zh: str
    prompt: str                 # 给 LLM 的提取指令
    required: bool = True
    columns: list[ColumnDef] = field(default_factory=list)

@dataclass
class SectionDef:
    """Markdown 输出章节的定义"""
    name: str
    level: int                  # 1 = #, 2 = ##, 3 = ###
    title_en: str
    title_zh: str
    fields: list[str] = field(default_factory=list)
    condition: Optional[str] = None
    subsections: list["SectionDef"] = field(default_factory=list)

@dataclass
class PaperTypeProfile:
    """一种论文类型的完整 schema"""
    type_name: str
    description: str            # 用于类型检测 prompt
    fields: list[FieldDef]
    sections: list[SectionDef]

@dataclass
class DomainProfile:
    """一个领域的完整配置"""
    domain_name: str
    domain_description: str     # 用于检测 prompt 的上下文
    paper_types: list[PaperTypeProfile]
    default_paper_type: str
```

### 8.2 StructuredUnderstanding 的定位

**当前状态**：`Paper.structured` 类型为 `Optional[StructuredUnderstanding]`（强类型 dataclass）。

**迁移目标**：将 `Paper.structured` 泛化为 `Optional[dict]`，使 pipeline 能接受任意 schema 产生的结构化数据。现有的 `StructuredUnderstanding` 成为 "AI/ML 实验型" schema 的具体输出：

```python
@dataclass
class Paper:
    ...
    # 迁移后:
    structured: Optional[dict] = None              # schema 驱动：灵活的 dict
    paper_type: str = ""                            # [新增] 检测到的论文类型
```

`StructuredUnderstanding.from_dict(d)` / `.to_dict()` 在 schema 为 "experimental" 时继续可用。其他 schema 产生的 dict 由 markdown 渲染器通过 dict key 访问，因此适用于任何 schema。

### 8.3 向后兼容

```
旧代码访问方式:                      新代码访问方式:
paper.structured                     paper.structured
  └─ StructuredUnderstanding           └─ dict (from schema)
       .problem                             ["problem"]
       .components                         ["components"]
       .training_procedure                 ["training_procedure"]
```

现有的 `StructuredUnderstanding` 已有 `.to_dict()` / `.from_dict()` 方法。markdown 渲染器通过 dict key 访问数据，因此适用于任何 schema。

---

## 9. 文件布局（目标状态）

> **注**: `src/domains/` 目录及所有 schema 相关文件尚未创建。当前代码结构见 `design_v3.md` 第 9 节。

```
src/
  domains/                   # [新增] 领域 profile 目录
    __init__.py              # 领域注册中心 + loader
    base.py                  # FieldDef, SectionDef, PaperTypeProfile, DomainProfile
    shared_fields.py         # 跨类型/跨领域共享字段定义
    ai_ml.py                 # AI/ML 领域 profile（6 种论文类型）
    biology.py               # 生物学领域 profile（5 种论文类型）
    materials_science.py     # 材料学领域 profile（5 种论文类型）

  # 现有文件 — 修改为 schema 驱动:
  structured_analyzer.py     # Schema → prompt 生成 + LLM 调用
  markdown_exporter_v3.py    # Schema → markdown 渲染
  paper_type_detector.py     # [新增] 论文类型检测（领域内）
  domain_detector.py         # [新增] 领域检测（跨领域）

  # 现有文件 — 不变:
  paper.py                   # StructuredUnderstanding 保留用于向后兼容
  run_v3.py                  # Pipeline 编排（少量接驳改动）
  ...
```

### 9.1 领域注册机制

```python
# src/domains/__init__.py

_registry: dict[str, DomainProfile] = {}

def register(profile: DomainProfile):
    _registry[profile.domain_name] = profile

def get_domain(name: str) -> DomainProfile:
    if name not in _registry:
        raise ValueError(f"未知领域: {name}。可用领域: {list(_registry)}")
    return _registry[name]

def list_domains() -> list[str]:
    return list(_registry)

# 自动注册内置领域
from .ai_ml import AI_ML_DOMAIN
register(AI_ML_DOMAIN)
```

---

## 10. 多领域 Profile 预设计

### 10.1 AI/ML 领域

AI/ML 论文按**贡献类型**划分为 6 种。区分关键：

- 贡献的是**新方法**（experimental）还是**新数据**（benchmark）？
- 贡献的是**新理论**（theoretical）还是**新工具**（system）？
- 还是不做新贡献，而是**梳理现有工作**（survey）或**表达观点**（position）？

#### 10.1.1 类型总览

| 类型 | 典型论文特征 | 频次 | 核心分析维度 |
|---|---|---|---|
| **experimental** | 提出模型/方法 + 基准实验 + 消融 | 最常用 | 架构、训练、实验 |
| **benchmark** | 新数据集/基准/评估协议 | 常见 | 数据、基线、评估 |
| **system** | 软件框架/库/分布式系统 | 常见 | 架构、设计决策、性能 |
| **theoretical** | 收敛性证明/泛化界/表达能力 | 较少 | 框架、定义、定理 |
| **survey** | 文献综述/领域梳理/分类 | 常见 | 分类体系、趋势 |
| **position** | 立场/愿景/号召行动 | 少见 | 论点、证据、影响 |

#### 10.1.2 experimental（实验型，默认）

```
描述: "提出新方法/模型/算法，并在标准基准上通过实验验证其有效性"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 本文解决什么问题？ |
| motivation | text | ✓ | 为什么这个问题重要？现有方法有哪些不足？ |
| key_insight | text | ✓ | 本文的核心思路/关键洞察是什么？ |
| architecture_overview | text | ✓ | 整体方法/架构的描述，包括数据流 |
| architecture_figure | text | | 对架构图的详细解释（描述每个组件和数据如何流转）。无图则为 null |
| components | component_table | | 每个主要架构组件：名称、功能、实现细节、对应图表 |
| formulas | formula_table | | 每个关键公式：名称、LaTeX、含义、为什么重要 |
| training_data | text | | 使用了哪些数据集进行训练？数据规模和预处理方式 |
| loss_functions | list[str] | | 使用了哪些损失函数？ |
| optimizer | text | | 优化器名称及学习率/调度策略 |
| training_procedure | text | | 详细训练流程：数据增强、分阶段训练、关键技巧 |
| inference_procedure | text | | 推理过程：前向传播的输入输出流程 |
| post_processing | text | | 后处理步骤（如 NMS、阈值过滤）。无则为 null |
| main_results | result_table | | 主要实验结果：数据集、指标、本方法数值、与 baseline 对比 |
| ablation_results | list[str] | | 消融实验的关键发现 |
| qualitative_results | text | | 定性分析/可视化结果。无则为 null |
| contributions | list[str] | ✓ | 本文的主要贡献 |
| limitations | list[str] | ✓ | 作者自述或明显可见的局限性 |

```
章节结构: 概览 → 问题与动机 → 方法架构(整体架构→架构图→组件表→公式表→训练→推理)
         → 实验结果(主结果→消融→定性) → 贡献与局限
         → 领域技术路线(若有关键论文) → 对比分析 → 参考文献分类
```

#### 10.1.3 benchmark（数据集/基准型）

```
描述: "主要贡献是引入新数据集、基准测试套件或评估协议，而非提出新方法"
```

与 experimental 的核心区别：贡献在**数据**而非方法。不需要提取 architecture_overview、formulas、training_procedure 等方法字段。

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 这个数据集/基准解决什么需求？现有数据集有什么不足？ |
| motivation | text | ✓ | 为什么需要这个新数据集/基准？ |
| key_insight | text | ✓ | 数据集设计的核心思路 |
| dataset_description | text | ✓ | 数据集的详细描述：数据来源、规模、标注方式、模态 |
| collection_methodology | text | ✓ | 数据采集/标注的方法论：流程、质量控制、伦理考量 |
| dataset_statistics | key_value_table | ✓ | 数据集统计：样本数、类别数、分布、平均长度等 |
| task_definition | text | ✓ | 该基准定义了什么任务？输入输出是什么？ |
| evaluation_protocol | text | ✓ | 评估协议：指标定义、数据划分、统计显著性检验 |
| baseline_results | result_table | ✓ | 在数据集上运行的基线方法结果 |
| baseline_methods | structured_list | | 被选为基线的方法及其选择理由 |
| known_biases | text | | 已知的数据偏差或局限。无则为 null |
| maintenance_plan | text | | 数据集的长期维护和版本更新计划。无则为 null |
| contributions | list[str] | ✓ | 本文的主要贡献 |
| limitations | list[str] | ✓ | 自述局限性 |

```
章节结构: 概览 → 需求与动机 → 数据集(数据集描述→采集方法→统计→偏差)
         → 任务与评估(任务定义→评估协议→基线方法→基线结果)
         → 贡献与局限 → 参考文献分类
```

#### 10.1.4 system（系统/工具型）

```
描述: "描述软件系统、框架、库或分布式基础设施的设计与实现，核心贡献在工程层面"
```

与 experimental 的区别：核心贡献在**工程设计和系统性能**，而非算法创新。

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 这个系统/工具解决什么需求？ |
| motivation | text | ✓ | 为什么需要这个系统？已有方案有什么不足？ |
| key_insight | text | ✓ | 系统设计的核心思路 |
| system_architecture | text | ✓ | 系统整体架构：模块划分、数据流、部署拓扑 |
| design_decisions | structured_list | ✓ | 关键设计决策：选择了什么方案、为什么、trade-off 分析 |
| api_design | text | | API/接口设计。无则为 null |
| implementation_stack | text | | 实现技术栈：语言、框架、依赖。无则为 null |
| performance_engineering | text | | 性能优化手段：并行、缓存、内存管理、编译优化等 |
| scalability_evaluation | result_table | | 可扩展性评测：规模/吞吐/延迟数据 |
| ecosystem_integration | text | | 与生态系统中其他工具的互操作性。无则为 null |
| comparisons_to_alternatives | result_table | | 与替代方案/竞品的对比 |
| contributions | list[str] | ✓ | 本文的主要贡献 |
| limitations | list[str] | ✓ | 系统局限性 |

```
章节结构: 概览 → 需求 → 系统架构(架构→设计决策→API→技术栈→性能工程)
         → 评测(可扩展性→竞品对比) → 生态定位 → 贡献与局限 → 参考文献分类
```

#### 10.1.5 theoretical（理论型）

```
描述: "以数学推导和理论证明为主，核心贡献是定理、边界或理论框架"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 本文分析什么理论问题？ |
| motivation | text | ✓ | 为什么这个理论问题重要？ |
| key_insight | text | ✓ | 核心理论洞察/证明思路 |
| theoretical_framework | text | ✓ | 理论框架：使用的数学工具和分析范式 |
| key_definitions | key_value_table | ✓ | 关键定义：术语 → 精确定义 |
| theorems | theorem_table | ✓ | 主要定理/引理：名称、陈述、证明概要、意义 |
| assumptions | list[str] | ✓ | 分析所依赖的假设条件 |
| theoretical_results | result_table | ✓ | 理论结果：类型(bound/guarantee/complexity)、陈述、成立条件 |
| proof_technique | text | | 核心证明技术（如 coupling、Lyapunov、信息论方法） |
| connections_to_empirical | text | | 理论结果与实践的联系。无则为 null |
| contributions | list[str] | ✓ | 本文的主要贡献 |
| limitations | list[str] | ✓ | 理论局限性和未解决的问题 |
| open_problems | list[str] | | 作者提出的开放问题 |

```
章节结构: 概览 → 问题与动机 → 理论框架 → 定义 → 假设条件
         → 定理与证明 → 理论结果 → 实践联系 → 贡献与局限 → 开放问题
```

#### 10.1.6 survey（综述型）

```
描述: "对领域文献进行系统梳理、分类和趋势分析，不提出新方法"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 本综述覆盖什么研究问题/领域？ |
| scope | text | ✓ | 综述的范围界定：覆盖时间、子领域、纳入/排除标准 |
| key_insight | text | ✓ | 综述得出的最重要发现/洞察 |
| taxonomy_overview | text | ✓ | 分类体系总览：分类维度和逻辑 |
| taxonomy_categories | structured_list | ✓ | 每个分类：名称、描述、代表论文、技术特征 |
| historical_evolution | text | | 该领域的历史演进脉络 |
| method_comparison | result_table | | 跨方法对比（可选） |
| trends_and_insights | text | ✓ | 观察到的趋势和深层洞察 |
| open_challenges | list[str] | ✓ | 领域当前面临的开放挑战 |
| future_directions | list[str] | | 预测的未来研究方向 |
| contributions | list[str] | ✓ | 本综述的主要贡献 |
| limitations | list[str] | ✓ | 综述的局限性 |

```
章节结构: 概览 → 范围与方法 → 分类体系 → 历史演进
         → 方法对比 → 趋势洞察 → 开放挑战 → 未来方向 → 贡献与局限
```

#### 10.1.7 AI/ML 共享字段

```
基础共享 (6 种类型都包含):
  problem, motivation, key_insight, contributions, limitations

方法类共享 (experimental + system + benchmark):
  (各自维度不同，但都有"方法描述"+"结果评估"结构)

理论类共享 (theoretical):
  独立维度，不与方法类共享
```

---

### 10.2 生物学领域

生物学论文按**证据类型和研究范式**划分。

#### 10.2.1 类型总览

| 类型 | 典型论文特征 | 类比 AI/ML |
|---|---|---|
| **experimental** | 湿实验：假设→方法→结果→解释 | experimental |
| **computational** | 生信工具/算法/数据库查询系统 | system + experimental |
| **review** | 文献综述/系统综述/meta-analysis | survey |
| **data_resource** | 基因组/蛋白质组/表达谱数据库 | benchmark |
| **method_protocol** | 详细实验协议/方法步骤 | 无直接类比 |

#### 10.2.2 experimental（实验研究型）

```
描述: "提出生物学假设并通过湿实验或干实验验证，遵循假设→方法→结果→讨论的经典结构"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 本研究的核心生物学问题是什么？ |
| hypothesis | text | ✓ | 明确的研究假设（若论文未明说，概括作者试图验证的命题） |
| motivation | text | ✓ | 为什么这个生物学问题重要？前人研究的基础与缺口 |
| biological_system | text | ✓ | 研究体系：物种、品系、细胞系、组织、发育阶段等 |
| experimental_methods | structured_list | ✓ | 使用的实验技术：每种方法的名称、用途、关键参数 |
| experimental_design | text | ✓ | 实验设计：分组、对照、重复数、随机化、盲法 |
| key_findings | result_table | ✓ | 核心发现：观测项、数值/趋势、统计显著性和效应量 |
| statistical_analysis | text | ✓ | 统计分析方法：检验名称、软件、显著性阈值、多重检验校正 |
| data_availability | text | | 数据可获取性： accession numbers、数据库、存储库 |
| reproducibility_notes | text | | 复现注意事项：关键实验条件、批次效应等。无则为 null |
| mechanistic_insight | text | | 对观察到的生物学机制层面的解释。无则为 null |
| contributions | list[str] | ✓ | 本研究的主要贡献 |
| limitations | list[str] | ✓ | 作者自述或明显可见的局限性 |

```
章节结构: 概览 → 假设与动机 → 研究体系 → 实验方法 → 实验设计
         → 核心发现 → 统计分析 → 机制解释 → 数据可用性 → 贡献与局限
```

#### 10.2.3 computational（计算工具型）

```
描述: "开发生物信息学工具、算法、pipeline 或数据库查询系统，核心贡献在计算层面"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 这个工具解决什么计算/分析需求？ |
| motivation | text | ✓ | 为什么需要这个工具？已有工具有何不足？ |
| algorithm_overview | text | ✓ | 算法/方法的整体描述 |
| input_output_spec | text | ✓ | 输入数据格式、输出结果格式 |
| software_architecture | text | | 软件架构。无则为 null |
| implementation | text | | 实现：编程语言、依赖、运行环境 |
| benchmark_datasets | text | ✓ | 基准测试使用的数据集 |
| performance_results | result_table | ✓ | 性能评测结果：准确性、速度、资源消耗 |
| comparisons_to_existing | result_table | | 与已有工具的对比 |
| availability | text | ✓ | 获取方式：GitHub URL、许可证、文档 |
| contributions | list[str] | ✓ | 主要贡献 |
| limitations | list[str] | ✓ | 工具局限性 |

```
章节结构: 概览 → 需求与动机 → 算法 → 输入输出 → 软件架构
         → 基准评测 → 工具对比 → 获取方式 → 贡献与局限
```

#### 10.2.4 review（综述型）

```
描述: "对生物学领域的文献进行系统梳理、整合或定量meta分析"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 本综述覆盖的生物学问题/领域 |
| scope | text | ✓ | 范围：纳入/排除标准、检索策略、覆盖时间 |
| key_insight | text | ✓ | 综述得出的最重要结论 |
| taxonomy_or_framework | text | ✓ | 组织框架/分类体系（若有） |
| key_papers_analyzed | structured_list | | 重点分析的论文及其贡献 |
| meta_analysis_methods | text | | Meta-analysis 统计方法（若适用） |
| meta_analysis_results | result_table | | Meta-analysis 定量结果（若适用） |
| trends | text | ✓ | 观察到的领域趋势 |
| open_questions | list[str] | ✓ | 未解决的关键问题 |
| contributions | list[str] | ✓ | 综述的主要贡献 |
| limitations | list[str] | ✓ | 综述的局限性 |

#### 10.2.5 data_resource（数据资源型）

```
描述: "发布大规模生物学数据集、基因组/蛋白质组资源、表达图谱等"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 这个数据资源填补了什么空白？ |
| data_description | text | ✓ | 数据的详细描述：类型、来源、覆盖范围 |
| generation_methods | text | ✓ | 数据生成/采集的实验和计算流程 |
| quality_control | text | ✓ | 质量控制：指标、过滤标准、验证方法 |
| data_statistics | key_value_table | ✓ | 数据规模统计 |
| access_and_format | text | ✓ | 访问方式、数据格式、API/下载链接 |
| validation_examples | text | | 数据验证：通过什么方式验证数据质量 |
| reuse_potential | text | | 数据的潜在应用场景 |
| contributions | list[str] | ✓ | 主要贡献 |
| limitations | list[str] | ✓ | 数据资源的局限性 |

#### 10.2.6 method_protocol（方法协议型）

```
描述: "详细描述一种实验技术或分析流程的具体操作步骤，类似Nature Protocols风格"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 这个协议解决什么实验需求？ |
| method_overview | text | ✓ | 方法概述：原理和适用范围 |
| protocol_steps | structured_list | ✓ | 关键步骤：步骤编号、操作描述、时间、关键注意事项 |
| required_materials | list[str] | ✓ | 所需材料、试剂、设备 |
| troubleshooting | key_value_table | | 常见问题 → 解决方案 |
| expected_results | text | | 预期结果和典型输出 |
| advantages_over_alternatives | text | | 相比替代方案的优劣 |
| applications | list[str] | | 已成功应用的场景 |
| limitations | list[str] | ✓ | 协议的局限性 |

#### 10.2.7 生物学共享字段

```
基础共享:
  problem, motivation, contributions, limitations

实验类共享 (experimental + data_resource):
  biological_system, experimental_methods, data_availability

计算类共享 (computational):
  algorithm_overview, performance_results, availability
```

---

### 10.3 材料学领域

材料学论文按**研究手段**划分：实验合成表征、计算模拟、理论建模是三个并列的研究范式。

#### 10.3.1 类型总览

| 类型 | 典型论文特征 | 类比 AI/ML |
|---|---|---|
| **experimental** | 合成+表征+性能测试 | experimental |
| **computational** | DFT/MD/ML 计算模拟材料性质 | experimental (but computational) |
| **review** | 某类材料/现象的系统综述 | survey |
| **theory** | 理论模型/本构关系/相场模型 | theoretical |
| **data_benchmark** | 材料数据库/高通量筛选/ML数据集 | benchmark |

#### 10.3.2 experimental（实验合成表征型）

```
描述: "通过实验手段合成材料并进行结构表征和性能测试"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 研究什么材料/解决什么材料学问题？ |
| motivation | text | ✓ | 为什么这个材料/方法重要？ |
| material_system | text | ✓ | 材料体系：化学组成、晶体结构、相 |
| synthesis_method | text | ✓ | 合成方法：原料、步骤、关键条件（温度、压力、时间） |
| processing_conditions | key_value_table | | 加工条件参数 |
| characterization_techniques | structured_list | ✓ | 表征技术：每项技术的名称、用途、关键参数、发现 |
| structure_properties | text | ✓ | 结构特征：微观结构、相组成、形貌 |
| performance_metrics | result_table | ✓ | 性能指标：测试条件、数值、与文献对比 |
| structure_property_relationship | text | | 结构-性能关系讨论 |
| contributions | list[str] | ✓ | 主要贡献 |
| limitations | list[str] | ✓ | 局限性 |

```
章节结构: 概览 → 动机 → 材料体系 → 合成方法 → 表征技术
         → 结构与性能 → 构效关系 → 贡献与局限
```

#### 10.3.3 computational（计算模拟型）

```
描述: "通过第一性原理、分子动力学、相场模拟或机器学习等计算方法研究材料性质"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 计算研究解决什么材料学问题？ |
| motivation | text | ✓ | 为什么需要计算方法？实验的局限是什么？ |
| computational_method | text | ✓ | 计算方法：DFT/ MD / Phase-field / ML 等的具体设置 |
| software_package | text | | 使用的软件包和版本 |
| model_system | text | ✓ | 模拟体系：原子数、超胞大小、边界条件 |
| computational_parameters | key_value_table | ✓ | 关键计算参数：泛函、赝势、截断能、k点、力场等 |
| calculated_properties | result_table | ✓ | 计算得到的性质与数值 |
| validation | text | | 计算方法验证：与实验或其他计算的对比 |
| mechanistic_insight | text | | 从计算结果获得的原子/电子层面的机制理解 |
| contributions | list[str] | ✓ | 主要贡献 |
| limitations | list[str] | ✓ | 计算方法的局限性 |

```
章节结构: 概览 → 动机 → 计算方法 → 模拟体系 → 计算参数
         → 计算结果 → 验证 → 机制理解 → 贡献与局限
```

#### 10.3.4 review（综述型）

```
描述: "对某类材料或某类现象的研究进展进行系统总结"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 综述覆盖的材料/现象 |
| scope | text | ✓ | 范围界定 |
| taxonomy_or_framework | text | ✓ | 分类框架：按材料类型/机制/方法分类 |
| key_advances | structured_list | | 关键研究进展 |
| comparative_analysis | result_table | | 不同材料/方法的性能对比 |
| challenges | list[str] | ✓ | 当前面临的挑战 |
| future_directions | list[str] | | 未来方向 |
| contributions | list[str] | ✓ | 综述贡献 |
| limitations | list[str] | ✓ | 局限性 |

#### 10.3.5 theory（理论模型型）

```
描述: "提出新的理论模型、本构关系或分析框架来描述和预测材料行为"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 理论模型要描述什么现象？ |
| motivation | text | ✓ | 现有理论有什么不足？ |
| theoretical_framework | text | ✓ | 理论基础：使用的理论工具和分析框架 |
| key_definitions | key_value_table | ✓ | 关键定义和符号体系 |
| model_equations | formula_table | ✓ | 模型方程：表达式、物理含义、适用范围 |
| assumptions | list[str] | ✓ | 模型假设条件 |
| model_predictions | result_table | | 模型预测值 vs 实验/模拟数据 |
| validation | text | | 模型验证方式 |
| applicable_range | text | | 模型适用范围和限制 |
| contributions | list[str] | ✓ | 主要贡献 |
| limitations | list[str] | ✓ | 局限性 |

#### 10.3.6 data_benchmark（数据/基准型）

```
描述: "发布材料数据库、高通量筛选结果或机器学习基准数据集"
```

| 字段 | kind | 必填 | prompt |
|---|---|---|---|
| problem | text | ✓ | 这个数据资源解决什么需求？ |
| data_description | text | ✓ | 数据描述：材料类型、性质、来源 |
| data_generation | text | ✓ | 数据生成：实验 or 计算 or 文献挖掘 |
| data_statistics | key_value_table | ✓ | 数据统计：条目数、特征维度、分布 |
| quality_metrics | text | | 数据质量评估 |
| benchmark_results | result_table | | 在数据集上的基线模型结果（若适用） |
| access_and_usage | text | ✓ | 访问方式和使用说明 |
| contributions | list[str] | ✓ | 主要贡献 |
| limitations | list[str] | ✓ | 局限性 |

#### 10.3.7 材料学共享字段

```
基础共享:
  problem, motivation, contributions, limitations

实验类共享 (experimental):
  synthesis_method, characterization_techniques, structure_properties

计算类共享 (computational + data_benchmark):
  computational_parameters, calculated_properties

理论类共享 (theory):
  theoretical_framework, model_equations, assumptions
```

---

### 10.4 跨领域共性抽象

三个领域虽然术语不同，但存在**结构性的共性**。识别这些共性可以指导 `FieldDef.kind` 的设计：

| 共性模式 | AI/ML | 生物学 | 材料学 | kind 复用 |
|---|---|---|---|---|
| 问题/动机 | problem, motivation | problem, motivation, hypothesis | problem, motivation | `text` |
| "方法"描述 | architecture_overview | experimental_methods | synthesis_method / computational_method | `text` + `structured_list` |
| 结构化方法列表 | components | experimental_methods | characterization_techniques | `structured_list` |
| 公式/方程 | formulas | (in computational) | model_equations | `formula_table` |
| 结果数据 | main_results | key_findings / performance_results | performance_metrics / calculated_properties | `result_table` |
| 数值统计 | (in benchmark) | data_statistics | data_statistics | `key_value_table` |
| 定义列表 | key_definitions | (n/a) | key_definitions | `key_value_table` |
| 简单列表 | contributions, limitations | contributions, limitations | contributions, limitations | `list[str]` |

**设计启示**：现有的 7 种 `field kind` 足以覆盖三个领域的所有需求。不需要增加新的 kind。跨领域扩展时只需编写新的 `FieldDef` 和 `SectionDef` 组合，不需要修改渲染引擎。

---

### 10.5 领域选择机制

```
优先级:
  1. DOMAIN 环境变量显式指定（如 DOMAIN=biology）
  2. 领域自动检测：一次 LLM 调用判断论文所属领域
     输入: title + abstract[:1000]
     输出: {"domain": "ai_ml", "confidence": 0.95}
  3. 默认: ai_ml
```

领域检测 prompt：
```
根据论文标题和摘要，判断这篇论文属于哪个学科领域。
候选领域:
- ai_ml: 人工智能、机器学习、计算机视觉、NLP
- biology: 分子生物学、遗传学、生物化学、生物信息学
- materials_science: 材料合成、表征、计算材料学、材料理论

返回 JSON: {"domain": "...", "confidence": 0.0-1.0, "reasoning": "..."}
```

---

## 11. Pipeline 变更

### 11.1 更新后的 V3 Pipeline

```
Phase 1: 论文解析 (不变)
    │
    ▼
Phase 1.2: 领域检测 [新增]（若未通过 DOMAIN 环境变量指定）
    │  轻量 LLM 调用: title + abstract → domain
    │  加载 DomainProfile
    │
    ▼
Phase 1.3: 论文类型检测 [新增]
    │  轻量 LLM 调用: title + abstract → paper_type
    │  加载 PaperTypeProfile (= schema)
    │
    ▼
Phase 1.5: 结构化理解 (修改)
    │  Prompt 从 schema.fields 生成，不再硬编码
    │  LLM 输出按 schema.fields 解析为 dict
    │
    ▼
Phase 2-4: 引用挖掘 + 关键论文分析 (不变)
    │
    ▼
Phase 5: 技术路线分析 [可选，V3_ROUTE_ANALYSIS_ENABLED=1] (不变)
    │
    ▼
Phase 6: Markdown 导出 (修改)
    │  章节结构 + 渲染逻辑由 schema.sections 驱动
    │  空章节静默跳过
    │  §6 (领域技术路线) 和 §7 (对比分析) 仅完整分析模式输出
    │  后置翻译不变
```

### 11.2 与运行模式的关系

Schema 驱动架构与两种运行模式（基础/完整）正交：

| 模式 | 使用的 Schema 部分 | 输出章节 |
|---|---|---|
| 基础分析 | `PaperTypeProfile.fields` → prompt 生成 | `PaperTypeProfile.sections` 中排除 `route_analysis` 和 `comparison` 条件章节 |
| 完整分析 | 同基础分析 | 全部 `sections`，含条件章节 |

技术路线分析和对比分析的章节可通过 `SectionDef.condition` 控制：
```python
SectionDef(name="route_analysis", level=1, condition="routes is not None", ...)
SectionDef(name="comparison", level=1, condition="comparison is not None", ...)
```

当 `V3_ROUTE_ANALYSIS_ENABLED=0` 时，`routes` 和 `comparison` 为 `None`，对应章节自动跳过。

### 11.3 Pipeline 代码差异（概念）

```python
# 当前:
structured = analyze_paper_structure(seed_paper, llm_client)
export_markdown(seed_paper, routes, comparison, all_refs, en_path, lang="en")

# Schema 驱动后:
domain = get_domain(config.DOMAIN) if config.DOMAIN else detect_domain(paper, client)
paper_type = detect_paper_type(paper, domain, client)   # Phase 1.3
schema = domain.get_paper_type(paper_type)               # 取 PaperTypeProfile
structured = analyze_with_schema(paper, schema, client)  # Phase 1.5: schema.fields → prompt
export_markdown(paper, structured, routes, comparison, all_refs, schema, en_path, lang="en")
# Phase 6: schema.sections → 章节结构
```

---

## 12. 迁移路径

> **当前状态**: 全部未实施。当前 V3 代码按 `design_v3.md` 描述的硬编码方式运行。

### Phase A：Schema 基础设施（无行为变化）`[未开始]`
1. 创建 `src/domains/base.py` — `FieldDef`, `SectionDef`, `PaperTypeProfile`, `DomainProfile`
2. 创建 `src/domains/ai_ml.py` — AI/ML 领域，仅含一种论文类型："experimental"
3. "experimental" schema 与当前硬编码的 `StructuredUnderstanding` 字段 1:1 对应
4. 所有现有测试继续通过 — schema 只是同一事物的另一种表示

### Phase B：将 Schema 接入 Pipeline `[未开始]`
5. 修改 `structured_analyzer.py`，接受 `PaperTypeProfile` 并从中生成 prompt
6. 修改 `markdown_exporter_v3.py`，接受 `PaperTypeProfile` 并按章节定义渲染
7. 测试：验证 "experimental" 类型的输出与当前硬编码版本完全一致

### Phase C：加入论文类型检测 `[未开始]`
8. 创建 `paper_type_detector.py`
9. 在 `run_v3.py` Phase 1.5 之前接入
10. 为 AI/ML 领域添加 "theoretical"、"survey"、"benchmark"、"system"、"position" 论文类型
11. 测试：每种类型产生不同结构的 markdown

### Phase D：多领域支持 `[未开始]`
12. 添加 `DOMAIN` 配置支持和领域自动检测
13. 创建 `src/domains/biology.py` — 生物学 5 种论文类型
14. 创建 `src/domains/materials_science.py` — 材料学 5 种论文类型
15. 测试：跨领域论文产生对应结构的 markdown

---

## 13. 权衡与待讨论问题

### 权衡

| 决策 | 优势 | 代价 |
|---|---|---|
| Dict 作为数据容器 | 最大灵活性 | 失去类型化属性访问 |
| Schema 驱动渲染 | 单一真相源 | 多一层间接引用，难以手工微调特殊格式 |
| 分析前做类型检测 | 适配论文类型 | 多一次 LLM 调用 (约 ¥0.0001) |
| Domain profile 用 Python 文件 | 类型检查，无需解析 | 非开发者难以添加领域 |
| 16 种论文类型预设计 | 覆盖面广，开箱即用 | 初期只用到其中几种，维护负担 |

### 待讨论问题

1. **论文类型粒度**：当前每个领域 5-6 种类型。是否需要子类型（如 AI/ML 中 "nlp_experimental" vs "cv_experimental"）？目前判断不需要——这些在术语上不同但分析结构相同。

2. **混合类型论文**：有些论文跨类型（既有理论又有实验，既是 benchmark 又提出新方法）。检测按主导模式分类。LLM 在分析时会自然处理次要维度——如果一篇 benchmark 论文也提出了简单基线方法，LLM 会在 `baseline_methods` 中自然描述。

3. **Schema 版本管理**：schema 演进时旧数据可能不匹配。方案：输出中存储 `schema_version`，读取旧数据时优雅降级。

4. **关键论文的分析**：Phase 4 对引用论文也做结构化分析。这些论文的类型也需要检测，与种子论文相同。类型检测开销=引用论文数 × 1次轻量 LLM 调用，可接受。

5. **技术路线分析(Phase 5)的兼容性**：当前 `route_analyzer.py` 依赖 `architecture_overview` 和 `key_insight` 字段。对于非 experimental 类型的论文，需要适配——schema 中可以标注哪些字段是"路线分析的输入字段"。

---

## 14. 小结

Schema 驱动架构将三个硬编码、独立维护的组件替换为单一 schema 定义。本文档预设计了 **3 个领域、16 种论文类型**的完整分析维度：

| 领域 | 类型数 | 类型 |
|---|---|---|
| AI/ML | 6 | experimental, benchmark, system, theoretical, survey, position |
| 生物学 | 5 | experimental, computational, review, data_resource, method_protocol |
| 材料学 | 5 | experimental, computational, review, theory, data_benchmark |

关键设计决策：
- **7 种 field kind 覆盖所有需求**，不需要新增渲染器能力
- **基础共享字段**（problem, motivation, contributions, limitations）保证最小可用性
- **类型特定字段**让 LLM 按论文的实际性质提取信息，不再"一刀切"
- **领域 + 类型两级检测**，开销为 2 次轻量 LLM 调用（约 ¥0.0002）

---

## 15. 理解深度改进（基于与人工解析的差异分析）

通过对两篇论文（ParkingWorld 和 MindVLA-U1）的工具输出与专业公众号解析文章进行系统对比，
发现了 Schema 驱动架构在**理解深度**上的几个结构性缺失。本节记录改进方案。

### 15.1 差异分析来源

| # | 论文 | 工具输出 | 公众号文章 |
|---|------|---------|-----------|
| 1 | ParkingWorld (2605.25029) | `output/v3/2026-06-02_18-03-45_ParkingWorld.../` | 自动驾驶之心公众号（搜狐转载） |
| 2 | MindVLA-U1 (2605.12624) | `output/v3/2026-06-03_12-01-24_MindVLA-U1.../` | 雷锋网/知乎/自动驾驶之心等 |

### 15.2 缺失的理解维度

通过对比发现，工具分析在以下维度上存在结构性缺失：

#### 维度 1：设计原理（Design Rationale）— "为什么"而非"是什么"

| 论文 | 工具输出（描述 WHAT） | 公众号输出（解释 WHY） |
|------|---------------------|----------------------|
| ParkingWorld | "3DGS 模拟器提供逼真的、物理交互的环境" | **为什么用 3DGS**？— CARLA/LGSVL 与真实世界存在巨大视觉域差距，3DGS 创建真实场景的数字孪生，策略可实现零样本 sim-to-real 迁移 |
| ParkingWorld | "使用自编码器将特权真实信息编码为潜在 critic 特征" | **为什么使用特权自编码器 critic**？— 人类干预样本质量高但可能造成 Q 值过高估计并传播到附近 OOD 动作，层归一化限制 Q 值幅度 |
| ParkingWorld | "多级回放缓冲区以结构化方式存储和采样转换" | **为什么是"错题本"设计**？— 耦合失败+纠正让策略看到错误及正确方式，比简单混合人类演示更有效 |
| MindVLA-U1 | "基于 MoT 架构，包含密集（快系统）和稀疏（慢系统）变体" | **为什么用 MoT 快速/慢速系统**？— 一个大脑两种形态：慢速用于复杂场景推理（激活全部专家组），快速仅激活动作专家以达到 VA 级延迟（9.7 FPS vs 0.39 FPS） |
| MindVLA-U1 | "使用流匹配生成连续空间中的动作轨迹" | **为什么用流匹配**？— 避免将连续轨迹离散化为语言 token 导致的精度损失，保持动作的自然连续形式 |

**根因**：当前 schema 的字段 prompt 只问"描述 X"，不问"为什么这样设计 X"、"考虑了哪些替代方案"、"权衡是什么"。

#### 维度 2：相关工作上下文（Related Work Context）

公众号文章会将论文放置在相关工作背景下进行解读：
- ParkingWorld 与 HIL-SERL（操作领域）、REAP（之前的 3DGS 泊车工作）、RAD（大规模 3DGS 驾驶 RL）的关系
- MindVLA-U1 与 RAP、OneVL、EMMA 的对比，以及为什么"VLA 落后于 VA 不是范式问题而是接口设计问题"

当前 schema **完全没有 `related_work_context` 字段**。这导致分析缺少论文贡献的"坐标系"——读者无法判断该工作的相对定位。

#### 维度 3：直观类比（Intuitive Framing）

公众号文章会使用"错题本"来解释 CIL-SERL、"一个大脑两种形态"来解释 MoT。这些类比是理解的关键桥梁。

当前工具给出技术性描述（"将失败的自主 rollout 与成功的人类纠正耦合"），但缺少直观类比。

#### 维度 4：数据工程意识（Data Engineering）

MindVLA-U1 的 MindLabel 自动标注系统（3.8M VQA 对、20 类意图、~250K 梦想轨迹、CoT 推理链）是该工作的重要工程贡献，
但当前 schema 没有字段捕获数据标注/管理方面的贡献。

#### 维度 5：训练的阶段分解

公众号文章将 MindVLA-U1 的训练分解为四个阶段：
1. 数据准备（日志→视频+标注）
2. 联合预训练（理解+轨迹预测）
3. 记忆优化（梯度跨帧传播）
4. RL 微调（以 RFS 为奖励信号）

当前 `training_procedure` 字段是一个文本块，LLM 倾向于将其压缩为一段话，丢失了阶段化结构。

#### 维度 6：消融的机制解释

当前 `ablation_results` 是 `list[str]`，每项只是发现的一句话摘要。缺少：
- 消融的**目的**（验证什么假设）
- 观察到的**机制**（为什么移除/修改组件产生了特定效果）

### 15.3 公式与图片的呈现方式问题

当前架构将公式和架构图描述作为**独立章节**输出（§3.4 关键公式、§3.2 架构图详解）。
这割裂了公式/图片与其讲解上下文的关联，导致：
- 公式以表格形式集中列出，缺乏与方法的自然融合
- 读者无法在看到架构讲解时同步看到相关公式
- 图片描述被单独放置在 §3.2，而非嵌入到组件讲解中

**根因**：`SectionDef` 将公式作为顶层独立字段放在 `components` 和 `training` 之间，
渲染时将公式表格插入到固定位置，而非根据论文的叙述顺序自然融入。

**改进方向**：
1. 公式和图片应作为**方法讲解的引导线索**，而非独立列表
2. `architecture_overview` 中应要求 LLM 自然地包含关键公式（使用 LaTeX）
3. `training_procedure` 中应要求 LLM 自然地包含损失函数公式
4. 架构图描述应融入组件讲解中，而非单独放置
5. 可保留公式的结构化提取数据（用于程序化访问），但在 markdown 渲染时应**按上下文自然散布**，而非集中表格

### 15.4 Schema 改进方案

#### 新增字段

```python
# ── 新增：设计原理 ──
_F_DESIGN_RATIONALE = FieldDef(
    name="design_rationale", kind="text",
    label_en="Design Rationale", label_zh="设计原理",
    prompt="For each key design choice in the method, explain WHY it was made: "
           "what problem did it solve? what alternatives were considered? "
           "what trade-offs were involved? Include formulas and figure "
           "references where relevant to explain the design.",
    required=False,
)

# ── 新增：相关工作上下文 ──
_F_RELATED_WORK_CONTEXT = FieldDef(
    name="related_work_context", kind="text",
    label_en="Related Work Context", label_zh="相关工作上下文",
    prompt="How does this work relate to and differ from prior approaches? "
           "Position it within the research landscape: what specific gap "
           "does it fill? What prior work does it directly build upon or "
           "contrast with?",
    required=False,
)

# ── 新增：数据工程 ──
_F_DATA_ENGINEERING = FieldDef(
    name="data_engineering", kind="text",
    label_en="Data Engineering", label_zh="数据工程",
    prompt="Describe any data collection, annotation, curation, or generation "
           "methodology. Include scale, quality control, labeling pipeline, "
           "and any automated annotation systems. Return null if not a "
           "significant contribution of this paper.",
    required=False,
)

# ── 新增：训练阶段分解（替换 flat training_procedure 文本） ──
_F_TRAINING_STAGES = FieldDef(
    name="training_stages", kind="structured_list",
    label_en="Training Stages", label_zh="训练阶段",
    prompt="Break down the training process into distinct stages. "
           "For each stage: title, description (include loss functions, "
           "optimization, and key hyper-parameters), and purpose.",
    required=False,
    columns=[
        ColumnDef(name="title", label_en="Stage", label_zh="阶段"),
        ColumnDef(name="description", label_en="Description", label_zh="描述"),
        ColumnDef(name="papers", label_en="Key Details", label_zh="关键参数"),
    ],
)

# ── 新增：直观类比 ──
_F_INTUITIVE_ANALOGY = FieldDef(
    name="intuitive_analogy", kind="text",
    label_en="Intuitive Analogy", label_zh="直观类比",
    prompt="If the paper uses an intuitive analogy or metaphor to explain "
           "its method (e.g., 'mistake notebook', 'two-system brain'), "
           "capture it here. Return null if none.",
    required=False,
)

# ── 新增：部署架构 ──
_F_DEPLOYMENT_ARCHITECTURE = FieldDef(
    name="deployment_architecture", kind="text",
    label_en="Deployment Architecture", label_zh="部署架构",
    prompt="If the paper involves real-world deployment, describe the "
           "deployment architecture: hardware setup, inference pipeline, "
           "model-server hierarchy, latency breakdown, and sim-to-real "
           "transfer mechanism. Return null if not applicable.",
    required=False,
)
```

#### 增强现有字段的 prompt（从"描述 WHAT"升级为"解释 WHY"）

```python
# architecture_overview — 增强：要求自然地包含公式
_F_ARCH_OVERVIEW.prompt = (
    "Describe the overall method/architecture including data flow. "
    "Integrate key mathematical formulas naturally into the explanation "
    "using LaTeX notation ($...$). Explain the RATIONALE behind each "
    "major design choice — why was it designed this way? "
    "Reference relevant figures and their roles."
)

# components — 增强：要求解释设计原因
_F_COMPONENTS.prompt = (
    "List every major architectural component with its name, purpose, "
    "implementation details (concrete numbers: dimensions, layer configs, "
    "parameter counts, etc.), referenced figure, AND the DESIGN RATIONALE "
    "— why was this component designed this way? What problem does it solve?"
)

# training_procedure — 增强：要求阶段分解和内联公式
_F_TRAINING_PROCEDURE.prompt = (
    "Describe the training procedure in detail, broken down by STAGES if "
    "multi-stage training is used. For each stage: what is trained, what "
    "losses are used (include formulas in LaTeX), what optimizer/schedule, "
    "what data, and WHY this stage is needed. Include data augmentation, "
    "key tricks, and hyper-parameters."
)

# ablation_results — 增强：要求机制解释
_F_ABLATION_RESULTS.prompt = (
    "For each ablation study, describe: (1) what was ablated and why this "
    "tests an important hypothesis, (2) the quantitative result, and "
    "(3) the MECHANISM — why did removing/modifying this component produce "
    "this effect? What does this reveal about the method?"
)
```

#### 章节结构调整

公式不再作为独立章节（§3.4），而是融入方法讲解中：

```python
# 修改前（公式独立章节）：
SectionDef(name="formulas", level=2, ... fields=["formulas"])
SectionDef(name="training", level=2, ... fields=["training_data", "loss_functions", ...])

# 修改后（公式融入叙述中）：
SectionDef(name="arch_overview", level=2, ...
    fields=["architecture_overview"])           # 包含内联公式
SectionDef(name="arch_figure", level=2, ...
    fields=["architecture_figure"])             # 融入组件讨论
SectionDef(name="components", level=2, ...
    fields=["components"])                      # 每个组件包含设计原理
SectionDef(name="design_rationale", level=2, ...
    fields=["design_rationale", "intuitive_analogy"])  # 新增：设计原理+类比
SectionDef(name="training", level=2, ...
    fields=["training_data", "training_stages",
            "loss_functions", "optimizer", "training_procedure"])
# formulas 字段保留在 fields 列表中用于提取（`get_field("formulas")` 仍可用），
# 但不在 sections 中作为独立章节渲染
```

**关键变化**：
- 移除 `formulas` 作为独立渲染章节 — 公式数据仍被提取，但在 `architecture_overview` 和 `training_procedure` 中以内联方式自然呈现
- 新增 `design_rationale` + `intuitive_analogy` 章节（§3 下作为 §3.x）
- 新增 `related_work_context` 章节（§2 下或独立为 §2.3）
- `training_stages` 提供结构化的阶段分解

### 15.5 错误处理与韧性改进

从 MindVLA-U1 分析中发现的 `full_text=null` 导致的静默降级问题：

#### 问题
1. PDF 下载失败后，流水线以摘要继续分析，**未向用户发出任何警告**
2. `limitations` 中将缺失归因为"论文未讨论"，而非标记数据提取失败
3. 空 sections 被跳过导致章节编号不连续（§3.3 → §3.5，缺失 §3.4）

#### 改进方案

```python
# 1. full_text 不可用时，在输出顶部添加警告横幅
if not paper.full_text:
    lines.insert(0, "> ⚠️ Full text unavailable — analysis based on abstract only. "
                     "Detailed implementation, formulas, and training details may be missing.")

# 2. full_text 不可用时，自动添加一条 limitations
if not paper.full_text:
    structured.setdefault("limitations", []).insert(0,
        "Full paper text was not available during analysis. "
        "Detailed implementation specs, formulas, and training procedures "
        "may be missing from this report."
    )

# 3. 空 sections 渲染占位符而非跳过，保持编号连续
# SectionDef.always_show = True 时，即使字段为空也渲染标题 + "(not available)"
```

#### _parse_metadata_from_text 作者提取启发式修复

当前逗号分隔的作者检测会错误匹配到摘要中的逗号分隔短语。
改进方案：

```python
# 优先在标题之后查找作者行（标题后 1-5 行内），而非扫描前 30 行找逗号
# 1. 先找标题行号
# 2. 在标题后 1-5 行内查找作者候选行（包含多个大写字母开头的单词）
# 3. 对候选行使用更严格的作者名正则验证
# 4. 如果都失败，回退到原来的逗号启发式（并记录警告）
```

### 15.6 改进的优先级与实施顺序

| 优先级 | 改进项 | 影响范围 | 涉及文件 |
|--------|--------|---------|---------|
| P0 | 公式融入叙述 + 移除独立公式章节 | 渲染结构 | `ai_ml.py`, `markdown_exporter_v3.py` |
| P0 | 新增 `design_rationale` + `related_work_context` 字段 | Schema + Prompt | `ai_ml.py`, `structured_analyzer.py` |
| P1 | 增强现有字段 prompt（WHAT → WHY） | Prompt 质量 | `ai_ml.py` |
| P1 | 增强 ablation prompt 要求机制解释 | Prompt 质量 | `ai_ml.py` |
| P1 | full_text=null 警告横幅 + 自动 limitations | 错误处理 | `markdown_exporter_v3.py`, `run_v3.py` |
| P2 | 新增 `training_stages`, `data_engineering`, `deployment_architecture`, `intuitive_analogy` | Schema | `ai_ml.py` |
| P2 | 空 section 渲染占位符保持编号连续 | 渲染 | `markdown_exporter_v3.py` |
| P3 | _parse_metadata_from_text 作者提取修复 | 鲁棒性 | `paper_resolver.py` |
| P3 | PDF 下载重试机制 | 鲁棒性 | `paper_resolver.py` |

### 15.7 实施记录

**实施日期**: 2026-06-03

**已实施**:
- [ ] P0: 公式融入叙述 + 移除独立公式渲染章节
- [ ] P0: 新增 `design_rationale` + `related_work_context` 字段
- [ ] P1: 增强现有字段 prompt（WHAT → WHY + 机制解释）
- [ ] P1: full_text=null 警告横幅 + 自动 limitations
- [ ] P2: 新增 4 个辅助字段
- [ ] P2: 空 section 占位符
- [ ] P3: 作者提取启发式修复
