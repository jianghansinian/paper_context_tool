在我看来，这份文档已经不是简单的设计补充，而是对整个产品定位的一次升级。

你最初想做的是：

```text
Paper Graph
```

后来变成：

```text
Research Evolution Engine
```

而经过这次 Review 和进一步推演，其实已经演化成：

```text
Research Narrative Engine
```

这三个阶段的本质差异非常大。

下面是整理后的 V4 思考文档，你可以直接保存为：

```text
design_v4_research_narrative_engine.md
```

# Research Narrative Engine V4

## 从 Paper Graph 到 Research Narrative

### 第一阶段：Paper Graph

典型代表：

* Google Scholar
* Semantic Scholar
* Connected Papers

核心能力：

```text
Paper
↓
Related Papers
```

解决的问题：

```text
找到相关论文
```

无法解决：

```text
为什么这些论文重要

这些论文之间是什么关系

这个领域是如何演化的
```

---

### 第二阶段：Research Evolution Engine

核心思想：

```text
Paper
↓
Problem
↓
Solution
↓
Evolution
```

相比 Citation Graph：

```text
谁引用了谁
```

更关注：

```text
谁解决了什么问题
```

已经开始接近技术路线分析。

但仍然存在问题：

```text
用户最终并不关心 Graph

用户关心的是：
这个领域为什么发展成今天这样
```

---

### 第三阶段：Research Narrative Engine

新的核心目标：

```text
帮助用户理解一个领域的发展故事
```

即：

```text
Research Story
```

用户真正的问题是：

```text
BEV为什么会出现？

为什么BEVDepth会出现？

为什么BEVFormer会流行？

为什么Sparse4D能够替代部分方案？
```

本质上：

```text
用户需要的是 Narrative

而不是 Graph
```

---

# 第一性原理

技术路线的本质：

不是 Citation

不是 Embedding

不是 Cluster

而是：

```text
Problem Evolution
```

进一步说：

```text
Research Claim Evolution
```

---

# Problem 与 Branch

V2设计中的问题：

```text
Temporal Modeling
```

同时承担了：

```text
Branch

Problem
```

两种角色。

这是错误的。

---

正确结构：

```text
Field
 └── Branch
       └── Problem Sequence
              └── Papers
```

示例：

Field：

```text
BEV Perception
```

Branch：

```text
Temporal BEV Modeling
```

Problems：

```text
P1:
如何融合多帧BEV特征

P2:
如何降低时序计算成本

P3:
如何与Planning联合建模
```

对应论文：

```text
BEVFormer
↓
Sparse4D
↓
SparseDrive
```

---

# 技术路线不是树

很多论文工具默认：

```text
Tree
```

实际上：

```text
DAG
```

更加符合真实科研发展。

示例：

```text
LSS
├── BEVDepth
└── BEVFormer

BEVDepth
└── SparseBEV

BEVFormer
└── Sparse4D

SparseBEV
      \
       ─── SparseDrive
      /
Sparse4D
```

因此：

```text
Research Evolution Graph
```

应设计为：

```text
Semantic DAG
```

而非简单树结构。

---

# 为什么 Problem 仍然不够

Problem：

```text
如何建模时序
```

BEVFormer：

```text
Problem:
如何建模时序
```

Sparse4D：

```text
Problem:
如何建模时序
```

Problem没有变化。

真正变化的是：

```text
解决问题的方法
```

即：

```text
Claim
```

---

BEVFormer：

```text
Claim:
Transformer Temporal Attention
能够有效利用历史帧信息
```

Sparse4D：

```text
Claim:
Sparse Query
能够以更低成本实现时序建模
```

---

因此未来核心对象应为：

```text
Paper
↓
Problem
↓
Claim
↓
Evidence
↓
Evolution
```

---

# 系统架构

```text
Seed Paper
↓
Paper Retrieval
↓
Paper Understanding
↓
Problem Extraction
↓
Claim Extraction
↓
Branch Discovery
↓
Evolution Graph Builder
↓
Narrative Builder
↓
Markdown Report
```

---

# 模块设计

## paper_retriever.py

目标：

构建候选论文池。

策略：

### 第一阶段

Citation Expansion

```text
Backward Citation
+
Forward Citation
```

目标：

```text
50~100篇
```

高精度论文。

---

### 第二阶段

Problem-based Semantic Search

使用：

```text
Extracted Problem
```

作为查询。

补充遗漏分支。

最终：

```text
80~120篇
```

候选论文。

---

# paper_understanding.py

复用现有V3能力。

已有内容：

```text
Motivation

Problem

Method

Components

Experiments

Limitations

Figures

Formulas
```

不重新发明轮子。

---

# problem_extractor.py

输出：

```json
{
  "problem":
  "...",

  "motivation":
  "...",

  "limitations":
  "..."
}
```

优先使用：

```text
Introduction

Conclusion
```

而非仅Abstract。

---

# claim_extractor.py

新增核心模块。

输出：

```json
{
  "claim":
  "...",

  "evidence":
  "...",

  "improvement_over_previous":
  "..."
}
```

示例：

```json
{
  "claim":
  "Sparse queries reduce temporal modeling cost",

  "evidence":
  "+3% NDS with 40% lower FLOPs"
}
```

---

# branch_discovery.py

目标：

自动发现研究方向。

输入：

```text
Problem
+
Claim
```

而不是：

```text
Embedding
```

单独聚类。

输出：

```text
Branch A

Temporal Modeling

Branch B

Depth Estimation

Branch C

Query-based BEV
```

---

# evolution_graph_builder.py

目标：

构建：

```text
Research Evolution DAG
```

节点：

```text
Claim
```

边：

```text
Improves

Extends

Replaces

Combines
```

示例：

```text
Dense Temporal Attention
↓
Sparse Temporal Attention
↓
Planning-aware Temporal Modeling
```

---

# narrative_builder.py

V4最重要模块。

输入：

```text
Branch

Problem Chain

Claim Chain
```

输出：

```text
Research Story
```

示例：

BEV最初通过Lift-Splat-Shoot解决多视角投影问题。

然而深度估计误差较大。

BEVDepth通过引入激光雷达监督提升深度质量。

随后BEVFormer利用Transformer实现时序建模，将历史信息引入BEV空间。

Sparse4D进一步发现Dense Query计算开销过高，因此提出Sparse Query机制降低成本。

最终SparseDrive将感知与规划统一建模。

````

---

# markdown_report_builder.py

最终输出：

```markdown
# Field

BEV Perception

## Branch

Temporal Modeling

## Research Story

...

## Key Problems

...

## Key Papers

...

## Evolution Timeline

...
````

---

# 为什么 Narrative 是护城河

已有产品：

```text
Google Scholar

Semantic Scholar

Connected Papers
```

解决：

```text
Paper Discovery
```

---

未来大量Agent：

```text
ChatGPT

Claude

Gemini

Kimi
```

解决：

```text
Paper Understanding
```

---

Research Narrative Engine：

解决：

```text
Research Understanding
```

即：

为什么这个领域会发展成今天这样。

---

# V4验证目标

不要先验证自动构图。

不要先验证自动聚类。

优先验证：

```text
Narrative Generation
```

实验：

选择：

```text
BEV
```

人工指定：

```text
20篇关键论文
```

然后仅验证：

```text
Problem Extraction

Claim Extraction

Narrative Generation
```

输出：

```text
BEV发展史
```

如果自动驾驶工程师阅读后认为：

“这基本符合我脑中的技术路线”

则说明方向成立。

---

# 最终愿景

不是：

```text
Paper Search Engine
```

不是：

```text
Paper Graph Tool
```

而是：

```text
Research Narrative Engine
```

成为研究人员理解一个领域的默认入口。

帮助用户回答：

为什么？

发生了什么？

下一步会走向哪里？

```
```
