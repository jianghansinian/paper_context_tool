# Research Evolution Engine V2 设计文档

## 项目背景

现有论文工具主要解决以下问题：

* 找论文（Paper Discovery）
* 读论文（Paper Reading）
* 写论文（Paper Writing）

但对于研究人员和工程师而言，还有一个更重要的问题：

> 如何快速理解一个领域的技术发展路线（Research Evolution）？

例如：

* BEV Perception 是如何发展的？
* Diffusion Models 为什么会出现？
* VLA 与传统 Robotics 方法的核心区别是什么？
* Sparse4D 为什么会在 BEVFormer 之后出现？

用户真正关心的并不是：

```text
这篇论文引用了谁？
```

而是：

```text
这篇论文解决了什么问题？

为什么会出现？

相比前人的方法改进了什么？

后来又被哪些工作继承？
```

因此本项目不再以 Citation Graph 为核心，而是以：

```text
Problem
↓
Solution
↓
Evolution
```

作为核心建模对象。

---

# 第一性原理

技术路线并不是论文之间的关系。

技术路线实际上是：

```text
Problem Evolution
```

例如：

BEV Perception

```text
问题1：
如何从多相机构建BEV

├─ Lift-Splat-Shoot
├─ BEVDet

问题2：
如何提升深度估计精度

├─ BEVDepth
├─ MatrixVT

问题3：
如何利用时序信息

├─ BEVFormer
├─ Sparse4D
├─ SparseDrive
```

用户真正想获得的是：

```text
问题是如何被提出的

问题是如何被解决的

后续工作如何持续优化
```

而不是简单的论文引用网络。

---

# 产品目标

输入：

```text
一篇论文
或
一个研究领域
```

输出：

```text
Field
├─ Branch
│  ├─ Problem
│  ├─ Key Papers
│  └─ Evolution Timeline
```

示例：

```text
Field:
BEV Perception

Branch:
Temporal Modeling

Problem:
如何利用时序信息提升BEV感知

Timeline:

2022
BEVFormer
提出Transformer时序建模

2023
Sparse4D
降低计算成本

2024
SparseDrive
融合规划任务
```

---

# 系统架构

整体流程：

```text
Paper
↓
Candidate Retrieval
↓
Problem Extraction
↓
Problem Clustering
↓
Branch Construction
↓
Evolution Construction
↓
Timeline Generation
↓
Markdown Export
```

---

# 核心模块设计

## 1. paper_retriever.py

职责：

根据用户输入寻找相关论文。

输入：

```text
BEVFormer
```

输出：

```text
Top-N Candidate Papers
```

数据来源：

* Semantic Search
* Citation Search
* OpenAlex
* Semantic Scholar
* arXiv

目标：

高召回率。

不要求特别精准。

---

## 2. problem_extractor.py

职责：

从论文中提取研究问题。

输入：

```text
title
abstract
```

输出：

```json
{
  "problem":
  "Temporal modeling in BEV perception",

  "solution":
  "Transformer temporal attention",

  "limitations":
  "High computational cost"
}
```

核心 Prompt：

```text
What problem does this paper solve?

What solution is proposed?

What are the limitations?
```

---

## 3. problem_graph.py

职责：

构建 Problem Graph。

节点：

```text
Problem
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
Multi-view Projection
↓
Depth Estimation
↓
Temporal Modeling
```

输出：

```json
problem_graph.json
```

---

## 4. problem_cluster.py

职责：

聚类研究问题。

注意：

聚类对象不再是论文。

而是：

```text
Problem Statement
```

例如：

```text
How to model temporal information?

How to aggregate multi-frame features?

How to use temporal context?
```

归并为：

```text
Temporal Modeling
```

Branch

推荐方案：

```text
Embedding
+
LLM-based merge
```

而不是纯 KMeans。

---

## 5. branch_builder.py

职责：

自动生成技术分支。

输入：

```text
Problem Clusters
```

输出：

```text
Branch A

Temporal Modeling

Branch B

Depth Estimation

Branch C

Query-based BEV
```

每个 Branch 包含：

* Problem
* Key Papers
* Timeline

---

## 6. evolution_builder.py

职责：

构建技术演化链。

输入：

```text
同一Branch中的论文
```

按年份排序：

```text
Paper A
↓
Paper B
↓
Paper C
```

然后分析：

```text
Paper B 相比 A 解决了什么问题

Paper C 相比 B 又改进了什么
```

输出：

```json
{
  "previous":
  "BEVFormer",

  "next":
  "Sparse4D",

  "improvement":
  "Reduced computation cost"
}
```

---

## 7. key_paper_selector.py

职责：

识别关键论文。

评分：

```text
Citation Score

+

Graph Centrality

+

LLM Importance Score
```

输出：

```text
Top 5 Key Papers
```

---

## 8. timeline_builder.py

职责：

生成时间线。

输出：

```text
2020
Lift-Splat-Shoot

2021
BEVDet

2022
BEVFormer

2023
Sparse4D
```

并附带：

```text
解决的问题

核心创新
```

---

## 9. markdown_export.py

职责：

生成最终报告。

输出：

```markdown
# Field

BEV Perception

## Branch

Temporal Modeling

### Problem

如何利用时序信息

### Key Papers

- BEVFormer
- Sparse4D
- SparseDrive

### Evolution

BEVFormer
↓
Sparse4D
↓
SparseDrive

### Timeline

2022 → BEVFormer

2023 → Sparse4D

2024 → SparseDrive
```

---

# 数据结构设计

## paper.json

```json
{
  "title": "",

  "abstract": "",

  "year": 2022,

  "citation_count": 1200,

  "url": ""
}
```

---

## enriched_paper.json

```json
{
  "title": "BEVFormer",

  "year": 2022,

  "problem":
    "Temporal modeling in BEV",

  "solution":
    "Transformer temporal attention",

  "limitations":
    "High computational cost",

  "citations": []
}
```

---

# 为什么放弃纯 Embedding Clustering

实践证明：

```text
Embedding Similarity
≠
Research Branch
```

例如：

BEVFormer

可能同时接近：

* Transformer
* DETR
* PETR
* Sparse4D

最终聚类结果往往混乱。

原因：

Embedding 学到的是：

```text
Semantic Similarity
```

而不是：

```text
Research Intent
```

因此：

V2 不再以论文聚类为核心。

而改为：

```text
Paper
↓
Problem
↓
Branch
↓
Evolution
```

---

# 为什么放弃纯 Citation Graph

Citation Graph 只能告诉我们：

```text
谁引用了谁
```

不能回答：

```text
为什么出现

解决了什么问题

后来如何演化
```

因此 Citation 仅作为辅助信号。

而不是核心逻辑。

---

# 产品护城河

未来真正的壁垒不是：

```text
Embedding

Citation Graph

Paper Search
```

这些能力已经被大量产品覆盖。

真正的壁垒是：

```text
Paper
↓
Problem
↓
Solution
↓
Evolution
```

即：

构建领域级别的：

```text
Research Cognition Map
```

帮助用户理解：

```text
一个领域为什么发展成今天这样
```

而不仅仅是：

```text
有哪些论文
```

---

# V2 验证目标

验证以下问题：

1. 是否能稳定提取 Problem / Solution？

2. 是否能自动形成合理的 Branch？

3. 是否能自动生成符合专家认知的 Evolution Timeline？

4. 用户是否认为：

```text
这比传统论文搜索工具更有价值
```

如果上述验证成立，则进入 V3。

---

# V3 展望

未来扩展方向：

* PDF Figure Understanding
* Equation Understanding
* Related Work Generation
* Research Gap Discovery
* Interactive Research Map
* Chrome Extension
* Personal Research Memory

最终目标：

```text
Research Evolution Engine

成为研究人员理解一个领域的默认入口。
```
