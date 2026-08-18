````markdown
# Research Narrative Engine V4 Review：从论文图谱到技术叙事引擎

## 背景

经过几轮设计与评审，我们的产品定位经历了三个阶段：

```text
Paper Graph
    ↓
Research Evolution Engine
    ↓
Research Narrative Engine
```

最开始关注的是：

- 找论文
- 看论文
- 建立论文关系

后来逐渐发现：

用户真正需要的并不是论文图谱（Graph）。

而是：

> 为什么这个领域发展成今天这样？

于是产品开始向：

```text
Research Narrative Engine
```

演进。

---

# 当前阶段的判断

如果给整个设计过程打分：

| 阶段 | 评分 |
|--------|--------|
| 第一版设计 | 60 |
| Evolution Engine | 75 |
| Narrative Engine | 85 |
| 最新 Review | 92 |

原因很简单：

讨论已经从：

```text
方向是否正确
```

变成：

```text
如何真正实现
```

说明产品定位已经逐渐收敛。

---

# 当前最重要的两个问题

## 1. Claim Extraction

这是整个系统最关键的模块。

甚至比：

- 检索
- 聚类
- DAG构建

都更重要。

---

## 为什么 Problem 不够？

例如：

### BEVFormer

Problem：

```text
如何建模时序信息？
```

### Sparse4D

Problem：

```text
如何建模时序信息？
```

问题完全相同。

---

但真正发生变化的是：

```text
Claim
```

### BEVFormer

Claim：

```text
Dense Transformer Attention
能够有效完成时序建模
```

### Sparse4D

Claim：

```text
Sparse Query
能够以更低成本实现时序建模
```

---

因此：

```text
Problem
定义战场

Claim
定义立场
```

---

# Solution 和 Claim 的区别

这是未来 Prompt 设计中的重点。

很多 LLM 会把：

```text
Solution
```

误认为：

```text
Claim
```

---

例如：

论文中写：

```text
We use V-JEPA for video pre-training.
```

这是：

```text
Solution
```

---

真正的 Claim 是：

```text
V-JEPA pretraining
can outperform perception-heavy pipelines.
```

---

再举例：

### Sparse4D

Solution：

```text
Sparse Query
```

Claim：

```text
Dense BEV representation is unnecessary.

Sparse representation can preserve performance.
```

---

用户真正想看的其实是：

```text
Claim War
```

而不是：

```text
Method List
```

---

# 研究发展的本质

很多人认为技术路线是：

```text
Paper A
↓
Paper B
↓
Paper C
```

实际上并不是。

真正的发展过程是：

```text
Claim A
↓
Claim B 质疑 A
↓
Claim C 修正 B
↓
Claim D 统一 A 与 B
```

---

这才是真正的：

```text
Research Story
```

---

# 2. Narrative Structure

这是第二个决定产品成败的模块。

---

## Timeline 不等于 Narrative

很多工具输出：

```text
2021
Paper A

2022
Paper B

2023
Paper C
```

这只是：

```text
Timeline
```

---

用户真正需要的是：

```text
Conflict
↓
Attempt
↓
Failure
↓
New Idea
↓
Breakthrough
```

---

# BEV 的例子

## 第一幕：矛盾出现

```text
多相机特征无法统一到BEV空间
```

---

## 第二幕：第一次突破

```text
Lift-Splat-Shoot

提出视图投影方案
```

---

## 第三幕：新问题暴露

```text
深度估计误差过大
```

---

## 第四幕：第二次突破

```text
BEVDepth

利用激光雷达监督深度
```

---

## 第五幕：新的矛盾

```text
依赖激光雷达成本过高
```

---

## 第六幕：范式转移

```text
BEVFormer

使用Transformer隐式学习深度
```

---

## 第七幕：效率问题

```text
Dense Attention成本太高
```

---

## 第八幕：效率优化

```text
Sparse4D

Sparse Query
```

---

这已经不是：

```text
Paper Timeline
```

而是：

```text
技术发展史
```

---

# Hidden Gap：Paradigm Shift

目前所有设计里还有一个隐藏问题。

大家都在讨论：

```text
Claim Evolution
```

但很多领域的发展实际上是：

```text
Paradigm Evolution
```

---

例如自动驾驶：

## 第一阶段

```text
Modular Stack
```

2016-2020

---

## 第二阶段

```text
End-to-End Driving
```

2020-2023

---

## 第三阶段

```text
Foundation Models
```

2023-2025

---

## 第四阶段

```text
World Models
```

2025-
```

---

用户真正想知道的是：

```text
为什么范式发生变化？
```

而不是：

```text
为什么论文B比论文A提升了3%？
```

---

因此未来可能需要：

```python
paradigm_shift_detector.py
```

输出：

```json
{
  "old_paradigm":
  "Dense BEV Grid",

  "new_paradigm":
  "Sparse Query",

  "evidence":
  [...]
}
```

---

# 关于 Branch Discovery

当前讨论中的一个争议点。

很多方案认为：

```text
Problem Embedding
+
LLM Refinement
```

即可自动发现 Branch。

---

但长期来看：

Branch 更像：

```text
人类形成的知识分类体系
```

而不是自然聚类结果。

---

例如：

BEV 领域：

```text
Depth-based

Transformer-based

Sparse-based
```

---

VLA 领域：

```text
Behavior Cloning

Diffusion Policy

Action Tokenization

Vision-Language Models
```

---

这些分类往往来自：

```text
社区共识
```

而不是：

```text
Embedding Cluster
```

---

因此更合理的方案可能是：

```text
LLM Suggestion
+
Human Confirmation
```

而不是完全自动发现。

---

# V3 与 V4 的关系

这一点必须定义清楚。

---

## V3

定位：

```text
Single Paper Understanding
```

输入：

```text
一篇论文
```

输出：

```text
结构化深度分析
```

包括：

- Motivation
- Problem
- Method
- Formula
- Experiment
- Limitation

---

## V4

定位：

```text
Field Narrative Understanding
```

输入：

```text
一个领域
或
一篇种子论文
```

输出：

```text
整个领域的发展故事
```

---

关系：

```text
Field Narrative
      ↓
Seed Paper
      ↓
Deep Analysis
```

---

最终体验：

```text
BEV发展史

↓
定位到BEVFormer

↓
详细解析BEVFormer
```

---

# 当前最值得验证的实验

不要继续讨论：

- DAG
- Citation
- Graph
- Clustering

这些都不是决定成败的因素。

---

直接建立实验目录：

```text
experiment_narrative/
```

---

## 第一步

人工选择：

```text
BEV领域
20篇关键论文
```

---

## 第二步

实现：

```text
claim_extractor.py
```

目标：

```text
提取真正的Claim
而非Method
```

---

## 第三步

实现：

```text
narrative_builder.py
```

目标：

```text
生成技术发展故事
```

---

## 第四步

输出：

```text
《BEV技术发展史》
```

---

## 第五步

找有自动驾驶背景的人阅读

验证：

```text
是否符合专家脑中的技术路线
```

---

# 最终判断

如果 Narrative Generation 做不到：

```text
让专家觉得：
“是这么回事”
```

那么问题不在：

- Citation
- Embedding
- Clustering
- DAG

而在：

```text
Narrative Generation
```

本身。

---

如果 Narrative Generation 成功：

那么后续：

```text
自动检索论文

自动发现 Branch

自动构建 DAG

自动生成路线图
```

都只是工程优化问题。

---

# 最终愿景

产品最终不应该是：

```text
Paper Search Engine
```

也不应该是：

```text
Paper Graph Tool
```

而应该是：

```text
Research Narrative Engine
```

帮助用户回答三个问题：

```text
为什么？

发生了什么？

下一步会走向哪里？
```

成为研究人员理解一个领域的默认入口。
````
