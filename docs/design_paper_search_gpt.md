这样就清楚了，我认为你的MVP定位其实是：

> **Field Name → Key Paper List**

后面的分析已经验证可行，所以这个模块唯一KPI就是：

> **Recall尽可能高，尤其不能漏Milestone。**

那我会完全按照Information Retrieval（IR）的思路设计，而不是按照LLM Agent来设计。

---

# 我会重新定义问题

首先我要明确一个观点：

> **关键论文不是一个集合，而是多个不同来源共同认为重要的论文。**

例如一篇论文为什么是Key Paper？

可能因为：

* Citation很高
* Survey引用了它
* 很多后续论文引用它
* LLM知道它
* PapersWithCode收录它
* GitHub Awesome List推荐它
* Semantic Scholar认为Relevant
* 它是某个Branch的起点

所以我不会让任何一个来源拥有决定权。

而是：

```
                Evidence

          LLM
           │
Survey ───┼──── Citation
           │
   PWC ───┼──── SS
           │
 GitHub ──┼──── OpenAlex

↓

Evidence Fusion

↓

Key Paper
```

整个系统其实就是Evidence Fusion。

---

# 第一阶段：Multi-source Recall（不是Dual-source）

你的方案只有：

```
SS

+

arXiv
```

我觉得远远不够。

我会把所有免费的知识源全部利用。

例如：

| Source            | 作用             | Recall价值 |
| ----------------- | -------------- | -------- |
| Semantic Scholar  | 全文检索+Citation  | ★★★★★    |
| OpenAlex          | Citation Graph | ★★★★★    |
| arXiv             | 最新论文           | ★★★★     |
| PapersWithCode    | SOTA方法         | ★★★★     |
| Survey论文Reference | Ground Truth   | ★★★★★    |
| Awesome List      | 工程社区           | ★★★      |
| Wikipedia         | 经典工作           | ★★       |
| LLM               | 隐式知识           | ★★★      |

注意：

这些不是互相替代。

而是：

互相补。

---

# 第二阶段：Query Expansion（整个系统最重要）

我认为这是整个系统的核心。

不是：

LLM生成Query。

而是：

Query不断长大。

例如：

输入：

```
BEV
```

第一轮：

```
BEV perception

Bird Eye View

View Transformation

```

检索回来：

LSS

BEVFormer

BEVDet

然后自动抽：

```
Lift Splat Shoot

Depth Distribution

View Transformer

Camera BEV
```

继续检索：

得到：

BEVDepth

SOLOFusion

继续抽：

```
Temporal BEV

Depth Supervision

```

继续：

Occupancy

Sparse

Gaussian

......

Query越来越丰富。

最后：

Recall自然越来越全。

这其实就是Google几十年前就在做的：

Query Expansion。

---

# 第三阶段：Citation Expansion

这里我不会：

Top30全部雪球。

而是：

每篇Candidate：

只扩一层。

得到：

```
Candidate

↓

Reference

+

Citation

↓

New Candidate
```

然后：

如果：

New Candidate

已经出现很多次。

说明：

它可能重要。

这里会产生：

一个Evidence Score。

例如：

```
BEVFormer

来自：

SS

Survey

Citation

LLM

PWC

Score=5
```

而：

某篇普通论文：

```
只有SS

Score=1
```

---

# 第四阶段：Evidence Voting（我认为这是整个系统核心）

不是：

LLM判断是不是Milestone。

而是：

Evidence投票。

例如：

| Paper     | SS | Survey | Citation | LLM | PWC | Total |
| --------- | -- | ------ | -------- | --- | --- | ----- |
| LSS       | ✔  | ✔      | ✔        | ✔   | ✔   | 5     |
| BEVFormer | ✔  | ✔      | ✔        | ✔   | ✔   | 5     |
| BEVerse   | ✔  | ✔      | ✔        | ✘   | ✔   | 4     |
| FIERY     | ✔  | ✔      | ✔        | ✘   | ✘   | 3     |
| 某普通论文     | ✔  | ✘      | ✘        | ✘   | ✘   | 1     |

这样：

LLM只是：

一个Vote。

而不是Judge。

---

# 第五阶段：Coverage Detection

这个地方我会加一个非常简单但是有效的方法。

Candidate出来以后。

不是：

LLM问：

有没有漏。

而是：

Embedding。

例如：

Candidate：

```
300篇
```

Sentence Transformer。

↓

UMAP。

↓

Cluster。

得到：

```
Cluster1

60 papers

Cluster2

55 papers

Cluster3

48 papers

Cluster4

3 papers

```

Cluster4：

明显Coverage不足。

LLM：

看看：

这是哪个Topic。

例如：

```
Occupancy
```

继续：

Recall Occupancy。

这样：

不是LLM记忆。

而是：

数据驱动。

---

# 第六阶段：Milestone Mining

直到最后。

才问LLM：

```
下面这些论文。

哪些是真正Milestone？

哪些只是Incremental？
```

而不是：

一开始：

LLM生成Milestone。

顺序完全反过来。

---

# 最终排序

最后：

不是：

Citation排序。

而是：

综合Evidence。

例如：

```
FinalScore

=

Evidence

+

Citation

+

Survey

+

Centrality

+

Venue

+

Year Balance

+

LLM Score
```

最后：

Top30。

---

# 整个流程只有六步

```
Field

↓

Multi-source Recall

↓

Dynamic Query Expansion

↓

Citation Expansion

↓

Evidence Voting

↓

Coverage Detection

↓

LLM Milestone Mining

↓

Final Key Paper List
```

---

# 与你当前方案最大的区别

我觉得最大的区别其实只有一句话：

**你的方案是"LLM驱动的检索（LLM-centric Retrieval）"，而我会设计成"证据驱动的检索（Evidence-centric Retrieval）"。**

在你的设计中，LLM承担了三个关键角色：生成 Query、判断 Milestone、发现遗漏论文。一旦LLM不知道某篇论文，它几乎没有机会再进入系统。

而我会让LLM退居为**众多证据源之一**。系统真正依赖的是多个独立来源形成的交叉证据：Semantic Scholar、OpenAlex、Survey 的参考文献、PapersWithCode、引用网络、动态 Query Expansion 等。论文是否是 Key Paper，不由LLM决定，而由这些证据共同投票决定。

---

## 如果只能保留一个设计原则，我会保留这一条

如果这是我要长期维护的系统，我会坚持一个原则：

> **任何关键论文，都至少应该有两条独立证据链能够把它带进候选池。**

例如：

* 通过领域 Query 检索到；
* 或者通过 Survey 的 Reference 找到；
* 或者通过 Citation Expansion 找到；
* 或者通过 PapersWithCode 找到；
* 或者通过动态 Query Expansion 找到。

只要保证**每篇真正重要的论文都有至少两种独立的进入路径**，系统的漏检率通常会比依赖任何单一策略（包括LLM）低得多，而且随着新证据源的加入还能持续提升，而不需要不断修改Prompt或增加规则补丁。
