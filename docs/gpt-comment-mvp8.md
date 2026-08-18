你说得对，我刚才那版实际上只保留了**Narrative组织方式**这一条建议，把之前几轮讨论里沉淀出来的其它几个重要结论丢掉了。

如果从整个 V4 → V5 演化过程来看，我认为真正稳定下来的建议其实有 **6 条**，而不是 1 条。

而且它们的重要程度并不一样。

---

# Narrative Engine V5 建议总览

## P0（必须修改）

### 1. Narrative 应该是 Tension + Paper 双驱动

这是目前最核心的结论。

不是：

```text
Paper → Paper
```

也不是：

```text
Tension → Tension
```

而是：

```text
Tension
 ↓
Paper A
 ↓
New Tension
 ↓
Paper B
```

原因：

技术史演化的是问题。

用户记住的是论文。

两者缺一不可。

---

### 2. Relation Builder 必须以 Problem 为前提

这是 DeepSeek 提出的那个 Problem 2。

目前很多关系实际上是：

```text
按时间排序
↓
自动连边
```

例如：

```text
Sparse4Dv2
↓
VAD
```

实际上根本不成立。

因为：

```text
Sparse4Dv2
= Perception

VAD
= Planning
```

是平行路线。

正确逻辑应该是：

```text
先判断：

是否解决同一个问题？

YES
  才允许建边

NO
  标记 Parallel
```

这个属于正确性问题。

优先级最高。

---

# P1（强烈建议修改）

### 3. Paradigm Shift 切分太细

这是我认为你最新版仍然存在的问题。

目前：

```text
Representation
  Dense → Sparse

Representation
  Raster → Vector

Geometry
  Explicit Depth → Attention

System
  Single Frame → Temporal

System
  Modular → E2E
```

看起来有 5 个 Paradigm Shift。

实际上很多并不是 Paradigm。

---

真正 Paradigm 的标准应该是：

```text
领域共识发生改变
```

例如：

```text
Dense BEV 必须
↓
Sparse Query 即可
```

这是 Paradigm。

---

但是：

```text
Single Frame
↓
Temporal Fusion
```

更像：

```text
Technique Evolution
```

不是 Paradigm。

---

我会压缩成：

### Paradigm 1

Dense Representation

↓

Sparse Representation

---

### Paradigm 2

Modular Driving Stack

↓

Planning-Oriented End-to-End Stack

---

### Paradigm 3（可选）

Explicit Geometry

↓

Learned Geometry

---

最多 3 个。

否则用户会觉得：

```text
什么都是 Paradigm
```

那其实等于没有 Paradigm。

---

### 4. Tension 应该成为第一公民

注意：

这不是说 Narrative 要按 Tension 写。

而是数据结构里。

当前更像：

```text
Paper
 ├── Claims
 ├── Evidence
 └── Relations
```

然后再推导：

```text
Tension
```

---

应该变成：

```text
Tension
 ├── Introduced By
 ├── Challenged By
 ├── Resolved By
 ├── Status
 └── Related Papers
```

Paper 成为参与者。

不是中心节点。

---

因为最终用户真正关心的是：

```text
Dense 为什么被 Sparse 替代？

Depth 为什么不再重要？

为什么开始做 E2E？
```

本质都是 Tension。

---

# P2（体验优化）

### 5. Claim Table 太重

最新版有个问题：

```text
Narrative
↓
Paper Table
↓
Claim Table
```

用户已经在 Narrative 里看到论文了。

后面又出现：

```text
论文
Claim
Evidence
```

会产生大量重复。

---

建议：

保留：

```text
Paper
Core Claim
Evidence
```

每篇论文最多一条核心主张。

---

其它 Claim 放到：

```text
Paper Detail Page
```

或者展开区域。

---

否则：

```text
7篇论文
×
4个Claim

=
28行
```

阅读体验会迅速崩。

---

### 6. Paper Relation Graph 不应该是主视图

这个是我看最新版最大的 UX 问题。

现在：

```text
Narrative
↓
Graph
↓
Paradigm
↓
Tension
```

实际上用户看完 Narrative 以后。

最想看的不是：

```text
Paper A → Paper B
```

而是：

```text
整个领域发生了什么变化
```

---

因此顺序建议改成：

```text
Narrative

↓

Paradigm Shift

↓

Tension Map

↓

Paper Graph

↓

Evidence
```

Paper Graph 更适合作为索引。

不是主菜。

---

# 最终我认为的 V5 版本

## P0

✅ Narrative = Tension + Paper

✅ Relation Builder 基于 Problem 而不是时间

---

## P1

✅ Paradigm 数量压缩到 2~3 个

✅ Tension 成为第一公民

---

## P2

✅ Claim Table 大幅瘦身

✅ Paper Graph 后移

---

# 一句话总结

如果只允许我选一个方向，我会认为你现在最大的问题已经不是 Narrative 本身，而是：

```text
Narrative
已经开始接近专家讲课

但底层数据结构
仍然是 Paper-Centric
```

所以系统会不断把结果拉回：

```text
Paper → Paper → Paper
```

而你真正想构建的其实是：

```text
Tension
 ↓
Paper
 ↓
Tension
 ↓
Paper
 ↓
Paradigm Shift
```

这也是为什么你会感觉最早版本反而有些地方更对——因为当时虽然粗糙，但实际上更接近“问题驱动叙事”，后面为了增强可解释性和可追溯性，逐渐又滑回了“论文驱动叙事”。这才是最近几轮讨论里真正暴露出来的根因。
