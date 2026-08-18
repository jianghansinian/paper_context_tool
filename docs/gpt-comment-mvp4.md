如果你已经增加了：

build_paper_chain_relations

和

classify_paradigm_relation

那我认为整个项目已经进入了一个新的阶段：

V1
Paper Summary

↓

V2
Paper Graph

↓

V3
Problem Evolution

↓

V4
Claim Evolution

↓

V5
Paradigm Evolution

而且从你前面贴出来的 Narrative 看，你现在已经在 V4 和 V5 之间了。

我现在最关心的问题

其实已经不是：

Branch怎么生成

了。

而是：

Relation是否足够准确

因为技术路线的本质其实是：

Node 不重要

Edge 才重要

举个例子

下面两种图：

图A
LSS

BEVDepth

BEVFormer

Sparse4D

这是节点。

没有任何价值。

图B
LSS
    ↓ establish

BEVDepth
    ↓ refine

BEVFormer
    ↓ challenge

Sparse4D

这是关系。

开始有价值了。

我最想看的不是 Narrative

而是：

{
  "paper": "BEVFormer",

  "claim":
  "Temporal transformer can implicitly learn depth",

  "relations": [
    {
      "target": "BEVDepth",
      "type": "challenge"
    }
  ]
}

如果这个层构建得好：

Narrative 基本自动就会好。

我怀疑你现在最大的风险

是 Relation 分类过于理想化

通常大家会设计：

support
challenge
replace
extend
merge

五类。

看起来合理。

但实际上论文关系远比这个复杂。

以 BEV 为例：

LSS

Claim

Need explicit depth distribution.
BEVDepth

Claim

Need depth supervision.

这不是：

challenge

也不是：

replace

更像：

strengthen
BEVFormer

Claim

Transformer can implicitly aggregate geometry.

这才是：

challenge
Sparse4D

Claim

Dense BEV is unnecessary.

这是：

paradigm shift

所以我越来越觉得：

你最终可能需要两层关系。

第一层

Paper Relation

support
extend
challenge
replace
combine
第二层

Paradigm Relation

incremental

optimization

paradigm_shift

convergence

dead_end

因为：

BEVDepth → BEVFormer

和

BEVFormer → Sparse4D

虽然都是 challenge。

但完全不是一个量级。

前者：

方法挑战

后者：

范式挑战
我认为你现在最值得检查的地方

看看：

classify_paradigm_relation()

是不是只在做：

Dense -> Sparse

这种表面判断。

真正应该判断的是：

Claim Space 是否改变

例如：

BEVDepth

Claim

Depth Quality
决定性能

BEVFormer

Claim

Temporal Modeling
决定性能

这里已经发生：

研究重心变化

了。

虽然仍然属于 Dense BEV。

而很多系统会漏掉这种 Paradigm Shift。

因为它们只看：

Method

不看：

Research Question
另外一个我会立刻补的东西

新增：

detect_research_tension()

输出：

{
  "tension":
  "Dense BEV cost grows quadratically",

  "introduced_by":
  [
    "BEVFormer",
    "BEVFormerV2"
  ],

  "resolved_by":
  [
    "Sparse4D"
  ]
}

为什么我认为这个模块重要？

因为用户真正记住的不是：

BEVFormer

而是：

Dense Attention 太贵

用户真正记住的不是：

Sparse4D

而是：

Sparse Query 解决了 Dense Query 的成本问题

换句话说：

技术路线真正的主角不是：

Paper

也不是：

Claim

而是：

Tension

如果让我预测 V6 长什么样。

我觉得会变成：

Paper
 ↓

Claim
 ↓

Claim Relation
 ↓

Research Tension
 ↓

Paradigm Shift
 ↓

Narrative

其中：

Research Tension

会成为整个系统最有价值的数据结构。

因为技术发展史本质上就是：

发现矛盾

↓

解决矛盾

↓

产生新矛盾

↓

再解决

这也是为什么优秀的专家讲技术路线时，总是在讲：

“当时大家发现 XX 不行，于是出现了 YY；后来 YY 又暴露出 ZZ，所以才有了后来的方法。”

如果你现在的新版已经有了 build_paper_chain_relations 和 classify_paradigm_relation，我下一步不会继续改 Branch，而会开始抽象：
ResearchTension

因为这很可能是让 Narrative 从 85 分走到 95 分的关键。