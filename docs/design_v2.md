# Paper Context Tool – Design V2

## 1. 项目目标 (Project Goal)

Paper Context Tool 是一个 **Research Assistant**，
用于自动分析某个研究领域，并生成该领域的 **技术路线图 (Research Map)**。

系统最终输出结构：

```
Field
 ├── Branch A
 │     ├── Key Papers
 │     └── Timeline
 ├── Branch B
 │     ├── Key Papers
 │     └── Timeline
```

相比 V1：

V1：

```
paper → predefined branch
```

V2：

```
paper dataset
↓
embedding
↓
branch discovery
↓
timeline
```

也就是说 **branch 不再人工定义，而是自动发现**。

---

# 2. 目标用户 (Target Users)

### 工程师 (Engineers)

需要快速了解一个新领域，例如：

* Diffusion Models
* BEV Perception
* Vision-Language-Action

### 研究人员 (Researchers)

希望理解：

* 一个领域有哪些技术分支
* 技术路线如何演化
* 哪些是关键论文

---

# 3. V2 MVP 功能目标

V2 的 MVP 需要实现以下能力：

### 1 获取论文

输入：

```
keyword
```

例如：

```
BEV perception
diffusion models
```

系统自动从：

* arXiv
* OpenAlex

抓取论文。

输出：

```
papers.json
```

---

### 2 生成论文 embedding

对以下文本生成向量：

```
title + abstract
```

使用模型：

```
text-embedding-3-small
```

输出：

```
paper embeddings
```

---

### 3 自动发现技术分支 (Branch Discovery)

算法流程：

```
paper embeddings
↓
dimensionality reduction (UMAP)
↓
clustering (HDBSCAN)
↓
branch clusters
```

输出：

```
branch_id
```

---

### 4 识别 Key Papers

每个 branch 需要识别核心论文。

评分方法：

```
score =
citation_count_weight
+
embedding_centrality
```

选出：

```
top 5 papers
```

---

### 5 生成技术路线 Timeline

根据论文年份排序：

```
year → paper
```

输出：

```
branch timeline
```

---

### 6 生成 Research Map

最终生成：

```
Field Map
```

示例：

```
Field: Diffusion Models

Branch 1: Score-based diffusion

Key Papers
- Score Matching
- DDPM

Timeline
2019 → Score Matching
2020 → DDPM

Branch 2: Latent diffusion

Key Papers
- LDM
- Stable Diffusion
```

---

# 4. 系统整体流程 (System Pipeline)

完整 pipeline：

```
input keyword
↓
paper crawler
↓
paper dataset
↓
embedding generation
↓
clustering
↓
branch discovery
↓
key paper detection
↓
timeline generation
↓
markdown export
```

---

# 5. 项目结构 (Project Structure)

```
paper_context_tool

src/

main.py
config.py

crawler.py
embedding.py
cluster.py
branch_discovery.py
citation_graph.py
key_paper.py
timeline.py
markdown_export.py

data/

papers.json
embeddings.pkl

output/

field_map.md
research_graph.json

docs/

design_v1.md
design_v2.md
```

---

# 6. 模块设计 (Modules)

## crawler.py

负责抓取论文。

数据来源：

* arXiv API
* OpenAlex API

输入：

```
keyword
```

输出：

```
papers.json
```

每篇论文包含：

```
title
abstract
year
citation_count
link
```

---

## embedding.py

生成论文向量。

输入：

```
title + abstract
```

输出：

```
embedding vector
```

模型：

```
OpenAI text-embedding-3-small
```

结果缓存到：

```
embeddings.pkl
```

---

## cluster.py

执行 embedding clustering。

流程：

```
embedding
↓
UMAP
↓
HDBSCAN
```

输出：

```
cluster_id
```

---

## branch_discovery.py

根据 cluster 生成 branch。

步骤：

```
cluster papers
↓
extract keywords
↓
generate branch name
```

branch 名称可通过 LLM 生成。

---

## citation_graph.py

构建论文引用关系图。

节点：

```
paper
```

边：

```
citation
```

输出：

```
research_graph.json
```

用于后续可视化。

---

## key_paper.py

识别每个 branch 的关键论文。

评分：

```
score =
citation_count
+
cluster_centrality
```

返回：

```
top 5 papers
```

---

## timeline.py

生成技术发展 timeline。

方法：

```
sort by year
```

输出：

```
timeline
```

---

## markdown_export.py

导出研究报告。

输出文件：

```
output/field_map.md
```

结构：

```
# Field

## Branch

Key Papers

Timeline
```

---

# 7. 依赖库 (Dependencies)

需要安装：

```
openai
numpy
scikit-learn
hdbscan
umap-learn
requests
networkx
```

安装：

```
pip install openai numpy scikit-learn hdbscan umap-learn requests networkx
```

---

# 8. 运行方式 (Execution)

运行：

```
python src/main.py "diffusion models"
```

输出：

```
output/field_map.md
```

---

# 9. 未来扩展 (Future Roadmap)

V3 可以增加：

### 1 PDF 解析

解析：

```
figures
tables
equations
```

---

### 2 Paper Reading Assistant

Chrome 插件读取：

* arXiv
* conference papers

并自动生成：

```
paper context
```

---

### 3 Research Map Visualization

使用：

```
D3.js
or
NetworkX
```

生成交互式研究图谱。

---

### 4 Long-term Research Memory

为用户建立：

```
personal research knowledge base
```
