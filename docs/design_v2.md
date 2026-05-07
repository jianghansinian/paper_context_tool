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

### 与 V1 的关系

V1（对应 `design_mvp.md`）和 V2 **共存而非替代**：

| | V1 | V2 |
|---|---|---|
| 代码位置 | 根目录 (`main.py`, `classifier.py`...) | `src/` 目录 |
| Branch 来源 | 人工手写 `branches.json` | 聚类自动发现 |
| 论文来源 | 本地 `papers.json` | arXiv / OpenAlex 抓取 |
| 适用场景 | 领域结构已知，快速出结果 | 领域不熟，需要先探索分支 |
| 运行命令 | `python main.py` | `python src/main.py "keyword"` |

选择建议：如果你对某个领域已经很熟悉，想按自己的理解组织结果，用 V1。如果想探索一个陌生的新领域，用 V2。

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
classifier.py
cluster.py
branch_discovery.py
citation_graph.py
key_paper.py
llm_namer.py
timeline.py
markdown_export.py

data/

papers.json
embeddings.pkl

output/

field_map.md
research_graph.json

docs/

design_mvp.md
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
OpenAI text-embedding-3-small（通过 EMBEDDING_API_KEY / EMBEDDING_BASE_URL 配置）
```

可换为任何 OpenAI 兼容的 embedding 服务，或完全不配置 API key
自动使用本地 HashingVectorizer fallback（免费，无需网络）。

结果缓存到：

```
embeddings.pkl
```

---

## llm_namer.py

为 cluster 生成语义化的 branch 名称（LLM 命名）。

与 embedding 配置独立，默认指向 DeepSeek：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_API_KEY` | 同 `DEEPSEEK_API_KEY` | 用于 LLM 调用的 API key |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容的 chat endpoint |
| `LLM_BRANCH_NAMING_MODEL` | `deepseek-chat` | 聊天模型名 |

调用失败或无 key 时自动回退到 `branch_discovery.py` 的关键词命名，
不会阻断 pipeline。

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

### 基本运行（纯本地，无需任何 API key）

```
python src/main.py "diffusion models"
```

此时 embedding 使用本地 HashingVectorizer，branch 命名使用关键词，
完全免费。

### 开启 LLM branch 命名（推荐 DeepSeek，极低成本）

```bash
# DeepSeek 案例（~$0/token 级别成本）
export LLM_API_KEY="sk-deepseek-xxx"
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_BRANCH_NAMING_MODEL="deepseek-chat"
```

### 同时开启 embedding（选配，不影响已有功能）

如需更高的 embedding 质量：

```bash
export EMBEDDING_API_KEY="sk-openai-xxx"
export EMBEDDING_BASE_URL="https://api.openai.com/v1"
export EMBEDDING_MODEL="text-embedding-3-small"
```

OpenAI `text-embedding-3-small` 成本约 $0.02/1M tokens（120 篇论文
不足 $0.001）。可选任何 OpenAI 兼容的 embedding 服务。

### 配置汇总

| 变量 | 默认值 | 用途 |
|---|---|---|
| `EMBEDDING_API_KEY` | 无 | Embedding API key（不设则本地 fallback） |
| `EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | Embedding endpoint |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding 模型名 |
| `LLM_API_KEY` | 同 `DEEPSEEK_API_KEY` | LLM branch 命名 API key（不设则关键词回退） |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | LLM chat endpoint |
| `LLM_BRANCH_NAMING_MODEL` | `deepseek-chat` | LLM 模型名 |
| `ENABLE_LOCAL_EMBEDDING_FALLBACK` | `1` | 无 key/API 不可达时是否本地降级 |

输出：

```
output/field_map.md
output/research_graph.json
```

---

# 9. 评估方法 (Evaluation)

当前系统缺少对输出质量的客观衡量。建议的评估方式：

### 9.1 聚类质量

```
Silhouette Score（轮廓系数）
Davies-Bouldin Index
```

用于衡量同一 cluster 内论文的紧密程度和不同 cluster 间的分离程度。注意：这两个指标对高维 embedding 数据有参考价值，但不应作为唯一标准。

### 9.2 Branch 命名质量

人工抽查 — 从每个 cluster 抽 3 篇论文，判断生成的 branch 名称是否准确地概括了该 cluster 的技术方向。建议至少覆盖 80% 的 cluster。

### 9.3 Key Paper 合理性

以 citation_count 最高的一篇作为基准，检查是否被遗漏。同时可以对比 Semantic Scholar 的 "Highly Influential Citations" 标记交叉验证。

---

# 10. 增量更新策略 (Incremental Update)

当用户对同一领域多次抓取时，需要处理去重和合并：

### 10.1 论文级去重

```
title（归一化：小写、去标点）
```
如果 title 相同则视为同一篇。保留 citation_count 更高的那一条（假设数据源更新的数据更准确），保留更早抓取的 embedding（减少 API 调用）。

### 10.2 增量 embedding

新论文先查 `embeddings.pkl` 缓存（SHA-256 比对 title+abstract），命中则复用。未命中才调用 API 生成新 embedding，追加到缓存。

### 10.3 重新聚类

论文数量变化 > 10% 时触发重新聚类，否则沿用上次的 branch 结构，仅更新 key paper 排名和 timeline。

---

# 11. 错误恢复与降级 (Error Recovery)

系统各环节的降级策略：

| 环节 | 失败场景 | 降级方案 |
|---|---|---|
| arXiv 抓取 | 超时 / 返回空 | 继续尝试 OpenAlex，不中断 |
| OpenAlex 抓取 | 超时 / rate limit | 跳过，仅用 arXiv 的结果 |
| 两个源都失败 | 网络不通 | 回退到本地 `papers.json` |
| Embedding API | 超时 / 404 / 无 key | HashingVectorizer 本地 fallback（1024 维 bigram） |
| UMAP + HDBSCAN | 数据分布不适合 / 依赖缺失 | PCA + KMeans（`_estimate_k` 动态估计 k） |
| HDBSCAN 全标注为噪声 | min_cluster_size 过大 | 全部归入 cluster 0 |
| 聚类数 < 2 | 论文太少 | 单 cluster |
| OpenAlex abstract | 存储为倒排索引 | 重建为连续文本 |

设计原则：**每一步都有 fallback，不要让单点故障终止整个 pipeline。**

---

# 12. 未来扩展 (Future Roadmap)

以下按优先级排列：

### 优先级 1 — 短期可落地

**LLM branch 命名**：目前 branch 名称来自 CountVectorizer 的 top n-gram，可读性一般。下一步用聚类 centroid 附近的论文作为 context 喂给 LLM 生成更有语义的名称。估算成本：每 cluster 约 500-1000 token prompt，使用快速模型即可。

**Semantic Scholar 引用数据**：目前 `citation_graph.py` 用标题相似度推断引用边，准确率很低。OpenAlex 和 Semantic Scholar 都提供真实的引用关系 API，接入后 citation graph 可以用于 PageRank 排序、引用链分析。

**基础测试**：至少给 `embedding.py`（API/fallback 切换）、`cluster.py`（输入输出形状）、`crawler.py`（dedup 逻辑）加单元测试。

---

### 优先级 2 — 中期目标

**Research Map 可视化**：用 Plotly 或 D3.js 生成交互式图，节点是论文/分支，边是引用关系。可放大缩小、点击查看详情。`research_graph.json` 已输出为 node-link 格式，可以直接作为前端数据源。

**增量更新**：实现 §10 描述的增量抓取与增量聚类。

---

### 优先级 3 — 长期探索

**PDF 解析**：解析论文 PDF 中的图表、公式、实验设置。需要 `PyMuPDF` 或 `pdfplumber`，输出结构化后可以用于更精细的论文对比分析。

**Chrome 插件**：浏览 arXiv 或会议论文页面时自动弹出该论文在领域中的位置（所属 branch、前后时序论文），需要后端 API 支持实时查询。
