# Paper Context Tool – MVP 设计文档

## 1. 项目目标 (Project Overview)

Paper Context Tool 是一个 **研究助手 (Research Assistant)**，
用于自动整理论文并生成一个结构化的知识地图：

Field → Branch → Key Papers → Timeline

核心目标不是简单做 **paper summary**，
而是帮助用户理解：

> 一篇论文在整个领域中的位置 (research context)。

示例输出：

Field: BEV Perception

Branch: Camera-based BEV

Key Papers

* Lift, Splat, Shoot (2020)
* BEVDet (2021)
* BEVFormer (2022)

Timeline
2020 → Lift, Splat, Shoot
2021 → BEVDet
2022 → BEVFormer

---

# 2. 目标用户 (Target Users)

### 工程师 (Engineers)

需要快速了解某个研究方向的技术路线，例如：

* BEV perception
* Diffusion models
* VLA models

### 研究人员 (Researchers)

希望理解：

* 某篇论文属于哪个技术分支
* 这个方向是如何发展的

---

# 3. MVP 功能目标 (MVP Goals)

第一版 MVP 只需要实现以下功能：

1. 从本地 JSON 文件加载论文数据
2. 使用 **embedding model** 生成论文向量
3. 根据相似度把论文分类到不同 **branch**
4. 识别每个 branch 的 **key papers**
5. 生成按时间排序的 **timeline**
6. 导出 Markdown 研究报告

---

# 4. 数据输入 (Data Sources)

MVP 版本先使用 **本地数据文件**。

未来可以扩展到：

* arXiv API
* OpenAlex API
* Semantic Scholar API

---

## 4.1 papers.json

每篇论文包含以下字段：

* title
* abstract
* year
* citation_count
* link

示例：

```json
{
"title": "BEVFormer",
"abstract": "We propose BEVFormer...",
"year": 2022,
"citation_count": 1200,
"link": "https://arxiv.org/abs/2203.17270"
}
```

---

## 4.2 branches.json

定义领域结构 (knowledge schema)。

示例：

```json
{
"field": "Autonomous Driving",
"branch": "Camera-based BEV",
"seed_papers": [
"Lift Splat Shoot",
"BEVDet"
]
}
```

seed_papers 用来帮助模型识别 branch 的语义。

---

# 5. 系统流程 (System Pipeline)

整体流程如下：

1️⃣ 加载论文数据 (papers.json)

2️⃣ 为论文生成 embedding

```
embedding(title + abstract)
```

3️⃣ 为 branch seed 生成 embedding

4️⃣ 计算 cosine similarity

5️⃣ 将论文分配到最相似的 branch

6️⃣ 根据 citation_count 识别 key papers

7️⃣ 按 year 生成 timeline

8️⃣ 输出 Markdown 报告

---

# 6. 项目结构 (Project Structure)

生成的 Python 项目结构如下：

```
paper_context_tool/

src/
main.py
config.py
embedding.py
classifier.py
timeline.py
markdown_export.py

data/
papers.json
branches.json

output/
field_map.md

docs/
design_mvp.md
```

---

# 7. 模块设计 (Module Responsibilities)

## main.py

负责 orchestrate 整个 pipeline。

主要流程：

1. 加载数据
2. 生成 embedding
3. 论文分类 (branch classification)
4. 识别 key papers
5. 生成 timeline
6. 导出 Markdown

---

## embedding.py

负责调用 **OpenAI embedding API**。

输入：

```
text
```

输出：

```
vector embedding
```

推荐模型：

```
text-embedding-3-small
```

---

## classifier.py

负责论文的 **branch classification**。

方法：

```
paper_embedding
vs
branch_seed_embedding
```

使用：

```
cosine similarity
```

返回：

```
best_branch
```

---

## timeline.py

根据论文年份生成技术发展时间线。

输出格式：

```
Year → Paper
```

---

## markdown_export.py

将最终结果导出为 Markdown 文档。

示例输出：

```
# Field: Autonomous Driving

## Branch: Camera-based BEV

Key Papers
- BEVFormer (2022)
- BEVDet (2021)

Timeline
2021 → BEVDet
2022 → BEVFormer
```

---

# 8. 依赖库 (Dependencies)

需要安装的 Python 包：

```
openai
numpy
scikit-learn
requests
```

安装：

```
pip install openai numpy scikit-learn requests
```

---

# 9. 运行方式 (Execution)

运行主程序：

```
python src/main.py
```

输出文件：

```
output/field_map.md
```

---

# 10. 未来扩展 (Future Extensions)

未来版本可以加入：

### 自动抓论文

```
arXiv API
OpenAlex
Semantic Scholar
```

### 自动发现 branch

使用：

```
embedding clustering
topic modeling
```

### citation graph 分析

构建论文引用关系图。

### PDF 解析

解析：

* figures
* tables
* equations

### Chrome 插件

实现：

```
paper reading assistant
```

并自动更新研究知识库。
