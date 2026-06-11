"""Biology domain profile.

Defines 5 paper types for the biology / life-sciences domain:
  experimental, computational, review, data_resource, method_protocol
"""

from __future__ import annotations

from domains.base import (ColumnDef, DomainProfile, FieldDef, PaperTypeProfile,
                           SectionDef)

# ══════════════════════════════════════════════════════════════════════
# Shared fields
# ══════════════════════════════════════════════════════════════════════

_F_PROBLEM = FieldDef(
    name="problem", kind="text", label_en="Problem", label_zh="生物学问题",
    prompt="What is the core biological question this study addresses?",
    required=True,
)

_F_MOTIVATION = FieldDef(
    name="motivation", kind="text", label_en="Motivation", label_zh="动机",
    prompt="Why is this question important? What prior work establishes the foundation?",
    required=True,
)

_F_CONTRIBUTIONS = FieldDef(
    name="contributions", kind="list[str]",
    label_en="Main Contributions", label_zh="主要贡献",
    prompt="List the main contributions of this study.",
    required=True,
)

_F_LIMITATIONS = FieldDef(
    name="limitations", kind="list[str]",
    label_en="Limitations", label_zh="局限性",
    prompt="List limitations acknowledged by the authors or obvious from the work.",
    required=True,
)

# ── Common columns ──
_COL_STRUCTURED = [
    ColumnDef(name="title", label_en="Item", label_zh="项目"),
    ColumnDef(name="description", label_en="Description", label_zh="描述"),
    ColumnDef(name="papers", label_en="References", label_zh="参考"),
]

_COL_RESULT = [
    ColumnDef(name="dataset", label_en="Endpoint", label_zh="观测项"),
    ColumnDef(name="metric", label_en="Finding", label_zh="发现"),
    ColumnDef(name="value", label_en="Value", label_zh="数值"),
    ColumnDef(name="comparison", label_en="Statistical Note", label_zh="统计说明"),
]

_COL_KV = [
    ColumnDef(name="key", label_en="Parameter", label_zh="参数"),
    ColumnDef(name="value", label_en="Value", label_zh="数值"),
]

# ── Shared biological fields ──

_F_HYPOTHESIS = FieldDef(
    name="hypothesis", kind="text",
    label_en="Hypothesis", label_zh="研究假设",
    prompt="What is the explicit research hypothesis? If not stated, "
           "summarize what the authors aim to demonstrate.",
    required=True,
)

_F_BIOLOGICAL_SYSTEM = FieldDef(
    name="biological_system", kind="text",
    label_en="Biological System", label_zh="研究体系",
    prompt="Describe the biological system: species, strain, cell line, "
           "tissue, developmental stage, etc.",
    required=True,
)

_F_EXPERIMENTAL_METHODS = FieldDef(
    name="experimental_methods", kind="structured_list",
    label_en="Experimental Methods", label_zh="实验方法",
    prompt="List each experimental technique used: name, purpose, key parameters.",
    required=True,
    columns=_COL_STRUCTURED,
)

_F_EXPERIMENTAL_DESIGN = FieldDef(
    name="experimental_design", kind="text",
    label_en="Experimental Design", label_zh="实验设计",
    prompt="Describe the experimental design: groups, controls, replicates, "
           "randomization, blinding.",
    required=True,
)

_F_KEY_FINDINGS = FieldDef(
    name="key_findings", kind="result_table",
    label_en="Key Findings", label_zh="核心发现",
    prompt="Summarize key findings: endpoint, finding, values, statistical "
           "significance and effect sizes.",
    required=False,
    columns=_COL_RESULT,
)

_F_STATISTICAL_ANALYSIS = FieldDef(
    name="statistical_analysis", kind="text",
    label_en="Statistical Analysis", label_zh="统计分析",
    prompt="Describe statistical methods: test names, software, significance "
           "thresholds, multiple testing corrections.",
    required=True,
)

_F_DATA_AVAILABILITY = FieldDef(
    name="data_availability", kind="text",
    label_en="Data Availability", label_zh="数据可获取性",
    prompt="Describe data availability: accession numbers, databases, "
           "repositories. Return null if not stated.",
    required=False,
)

_F_REPRODUCIBILITY_NOTES = FieldDef(
    name="reproducibility_notes", kind="text",
    label_en="Reproducibility Notes", label_zh="复现注意事项",
    prompt="Note any critical experimental conditions, batch effects, etc. "
           "affecting reproducibility. Return null if not discussed.",
    required=False,
)

_F_MECHANISTIC_INSIGHT = FieldDef(
    name="mechanistic_insight", kind="text",
    label_en="Mechanistic Insight", label_zh="机制解释",
    prompt="Describe the mechanistic interpretation of the observations at "
           "the biological level. Return null if not discussed.",
    required=False,
)

# ── Computational-specific ──

_F_ALGORITHM_OVERVIEW = FieldDef(
    name="algorithm_overview", kind="text",
    label_en="Algorithm Overview", label_zh="算法概述",
    prompt="Describe the algorithm or computational method.",
    required=True,
)

_F_INPUT_OUTPUT_SPEC = FieldDef(
    name="input_output_spec", kind="text",
    label_en="Input/Output Specification", label_zh="输入输出规格",
    prompt="Describe input data format and output result format.",
    required=True,
)

_F_SOFTWARE_ARCHITECTURE = FieldDef(
    name="software_architecture", kind="text",
    label_en="Software Architecture", label_zh="软件架构",
    prompt="Describe the software architecture. Return null if not detailed.",
    required=False,
)

_F_IMPLEMENTATION = FieldDef(
    name="implementation", kind="text",
    label_en="Implementation", label_zh="实现",
    prompt="Describe the implementation: programming language, dependencies, "
           "runtime environment. Return null if not specified.",
    required=False,
)

_F_BENCHMARK_DATASETS = FieldDef(
    name="benchmark_datasets", kind="text",
    label_en="Benchmark Datasets", label_zh="基准数据集",
    prompt="What datasets were used for benchmarking?",
    required=True,
)

_F_PERFORMANCE_RESULTS = FieldDef(
    name="performance_results", kind="result_table",
    label_en="Performance Results", label_zh="性能评测",
    prompt="Summarize performance: accuracy, speed, resource usage.",
    required=False,
    columns=_COL_RESULT,
)

_F_COMPARISONS_TO_EXISTING = FieldDef(
    name="comparisons_to_existing", kind="result_table",
    label_en="Comparisons to Existing Tools", label_zh="工具对比",
    prompt="Compare against existing tools.",
    required=False,
    columns=_COL_RESULT,
)

_F_AVAILABILITY = FieldDef(
    name="availability", kind="text",
    label_en="Availability", label_zh="获取方式",
    prompt="Describe how to access: GitHub URL, license, documentation.",
    required=True,
)

# ── Review-specific ──

_F_SCOPE = FieldDef(
    name="scope", kind="text", label_en="Scope", label_zh="范围",
    prompt="Define the scope: inclusion/exclusion criteria, search strategy, "
           "time range covered.",
    required=True,
)

_F_KEY_INSIGHT = FieldDef(
    name="key_insight", kind="text", label_en="Key Insight", label_zh="核心发现",
    prompt="What is the most important conclusion from this review?",
    required=True,
)

_F_TAXONOMY_OR_FRAMEWORK = FieldDef(
    name="taxonomy_or_framework", kind="text",
    label_en="Organizing Framework", label_zh="组织框架",
    prompt="Describe the organizational framework/taxonomy used (if any).",
    required=True,
)

_F_KEY_PAPERS_ANALYZED = FieldDef(
    name="key_papers_analyzed", kind="structured_list",
    label_en="Key Papers Analyzed", label_zh="重点论文分析",
    prompt="List key papers analyzed and their contributions.",
    required=False,
    columns=_COL_STRUCTURED,
)

_F_META_ANALYSIS_METHODS = FieldDef(
    name="meta_analysis_methods", kind="text",
    label_en="Meta-analysis Methods", label_zh="Meta分析统计方法",
    prompt="Describe meta-analysis statistical methods if applicable. "
           "Return null if not a meta-analysis.",
    required=False,
)

_F_META_ANALYSIS_RESULTS = FieldDef(
    name="meta_analysis_results", kind="result_table",
    label_en="Meta-analysis Results", label_zh="Meta分析定量结果",
    prompt="Provide quantitative meta-analysis results if applicable. "
           "Return null if not a meta-analysis.",
    required=False,
    columns=_COL_RESULT,
)

_F_TRENDS = FieldDef(
    name="trends", kind="text", label_en="Trends", label_zh="领域趋势",
    prompt="Summarize observed trends in the field.",
    required=True,
)

_F_OPEN_QUESTIONS = FieldDef(
    name="open_questions", kind="list[str]",
    label_en="Open Questions", label_zh="未解决问题",
    prompt="List key unresolved questions.",
    required=True,
)

# ── Data resource-specific ──

_F_DATA_DESCRIPTION = FieldDef(
    name="data_description", kind="text",
    label_en="Data Description", label_zh="数据描述",
    prompt="Describe the data: type, source, coverage.",
    required=True,
)

_F_GENERATION_METHODS = FieldDef(
    name="generation_methods", kind="text",
    label_en="Generation Methods", label_zh="数据生成方法",
    prompt="Describe the experimental and computational workflows for data generation.",
    required=True,
)

_F_QUALITY_CONTROL = FieldDef(
    name="quality_control", kind="text",
    label_en="Quality Control", label_zh="质量控制",
    prompt="Describe quality control: metrics, filtering criteria, validation.",
    required=True,
)

_F_DATA_STATISTICS = FieldDef(
    name="data_statistics", kind="key_value_table",
    label_en="Data Statistics", label_zh="数据统计",
    prompt="Provide data scale statistics.",
    required=True,
    columns=_COL_KV,
)

_F_ACCESS_AND_FORMAT = FieldDef(
    name="access_and_format", kind="text",
    label_en="Access & Format", label_zh="访问与格式",
    prompt="Describe access methods, data format, API/download links.",
    required=True,
)

_F_VALIDATION_EXAMPLES = FieldDef(
    name="validation_examples", kind="text",
    label_en="Validation Examples", label_zh="验证示例",
    prompt="Describe how the data quality was validated.",
    required=False,
)

_F_REUSE_POTENTIAL = FieldDef(
    name="reuse_potential", kind="text",
    label_en="Reuse Potential", label_zh="应用潜力",
    prompt="Describe potential application scenarios for the data.",
    required=False,
)

# ── Method protocol-specific ──

_F_METHOD_OVERVIEW = FieldDef(
    name="method_overview", kind="text",
    label_en="Method Overview", label_zh="方法概述",
    prompt="Describe the method: principle and application scope.",
    required=True,
)

_F_PROTOCOL_STEPS = FieldDef(
    name="protocol_steps", kind="structured_list",
    label_en="Protocol Steps", label_zh="协议步骤",
    prompt="List key steps: step number, operation description, time, "
           "critical notes.",
    required=True,
    columns=_COL_STRUCTURED,
)

_F_REQUIRED_MATERIALS = FieldDef(
    name="required_materials", kind="list[str]",
    label_en="Required Materials", label_zh="所需材料",
    prompt="List required materials, reagents, and equipment.",
    required=True,
)

_F_TROUBLESHOOTING = FieldDef(
    name="troubleshooting", kind="key_value_table",
    label_en="Troubleshooting", label_zh="故障排除",
    prompt="List common problems and their solutions.",
    required=False,
    columns=_COL_KV,
)

_F_EXPECTED_RESULTS = FieldDef(
    name="expected_results", kind="text",
    label_en="Expected Results", label_zh="预期结果",
    prompt="Describe expected results and typical outputs.",
    required=False,
)

_F_ADVANTAGES_OVER_ALTERNATIVES = FieldDef(
    name="advantages_over_alternatives", kind="text",
    label_en="Advantages Over Alternatives", label_zh="相比替代方案的优势",
    prompt="Describe advantages compared to alternative protocols/methods.",
    required=False,
)

_F_APPLICATIONS = FieldDef(
    name="applications", kind="list[str]",
    label_en="Applications", label_zh="应用场景",
    prompt="List scenarios where this protocol has been successfully applied.",
    required=False,
)

# ══════════════════════════════════════════════════════════════════════
# Common section helpers
# ══════════════════════════════════════════════════════════════════════

def _overview_section() -> SectionDef:
    return SectionDef(
        name="overview", level=1,
        title_en="Paper Overview", title_zh="论文概览",
        fields=["meta:title", "meta:authors", "meta:year",
                "meta:citation_count", "meta:url"],
    )

def _contrib_limits_section() -> SectionDef:
    return SectionDef(
        name="contrib_limits", level=1,
        title_en="Contributions & Limitations", title_zh="贡献与局限性",
        subsections=[
            SectionDef(name="contributions", level=2,
                       title_en="Main Contributions", title_zh="主要贡献",
                       fields=["contributions"]),
            SectionDef(name="limitations", level=2,
                       title_en="Limitations", title_zh="局限性",
                       fields=["limitations"]),
        ],
    )

def _routes_section() -> SectionDef:
    return SectionDef(name="field_routes", level=1,
                      title_en="Field Technical Landscape", title_zh="领域技术路线",
                      condition="routes is not None")

def _comparison_section() -> SectionDef:
    return SectionDef(name="comparison", level=1,
                      title_en="Comparative Analysis", title_zh="对比分析",
                      condition="comparison is not None")

def _references_section() -> SectionDef:
    return SectionDef(name="references", level=1,
                      title_en="Reference Classification", title_zh="参考文献分类",
                      condition="references is not None")

_TAIL_SECTIONS = [_routes_section(), _comparison_section(), _references_section()]

# ══════════════════════════════════════════════════════════════════════
# 1. Experimental (wet-lab)
# ══════════════════════════════════════════════════════════════════════

_EXP_FIELDS = [
    _F_PROBLEM, _F_HYPOTHESIS, _F_MOTIVATION,
    _F_BIOLOGICAL_SYSTEM, _F_EXPERIMENTAL_METHODS, _F_EXPERIMENTAL_DESIGN,
    _F_KEY_FINDINGS, _F_STATISTICAL_ANALYSIS, _F_DATA_AVAILABILITY,
    _F_REPRODUCIBILITY_NOTES, _F_MECHANISTIC_INSIGHT,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_EXP_SECTIONS = [
    _overview_section(),
    SectionDef(name="hypothesis", level=1,
               title_en="Hypothesis & Motivation", title_zh="假设与动机",
               fields=["hypothesis", "motivation"]),
    SectionDef(name="system", level=1,
               title_en="Biological System", title_zh="研究体系",
               fields=["biological_system"]),
    SectionDef(name="methods", level=1,
               title_en="Experimental Methods", title_zh="实验方法",
               subsections=[
                   SectionDef(name="method_list", level=2,
                              title_en="Techniques Used", title_zh="使用技术",
                              fields=["experimental_methods"]),
                   SectionDef(name="design", level=2,
                              title_en="Experimental Design", title_zh="实验设计",
                              fields=["experimental_design"]),
               ]),
    SectionDef(name="findings", level=1,
               title_en="Key Findings", title_zh="核心发现",
               subsections=[
                   SectionDef(name="results", level=2,
                              title_en="Results", title_zh="实验结果",
                              fields=["key_findings"]),
                   SectionDef(name="stats", level=2,
                              title_en="Statistical Analysis", title_zh="统计分析",
                              fields=["statistical_analysis"]),
                   SectionDef(name="mechanism", level=2,
                              title_en="Mechanistic Insight", title_zh="机制解释",
                              fields=["mechanistic_insight"]),
               ]),
    SectionDef(name="data", level=1,
               title_en="Data & Reproducibility", title_zh="数据与复现性",
               fields=["data_availability", "reproducibility_notes"]),
    _contrib_limits_section(),
    *_TAIL_SECTIONS,
]

EXPERIMENTAL_PROFILE = PaperTypeProfile(
    type_name="experimental",
    description="Presents a biological hypothesis and validates it through "
                "wet-lab or dry-lab experiments following hypothesis→methods→"
                "results→discussion structure.",
    fields=_EXP_FIELDS,
    sections=_EXP_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# 2. Computational (bioinformatics tools)
# ══════════════════════════════════════════════════════════════════════

_COMP_FIELDS = [
    _F_PROBLEM, _F_MOTIVATION,
    _F_ALGORITHM_OVERVIEW, _F_INPUT_OUTPUT_SPEC, _F_SOFTWARE_ARCHITECTURE,
    _F_IMPLEMENTATION, _F_BENCHMARK_DATASETS, _F_PERFORMANCE_RESULTS,
    _F_COMPARISONS_TO_EXISTING, _F_AVAILABILITY,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_COMP_SECTIONS = [
    _overview_section(),
    SectionDef(name="problem", level=1,
               title_en="Motivation", title_zh="需求与动机",
               fields=["problem", "motivation"]),
    SectionDef(name="algorithm", level=1,
               title_en="Algorithm", title_zh="算法",
               fields=["algorithm_overview"]),
    SectionDef(name="io", level=1,
               title_en="Input / Output", title_zh="输入输出",
               fields=["input_output_spec"]),
    SectionDef(name="software", level=1,
               title_en="Software Architecture", title_zh="软件架构",
               fields=["software_architecture", "implementation"]),
    SectionDef(name="benchmark", level=1,
               title_en="Benchmark & Performance", title_zh="基准评测",
               subsections=[
                   SectionDef(name="datasets", level=2,
                              title_en="Benchmark Datasets", title_zh="基准数据集",
                              fields=["benchmark_datasets"]),
                   SectionDef(name="perf", level=2,
                              title_en="Performance", title_zh="性能",
                              fields=["performance_results"]),
                   SectionDef(name="comparisons", level=2,
                              title_en="Comparisons to Existing", title_zh="工具对比",
                              fields=["comparisons_to_existing"]),
               ]),
    SectionDef(name="availability", level=1,
               title_en="Availability", title_zh="获取方式",
               fields=["availability"]),
    _contrib_limits_section(),
    *_TAIL_SECTIONS,
]

COMPUTATIONAL_PROFILE = PaperTypeProfile(
    type_name="computational",
    description="Develops bioinformatics tools, algorithms, pipelines, or "
                "database query systems.",
    fields=_COMP_FIELDS,
    sections=_COMP_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# 3. Review
# ══════════════════════════════════════════════════════════════════════

_REVIEW_FIELDS = [
    _F_PROBLEM, _F_SCOPE, _F_KEY_INSIGHT,
    _F_TAXONOMY_OR_FRAMEWORK, _F_KEY_PAPERS_ANALYZED,
    _F_META_ANALYSIS_METHODS, _F_META_ANALYSIS_RESULTS,
    _F_TRENDS, _F_OPEN_QUESTIONS,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_REVIEW_SECTIONS = [
    _overview_section(),
    SectionDef(name="scope", level=1,
               title_en="Scope & Problem", title_zh="范围与问题",
               fields=["problem", "scope"]),
    SectionDef(name="framework", level=1,
               title_en="Organizing Framework", title_zh="组织框架",
               subsections=[
                   SectionDef(name="tax", level=2,
                              title_en="Taxonomy", title_zh="分类体系",
                              fields=["taxonomy_or_framework"]),
                   SectionDef(name="key_papers", level=2,
                              title_en="Key Papers", title_zh="重点论文",
                              fields=["key_papers_analyzed"]),
               ]),
    SectionDef(name="meta", level=1,
               title_en="Meta-analysis", title_zh="Meta分析",
               fields=["meta_analysis_methods", "meta_analysis_results"]),
    SectionDef(name="insights", level=1,
               title_en="Trends & Open Questions", title_zh="趋势与开放问题",
               fields=["trends", "open_questions"]),
    SectionDef(name="conclusion", level=1,
               title_en="Key Insight", title_zh="核心发现",
               fields=["key_insight"]),
    _contrib_limits_section(),
    *_TAIL_SECTIONS,
]

REVIEW_PROFILE = PaperTypeProfile(
    type_name="review",
    description="Systematic literature review, integration, or quantitative "
                "meta-analysis of biological findings.",
    fields=_REVIEW_FIELDS,
    sections=_REVIEW_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# 4. Data Resource
# ══════════════════════════════════════════════════════════════════════

_DATA_RESOURCE_FIELDS = [
    _F_PROBLEM, _F_MOTIVATION,
    _F_DATA_DESCRIPTION, _F_GENERATION_METHODS, _F_QUALITY_CONTROL,
    _F_DATA_STATISTICS, _F_ACCESS_AND_FORMAT,
    _F_VALIDATION_EXAMPLES, _F_REUSE_POTENTIAL,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_DATA_RESOURCE_SECTIONS = [
    _overview_section(),
    SectionDef(name="problem", level=1,
               title_en="Motivation", title_zh="动机",
               fields=["problem", "motivation"]),
    SectionDef(name="data", level=1,
               title_en="Data Description", title_zh="数据描述",
               subsections=[
                   SectionDef(name="desc", level=2,
                              title_en="Overview", title_zh="概述",
                              fields=["data_description"]),
                   SectionDef(name="gen", level=2,
                              title_en="Generation Methods", title_zh="生成方法",
                              fields=["generation_methods"]),
                   SectionDef(name="qc", level=2,
                              title_en="Quality Control", title_zh="质量控制",
                              fields=["quality_control"]),
                   SectionDef(name="stats", level=2,
                              title_en="Statistics", title_zh="统计",
                              fields=["data_statistics"]),
               ]),
    SectionDef(name="access", level=1,
               title_en="Access & Usage", title_zh="访问与使用",
               subsections=[
                   SectionDef(name="fmt", level=2,
                              title_en="Access & Format", title_zh="访问与格式",
                              fields=["access_and_format"]),
                   SectionDef(name="validation", level=2,
                              title_en="Validation", title_zh="验证",
                              fields=["validation_examples"]),
                   SectionDef(name="reuse", level=2,
                              title_en="Reuse Potential", title_zh="应用潜力",
                              fields=["reuse_potential"]),
               ]),
    _contrib_limits_section(),
    *_TAIL_SECTIONS,
]

DATA_RESOURCE_PROFILE = PaperTypeProfile(
    type_name="data_resource",
    description="Releases large-scale biological datasets, genomic/proteomic "
                "resources, expression atlases, etc.",
    fields=_DATA_RESOURCE_FIELDS,
    sections=_DATA_RESOURCE_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# 5. Method Protocol
# ══════════════════════════════════════════════════════════════════════

_METHOD_PROTOCOL_FIELDS = [
    _F_PROBLEM, _F_METHOD_OVERVIEW, _F_PROTOCOL_STEPS,
    _F_REQUIRED_MATERIALS, _F_TROUBLESHOOTING,
    _F_EXPECTED_RESULTS, _F_ADVANTAGES_OVER_ALTERNATIVES,
    _F_APPLICATIONS,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_METHOD_PROTOCOL_SECTIONS = [
    _overview_section(),
    SectionDef(name="overview", level=1,
               title_en="Method Overview", title_zh="方法概述",
               fields=["problem", "method_overview"]),
    SectionDef(name="protocol", level=1,
               title_en="Protocol Steps", title_zh="协议步骤",
               fields=["protocol_steps"]),
    SectionDef(name="materials", level=1,
               title_en="Materials", title_zh="材料与设备",
               fields=["required_materials"]),
    SectionDef(name="troubleshooting", level=1,
               title_en="Troubleshooting & Expected Results",
               title_zh="故障排除与预期结果",
               fields=["troubleshooting", "expected_results"]),
    SectionDef(name="advantages", level=1,
               title_en="Advantages & Applications", title_zh="优势与应用",
               fields=["advantages_over_alternatives", "applications"]),
    _contrib_limits_section(),
    *_TAIL_SECTIONS,
]

METHOD_PROTOCOL_PROFILE = PaperTypeProfile(
    type_name="method_protocol",
    description="Detailed experimental technique or analysis protocol, "
                "similar to Nature Protocols style.",
    fields=_METHOD_PROTOCOL_FIELDS,
    sections=_METHOD_PROTOCOL_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# Domain
# ══════════════════════════════════════════════════════════════════════

BIOLOGY_DOMAIN = DomainProfile(
    domain_name="biology",
    domain_description="Molecular biology, genetics, biochemistry, "
                       "bioinformatics, and related life sciences.",
    paper_types=[
        EXPERIMENTAL_PROFILE,
        COMPUTATIONAL_PROFILE,
        REVIEW_PROFILE,
        DATA_RESOURCE_PROFILE,
        METHOD_PROTOCOL_PROFILE,
    ],
    default_paper_type="experimental",
)
