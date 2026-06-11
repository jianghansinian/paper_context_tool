"""Materials Science domain profile.

Defines 5 paper types for materials science:
  experimental, computational, review, theory, data_benchmark
"""

from __future__ import annotations

from domains.base import (ColumnDef, DomainProfile, FieldDef, PaperTypeProfile,
                           SectionDef)

# ══════════════════════════════════════════════════════════════════════
# Shared fields
# ══════════════════════════════════════════════════════════════════════

_F_PROBLEM = FieldDef(
    name="problem", kind="text", label_en="Problem", label_zh="材料学问题",
    prompt="What material or materials-science problem does this study address?",
    required=True,
)

_F_MOTIVATION = FieldDef(
    name="motivation", kind="text", label_en="Motivation", label_zh="动机",
    prompt="Why is this material/approach important? What gap does it fill?",
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
    prompt="List limitations acknowledged by the authors or inherent to the approach.",
    required=True,
)

# ── Common columns ──

_COL_STRUCTURED = [
    ColumnDef(name="title", label_en="Item", label_zh="项目"),
    ColumnDef(name="description", label_en="Description", label_zh="描述"),
    ColumnDef(name="papers", label_en="References", label_zh="参考"),
]

_COL_RESULT = [
    ColumnDef(name="dataset", label_en="Property / Condition", label_zh="性质/条件"),
    ColumnDef(name="metric", label_en="Value / Observation", label_zh="数值/观察"),
    ColumnDef(name="value", label_en="Unit / Standard", label_zh="单位/标准"),
    ColumnDef(name="comparison", label_en="Comparison / Note", label_zh="对比/备注"),
]

_COL_KV = [
    ColumnDef(name="key", label_en="Parameter", label_zh="参数"),
    ColumnDef(name="value", label_en="Value", label_zh="数值"),
]

_COL_FORMULA = [
    ColumnDef(name="id", label_en="Eq.", label_zh="方程"),
    ColumnDef(name="expression", label_en="Expression", label_zh="表达式"),
    ColumnDef(name="meaning", label_en="Physical Meaning", label_zh="物理含义"),
    ColumnDef(name="conditions", label_en="Applicable Conditions", label_zh="适用范围"),
]

# ── Experimental-specific fields ──

_F_MATERIAL_SYSTEM = FieldDef(
    name="material_system", kind="text",
    label_en="Material System", label_zh="材料体系",
    prompt="Describe the material system: chemical composition, crystal "
           "structure, phases, doping, etc.",
    required=True,
)

_F_SYNTHESIS_METHOD = FieldDef(
    name="synthesis_method", kind="text",
    label_en="Synthesis Method", label_zh="合成方法",
    prompt="Describe the synthesis method: precursors, steps, critical conditions "
           "(temperature, pressure, time, atmosphere).",
    required=True,
)

_F_PROCESSING_CONDITIONS = FieldDef(
    name="processing_conditions", kind="key_value_table",
    label_en="Processing Conditions", label_zh="加工条件",
    prompt="List key processing parameters and their values. "
           "Return null if not detailed.",
    required=False,
    columns=_COL_KV,
)

_F_CHARACTERIZATION_TECHNIQUES = FieldDef(
    name="characterization_techniques", kind="structured_list",
    label_en="Characterization Techniques", label_zh="表征技术",
    prompt="List each characterization technique: name, purpose, key parameters, "
           "and main findings.",
    required=True,
    columns=_COL_STRUCTURED,
)

_F_STRUCTURE_PROPERTIES = FieldDef(
    name="structure_properties", kind="text",
    label_en="Structure & Properties", label_zh="结构与性能",
    prompt="Describe structural features: microstructure, phase composition, "
           "morphology, grain size, defects, etc.",
    required=True,
)

_F_PERFORMANCE_METRICS = FieldDef(
    name="performance_metrics", kind="result_table",
    label_en="Performance Metrics", label_zh="性能指标",
    prompt="Summarize performance metrics: test conditions, measured values, "
           "comparison with literature benchmarks.",
    required=True,
    columns=_COL_RESULT,
)

_F_STRUCTURE_PROPERTY_RELATIONSHIP = FieldDef(
    name="structure_property_relationship", kind="text",
    label_en="Structure-Property Relationship", label_zh="构效关系",
    prompt="Discuss the relationship between structure/microstructure and the "
           "observed properties. Return null if not explicitly discussed.",
    required=False,
)

# ── Computational-specific fields ──

_F_COMPUTATIONAL_METHOD = FieldDef(
    name="computational_method", kind="text",
    label_en="Computational Method", label_zh="计算方法",
    prompt="Describe the computational method: DFT, MD, Phase-field, ML, etc. "
           "Include specific settings and rationale.",
    required=True,
)

_F_SOFTWARE_PACKAGE = FieldDef(
    name="software_package", kind="text",
    label_en="Software Package", label_zh="软件包",
    prompt="List software packages and versions used. "
           "Return null if not specified.",
    required=False,
)

_F_MODEL_SYSTEM = FieldDef(
    name="model_system", kind="text",
    label_en="Model System", label_zh="模拟体系",
    prompt="Describe the model system: number of atoms, supercell size, "
           "boundary conditions, etc.",
    required=True,
)

_F_COMPUTATIONAL_PARAMETERS = FieldDef(
    name="computational_parameters", kind="key_value_table",
    label_en="Computational Parameters", label_zh="计算参数",
    prompt="List key computational parameters: functional, pseudopotential, "
           "cutoff energy, k-points, force field, etc.",
    required=True,
    columns=_COL_KV,
)

_F_CALCULATED_PROPERTIES = FieldDef(
    name="calculated_properties", kind="result_table",
    label_en="Calculated Properties", label_zh="计算性质",
    prompt="Summarize calculated properties and their numerical values.",
    required=True,
    columns=_COL_RESULT,
)

_F_VALIDATION = FieldDef(
    name="validation", kind="text",
    label_en="Validation", label_zh="方法验证",
    prompt="Describe validation of computational results against experiments "
           "or other calculations. Return null if not performed.",
    required=False,
)

_F_MECHANISTIC_INSIGHT = FieldDef(
    name="mechanistic_insight", kind="text",
    label_en="Mechanistic Insight", label_zh="机制理解",
    prompt="Describe atomic/electronic-level mechanistic insights gained from "
           "the calculations. Return null if not discussed.",
    required=False,
)

# ── Review-specific fields ──

_F_SCOPE = FieldDef(
    name="scope", kind="text", label_en="Scope", label_zh="范围",
    prompt="Define the scope: materials class, phenomena, and time range covered.",
    required=True,
)

_F_TAXONOMY_OR_FRAMEWORK = FieldDef(
    name="taxonomy_or_framework", kind="text",
    label_en="Organizing Framework", label_zh="分类框架",
    prompt="Describe the organizational framework or taxonomy used to structure "
           "the review (by material type, mechanism, method, etc.).",
    required=True,
)

_F_KEY_ADVANCES = FieldDef(
    name="key_advances", kind="structured_list",
    label_en="Key Advances", label_zh="关键进展",
    prompt="List key research advances covered in the review.",
    required=False,
    columns=_COL_STRUCTURED,
)

_F_COMPARATIVE_ANALYSIS = FieldDef(
    name="comparative_analysis", kind="result_table",
    label_en="Comparative Analysis", label_zh="对比分析",
    prompt="Compare performance/properties across different materials or methods.",
    required=False,
    columns=_COL_RESULT,
)

_F_CHALLENGES = FieldDef(
    name="challenges", kind="list[str]",
    label_en="Challenges", label_zh="当前挑战",
    prompt="List current challenges identified in the field.",
    required=True,
)

_F_FUTURE_DIRECTIONS = FieldDef(
    name="future_directions", kind="list[str]",
    label_en="Future Directions", label_zh="未来方向",
    prompt="List proposed future research directions.",
    required=False,
)

# ── Theory-specific fields ──

_F_THEORETICAL_FRAMEWORK = FieldDef(
    name="theoretical_framework", kind="text",
    label_en="Theoretical Framework", label_zh="理论框架",
    prompt="Describe the theoretical foundation: tools, analytical framework, "
           "and mathematical approach.",
    required=True,
)

_F_KEY_DEFINITIONS = FieldDef(
    name="key_definitions", kind="key_value_table",
    label_en="Key Definitions", label_zh="关键定义",
    prompt="List key definitions, notation, and symbol conventions.",
    required=True,
    columns=_COL_KV,
)

_F_MODEL_EQUATIONS = FieldDef(
    name="model_equations", kind="formula_table",
    label_en="Model Equations", label_zh="模型方程",
    prompt="List key model equations: expression, physical meaning, "
           "and applicable conditions.",
    required=True,
    columns=_COL_FORMULA,
)

_F_ASSUMPTIONS = FieldDef(
    name="assumptions", kind="list[str]",
    label_en="Assumptions", label_zh="模型假设",
    prompt="List the model assumptions and simplifications.",
    required=True,
)

_F_MODEL_PREDICTIONS = FieldDef(
    name="model_predictions", kind="result_table",
    label_en="Model Predictions", label_zh="模型预测",
    prompt="Compare model predictions against experimental or simulation data. "
           "Return null if no validation performed.",
    required=False,
    columns=_COL_RESULT,
)

_F_APPLICABLE_RANGE = FieldDef(
    name="applicable_range", kind="text",
    label_en="Applicable Range", label_zh="适用范围",
    prompt="Describe the applicable range and limitations of the model. "
           "Return null if not explicitly stated.",
    required=False,
)

# ── Data/benchmark-specific fields ──

_F_DATA_DESCRIPTION = FieldDef(
    name="data_description", kind="text",
    label_en="Data Description", label_zh="数据描述",
    prompt="Describe the data: material types, properties, source, and coverage.",
    required=True,
)

_F_DATA_GENERATION = FieldDef(
    name="data_generation", kind="text",
    label_en="Data Generation", label_zh="数据生成",
    prompt="Describe how the data was generated: experiments, computations, "
           "or literature mining.",
    required=True,
)

_F_DATA_STATISTICS = FieldDef(
    name="data_statistics", kind="key_value_table",
    label_en="Data Statistics", label_zh="数据统计",
    prompt="Provide data statistics: number of entries, feature dimensions, "
           "distributions.",
    required=True,
    columns=_COL_KV,
)

_F_QUALITY_METRICS = FieldDef(
    name="quality_metrics", kind="text",
    label_en="Quality Metrics", label_zh="质量评估",
    prompt="Describe data quality assessment and validation methods. "
           "Return null if not available.",
    required=False,
)

_F_BENCHMARK_RESULTS = FieldDef(
    name="benchmark_results", kind="result_table",
    label_en="Benchmark Results", label_zh="基准结果",
    prompt="Provide baseline model results on this dataset if applicable. "
           "Return null if not a benchmark dataset.",
    required=False,
    columns=_COL_RESULT,
)

_F_ACCESS_AND_USAGE = FieldDef(
    name="access_and_usage", kind="text",
    label_en="Access & Usage", label_zh="访问与使用",
    prompt="Describe how to access and use the data: URL, API, format, license.",
    required=True,
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
# 1. Experimental
# ══════════════════════════════════════════════════════════════════════

_EXP_FIELDS = [
    _F_PROBLEM, _F_MOTIVATION,
    _F_MATERIAL_SYSTEM, _F_SYNTHESIS_METHOD, _F_PROCESSING_CONDITIONS,
    _F_CHARACTERIZATION_TECHNIQUES, _F_STRUCTURE_PROPERTIES,
    _F_PERFORMANCE_METRICS, _F_STRUCTURE_PROPERTY_RELATIONSHIP,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_EXP_SECTIONS = [
    _overview_section(),
    SectionDef(name="motivation", level=1,
               title_en="Motivation & Problem", title_zh="动机与问题",
               fields=["problem", "motivation"]),
    SectionDef(name="material_system", level=1,
               title_en="Material System", title_zh="材料体系",
               fields=["material_system"]),
    SectionDef(name="synthesis", level=1,
               title_en="Synthesis & Processing", title_zh="合成与加工",
               subsections=[
                   SectionDef(name="synth_method", level=2,
                              title_en="Synthesis Method", title_zh="合成方法",
                              fields=["synthesis_method"]),
                   SectionDef(name="proc_cond", level=2,
                              title_en="Processing Conditions", title_zh="加工条件",
                              fields=["processing_conditions"]),
               ]),
    SectionDef(name="characterization", level=1,
               title_en="Characterization", title_zh="表征",
               subsections=[
                   SectionDef(name="techniques", level=2,
                              title_en="Techniques", title_zh="表征技术",
                              fields=["characterization_techniques"]),
                   SectionDef(name="structure", level=2,
                              title_en="Structure & Properties", title_zh="结构与性能",
                              fields=["structure_properties"]),
               ]),
    SectionDef(name="performance", level=1,
               title_en="Performance", title_zh="性能",
               subsections=[
                   SectionDef(name="metrics", level=2,
                              title_en="Performance Metrics", title_zh="性能指标",
                              fields=["performance_metrics"]),
                   SectionDef(name="sp_relationship", level=2,
                              title_en="Structure-Property Relationship",
                              title_zh="构效关系",
                              fields=["structure_property_relationship"]),
               ]),
    _contrib_limits_section(),
    *_TAIL_SECTIONS,
]

EXPERIMENTAL_PROFILE = PaperTypeProfile(
    type_name="experimental",
    description="Synthesizes materials experimentally and characterizes "
                "structure and performance through measurements.",
    fields=_EXP_FIELDS,
    sections=_EXP_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# 2. Computational
# ══════════════════════════════════════════════════════════════════════

_COMP_FIELDS = [
    _F_PROBLEM, _F_MOTIVATION,
    _F_COMPUTATIONAL_METHOD, _F_SOFTWARE_PACKAGE, _F_MODEL_SYSTEM,
    _F_COMPUTATIONAL_PARAMETERS, _F_CALCULATED_PROPERTIES,
    _F_VALIDATION, _F_MECHANISTIC_INSIGHT,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_COMP_SECTIONS = [
    _overview_section(),
    SectionDef(name="motivation", level=1,
               title_en="Motivation & Problem", title_zh="动机与问题",
               fields=["problem", "motivation"]),
    SectionDef(name="method", level=1,
               title_en="Computational Method", title_zh="计算方法",
               subsections=[
                   SectionDef(name="comp_method", level=2,
                              title_en="Method & Software", title_zh="方法与软件",
                              fields=["computational_method", "software_package"]),
                   SectionDef(name="model_system", level=2,
                              title_en="Model System", title_zh="模拟体系",
                              fields=["model_system"]),
                   SectionDef(name="params", level=2,
                              title_en="Computational Parameters", title_zh="计算参数",
                              fields=["computational_parameters"]),
               ]),
    SectionDef(name="results", level=1,
               title_en="Results & Validation", title_zh="结果与验证",
               subsections=[
                   SectionDef(name="calc_props", level=2,
                              title_en="Calculated Properties", title_zh="计算性质",
                              fields=["calculated_properties"]),
                   SectionDef(name="validation", level=2,
                              title_en="Validation", title_zh="方法验证",
                              fields=["validation"]),
                   SectionDef(name="mechanism", level=2,
                              title_en="Mechanistic Insight", title_zh="机制理解",
                              fields=["mechanistic_insight"]),
               ]),
    _contrib_limits_section(),
    *_TAIL_SECTIONS,
]

COMPUTATIONAL_PROFILE = PaperTypeProfile(
    type_name="computational",
    description="Uses first-principles, molecular dynamics, phase-field, or "
                "machine learning methods to study material properties.",
    fields=_COMP_FIELDS,
    sections=_COMP_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# 3. Review
# ══════════════════════════════════════════════════════════════════════

_REVIEW_FIELDS = [
    _F_PROBLEM, _F_SCOPE, _F_TAXONOMY_OR_FRAMEWORK,
    _F_KEY_ADVANCES, _F_COMPARATIVE_ANALYSIS,
    _F_CHALLENGES, _F_FUTURE_DIRECTIONS,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_REVIEW_SECTIONS = [
    _overview_section(),
    SectionDef(name="scope", level=1,
               title_en="Scope & Problem", title_zh="范围与问题",
               fields=["problem", "scope"]),
    SectionDef(name="framework", level=1,
               title_en="Organizing Framework", title_zh="分类框架",
               subsections=[
                   SectionDef(name="tax", level=2,
                              title_en="Taxonomy", title_zh="分类体系",
                              fields=["taxonomy_or_framework"]),
                   SectionDef(name="advances", level=2,
                              title_en="Key Advances", title_zh="关键进展",
                              fields=["key_advances"]),
               ]),
    SectionDef(name="analysis", level=1,
               title_en="Analysis & Comparison", title_zh="分析与对比",
               subsections=[
                   SectionDef(name="comparative", level=2,
                              title_en="Comparative Analysis", title_zh="对比分析",
                              fields=["comparative_analysis"]),
                   SectionDef(name="challenges", level=2,
                              title_en="Challenges", title_zh="当前挑战",
                              fields=["challenges"]),
               ]),
    SectionDef(name="outlook", level=1,
               title_en="Future Directions", title_zh="未来方向",
               fields=["future_directions"]),
    _contrib_limits_section(),
    *_TAIL_SECTIONS,
]

REVIEW_PROFILE = PaperTypeProfile(
    type_name="review",
    description="Systematic review of research progress on a specific class "
                "of materials or phenomena.",
    fields=_REVIEW_FIELDS,
    sections=_REVIEW_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# 4. Theory
# ══════════════════════════════════════════════════════════════════════

_THEORY_FIELDS = [
    _F_PROBLEM, _F_MOTIVATION,
    _F_THEORETICAL_FRAMEWORK, _F_KEY_DEFINITIONS, _F_MODEL_EQUATIONS,
    _F_ASSUMPTIONS, _F_MODEL_PREDICTIONS, _F_VALIDATION,
    _F_APPLICABLE_RANGE,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_THEORY_SECTIONS = [
    _overview_section(),
    SectionDef(name="motivation", level=1,
               title_en="Motivation & Problem", title_zh="动机与问题",
               fields=["problem", "motivation"]),
    SectionDef(name="theory", level=1,
               title_en="Theoretical Framework", title_zh="理论框架",
               subsections=[
                   SectionDef(name="framework", level=2,
                              title_en="Framework", title_zh="理论基础",
                              fields=["theoretical_framework"]),
                   SectionDef(name="definitions", level=2,
                              title_en="Key Definitions", title_zh="关键定义",
                              fields=["key_definitions"]),
                   SectionDef(name="equations", level=2,
                              title_en="Model Equations", title_zh="模型方程",
                              fields=["model_equations"]),
                   SectionDef(name="assumptions", level=2,
                              title_en="Assumptions", title_zh="假设条件",
                              fields=["assumptions"]),
               ]),
    SectionDef(name="validation", level=1,
               title_en="Validation & Predictions", title_zh="验证与预测",
               subsections=[
                   SectionDef(name="predictions", level=2,
                              title_en="Model Predictions", title_zh="模型预测",
                              fields=["model_predictions"]),
                   SectionDef(name="validation", level=2,
                              title_en="Validation", title_zh="模型验证",
                              fields=["validation"]),
                   SectionDef(name="range", level=2,
                              title_en="Applicable Range", title_zh="适用范围",
                              fields=["applicable_range"]),
               ]),
    _contrib_limits_section(),
    *_TAIL_SECTIONS,
]

THEORY_PROFILE = PaperTypeProfile(
    type_name="theory",
    description="Proposes new theoretical models, constitutive relations, or "
                "analytical frameworks to describe and predict material behavior.",
    fields=_THEORY_FIELDS,
    sections=_THEORY_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# 5. Data / Benchmark
# ══════════════════════════════════════════════════════════════════════

_DATA_BENCHMARK_FIELDS = [
    _F_PROBLEM,
    _F_DATA_DESCRIPTION, _F_DATA_GENERATION, _F_DATA_STATISTICS,
    _F_QUALITY_METRICS, _F_BENCHMARK_RESULTS,
    _F_ACCESS_AND_USAGE,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_DATA_BENCHMARK_SECTIONS = [
    _overview_section(),
    SectionDef(name="problem", level=1,
               title_en="Motivation", title_zh="动机",
               fields=["problem"]),
    SectionDef(name="data", level=1,
               title_en="Data Resource", title_zh="数据资源",
               subsections=[
                   SectionDef(name="desc", level=2,
                              title_en="Data Description", title_zh="数据描述",
                              fields=["data_description"]),
                   SectionDef(name="gen", level=2,
                              title_en="Generation Methods", title_zh="数据生成",
                              fields=["data_generation"]),
                   SectionDef(name="stats", level=2,
                              title_en="Statistics", title_zh="数据统计",
                              fields=["data_statistics"]),
                   SectionDef(name="quality", level=2,
                              title_en="Quality Metrics", title_zh="质量评估",
                              fields=["quality_metrics"]),
               ]),
    SectionDef(name="benchmark", level=1,
               title_en="Benchmark", title_zh="基准测试",
               fields=["benchmark_results"]),
    SectionDef(name="access", level=1,
               title_en="Access & Usage", title_zh="访问与使用",
               fields=["access_and_usage"]),
    _contrib_limits_section(),
    *_TAIL_SECTIONS,
]

DATA_BENCHMARK_PROFILE = PaperTypeProfile(
    type_name="data_benchmark",
    description="Releases materials databases, high-throughput screening results, "
                "or machine learning benchmark datasets.",
    fields=_DATA_BENCHMARK_FIELDS,
    sections=_DATA_BENCHMARK_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# Domain
# ══════════════════════════════════════════════════════════════════════

MATERIALS_SCIENCE_DOMAIN = DomainProfile(
    domain_name="materials_science",
    domain_description="Materials synthesis, characterization, computational "
                       "materials science, and materials theory.",
    paper_types=[
        EXPERIMENTAL_PROFILE,
        COMPUTATIONAL_PROFILE,
        REVIEW_PROFILE,
        THEORY_PROFILE,
        DATA_BENCHMARK_PROFILE,
    ],
    default_paper_type="experimental",
)
