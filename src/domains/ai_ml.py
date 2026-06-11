"""AI / ML domain profile — the first (and default) domain.

Defines 6 paper types.  Only ``experimental`` is active in Phase A; the
remaining 5 types will be activated in Phase C when paper-type detection is
wired in.
"""

from __future__ import annotations

from domains.base import (ColumnDef, DomainProfile, FieldDef, PaperTypeProfile,
                           SectionDef)

# ══════════════════════════════════════════════════════════════════════
# Shared column definitions
# ══════════════════════════════════════════════════════════════════════

_COL_COMPONENT = [
    ColumnDef(name="name", label_en="Component", label_zh="组件"),
    ColumnDef(name="purpose", label_en="Purpose", label_zh="功能"),
    ColumnDef(name="details", label_en="Implementation Details", label_zh="实现细节"),
    ColumnDef(name="referenced_figure", label_en="Figure", label_zh="对应图表"),
]

_COL_FORMULA = [
    ColumnDef(name="name", label_en="Formula", label_zh="公式"),
    ColumnDef(name="latex", label_en="Expression", label_zh="表达式"),
    ColumnDef(name="explanation", label_en="Explanation", label_zh="含义"),
    ColumnDef(name="significance", label_en="Significance", label_zh="重要性"),
]

_COL_RESULT = [
    ColumnDef(name="dataset", label_en="Dataset", label_zh="数据集"),
    ColumnDef(name="metric", label_en="Metric", label_zh="指标"),
    ColumnDef(name="value", label_en="Value", label_zh="数值"),
    ColumnDef(name="comparison", label_en="vs Baseline", label_zh="对比Baseline"),
]

_COL_ABLATION = [
    ColumnDef(name="configuration", label_en="Configuration", label_zh="配置"),
    ColumnDef(name="impact", label_en="Performance Impact", label_zh="性能影响"),
    ColumnDef(name="insight", label_en="Key Insight", label_zh="关键结论"),
]

_COL_INDUSTRY = [
    ColumnDef(name="dimension", label_en="Dimension", label_zh="维度"),
    ColumnDef(name="traditional", label_en="Traditional Approach", label_zh="传统方案"),
    ColumnDef(name="this_paper", label_en="This Paper", label_zh="本文方案"),
    ColumnDef(name="advantage", label_en="Advantage", label_zh="优势"),
]

# ══════════════════════════════════════════════════════════════════════
# Shared field definitions (used across multiple paper types)
# ══════════════════════════════════════════════════════════════════════

_F_PROBLEM = FieldDef(
    name="problem", kind="text", label_en="Problem", label_zh="问题定义",
    prompt="Identify the CONCRETE TECHNICAL CHALLENGES this paper addresses. "
           "Break them into distinct sub-problems with clear headings. "
           "For each sub-problem: (1) Define it PRECISELY — what is the gap or "
           "bottleneck? (2) What specific failure modes or limitations make "
           "this a hard problem? (3) What happens in practice WITHOUT a "
           "solution — what breaks, what scenarios are unhandled? "
           "Use **bold** for key concept names. Separate sub-problems with "
           "blank lines. "
           "Focus ONLY on defining the problem itself. Do NOT explain why "
           "it matters or describe existing solutions (those go in Motivation). "
           "Be precise and concise — every sentence should define the challenge.",
    required=True,
)

_F_MOTIVATION = FieldDef(
    name="motivation", kind="text", label_en="Motivation", label_zh="动机",
    prompt="Explain WHY solving this problem MATTERS. "
           "The reader has just read the field evolution and problem "
           "definition — do NOT re-describe what each approach category "
           "does or re-list the sub-problems. "
           "Instead, focus on IMPACT and URGENCY: "
           "(1) What is AT STAKE — safety, cost, scalability, real-world "
           "deployment? What real scenarios break without a solution? "
           "(2) What CONSEQUENCES follow if the gap remains — what systems "
           "will continue to fail, and where? "
           "(3) What CAPABILITY would be UNLOCKED if solved — what becomes "
           "possible that was previously impractical? "
           "Write as 1 tight paragraph. Use **bold** sparingly. "
           "Do NOT name individual prior methods (evolution already named "
           "them, related_work_context will detail them). "
           "This is about STAKES, not a second catalog of approaches.",
    required=True,
)

_F_KEY_INSIGHT = FieldDef(
    name="key_insight", kind="text", label_en="Key Insight", label_zh="核心思路",
    prompt="Summarize this paper in exactly 2-3 sentences covering: "
           "(1) WHAT specific problem or bottleneck it addresses, "
           "(2) WHAT approach or method it proposes to solve it, "
           "(3) WHAT key result or new capability this achieves. "
           "Make this self-contained — a reader who reads nothing else "
           "should understand what the paper does and why it matters. "
           "Use **bold** for the central method name.",
    required=True,
)

_F_ARCH_OVERVIEW = FieldDef(
    name="architecture_overview", kind="text",
    label_en="Overall Architecture", label_zh="整体架构",
    prompt="Describe the overall architecture CLEARLY AND CONCISELY. "
           "Start with a one-sentence overview of the full pipeline. "
           "Then, for each major module/component: (1) WHAT it does and "
           "HOW it transforms data — input format, processing steps, "
           "output format. (2) How it CONNECTS to upstream and downstream "
           "modules — what data flows in, what flows out. (3) Include key "
           "mathematical formulas ($...$) inline where they define a "
           "module's core computation. (4) Note training-only vs "
           "inference-only paths if applicable. "
           "Use **bold** for module names. Separate modules with blank "
           "lines for readability. "
           "Focus on WHAT the pipeline does and HOW data flows through it. "
           "Do NOT explain WHY each module was designed this way, what "
           "alternatives were rejected, or what trade-offs were accepted "
           "(that goes in Design Rationale). Do NOT re-describe the "
           "problem (covered in Problem & Motivation). "
           "Aim for clarity — a reader should understand the pipeline "
           "topology at a glance.",
    required=True,
)

_F_ARCH_FIGURE = FieldDef(
    name="architecture_figure", kind="text",
    label_en="Architecture Diagram", label_zh="架构图详解",
    prompt="Explain the architecture diagram (usually Figure 1 or 2) in detail. "
           "Trace the FULL data flow: what enters where, how it is transformed, "
           "and what comes out. For each block in the diagram: identify its "
           "role, its inputs/outputs, and how it connects to other blocks. "
           "Note any training-only vs inference-only paths. "
           "Return null if no such figure exists.",
    required=False,
)

_F_DESIGN_RATIONALE = FieldDef(
    name="design_rationale", kind="text",
    label_en="Design Rationale", label_zh="设计原理",
    prompt="Explain the KEY DESIGN DECISIONS and their INTER-DEPENDENCIES. "
           "This is the \"WHY\" section — why is the architecture the way it is? "
           "(1) Identify the 3-5 most critical design decisions that SHAPE "
           "the rest of the system. "
           "(2) For each: what ALTERNATIVE was considered and WHY was it "
           "REJECTED? What would break if a simpler alternative were used? "
           "(3) TRACE DEPENDENCY CHAINS — the most valuable insight: how does "
           "Design Decision A CREATE THE CONDITIONS for Design Decision B "
           "to work? Example: \"They chose X, which meant they could do Y, "
           "which in turn required Z.\" "
           "(4) What TRADE-OFFS were accepted and why were they acceptable? "
           "Use **bold** for design choices and alternatives. "
           "Write as a coherent narrative about design logic, not a "
           "disconnected list. "
           "Do NOT re-describe WHAT each module does (covered in Architecture). "
           "Do NOT re-describe the problem itself (covered in Problem & "
           "Motivation). Reference the problem and architecture briefly, "
           "but focus on the RATIONALE connecting them.",
    required=False,
)

_F_RELATED_WORK_CONTEXT = FieldDef(
    name="related_work_context", kind="text",
    label_en="Related Work Context", label_zh="相关工作上下文",
    prompt="Position this paper in the research landscape by focusing on "
           "TECHNICAL LINEAGE and DIFFERENTIATION — what it inherits and "
           "what it argues against. "
           "Organize by research direction. For each direction: "
           "(1) Name 1-3 specific papers or methods that this paper "
           "DIRECTLY BUILDS ON — which exact component, technique, or "
           "design principle was reused, and what was CHANGED or improved. "
           "Use **bold** for method/paper names. "
           "(2) Name 1-3 specific alternative approaches that this paper "
           "EXPLICITLY CONTRASTS with — on what technical grounds does it "
           "argue against them? "
           "CRITICAL: Do NOT describe what each research direction DOES "
           "in general — the evolution and motivation sections already "
           "cover the landscape. Assume the reader knows what VA and VLA "
           "models are. Focus ONLY on the specific inheritance and "
           "opposition relationships. "
           "Write CONCISELY — 2-3 sentences per direction. "
           "Return null if the paper does not have a clear related-work "
           "discussion.",
    required=False,
)

_F_DATA_ENGINEERING = FieldDef(
    name="data_engineering", kind="text",
    label_en="Data Engineering", label_zh="数据工程",
    prompt="Describe any data collection, annotation, curation, or generation "
           "methodology. Include scale (dataset size), quality control measures, "
           "labeling pipeline details, and any automated annotation systems. "
           "Return null if not a significant contribution of this paper.",
    required=False,
)

_COL_TRAINING_STAGE = [
    ColumnDef(name="title", label_en="Stage", label_zh="阶段"),
    ColumnDef(name="description", label_en="Description", label_zh="描述"),
]

_F_TRAINING_STAGES = FieldDef(
    name="training_stages", kind="structured_list",
    label_en="Training Stages", label_zh="训练阶段",
    prompt="Break down the training process into distinct stages. "
           "For each stage: (1) TITLE (e.g. 'Stage 1: Pretraining'), "
           "(2) DESCRIPTION including: WHAT is trained and what is frozen, "
           "WHAT data is used and how it is sampled, loss functions with "
           "LaTeX formulas, optimizer settings, key hyper-parameters. "
           "(3) PURPOSE: why is this stage needed? Why in this ORDER? "
           "What would happen if you skipped it or changed the order? "
           "(4) DEPENDENCY: how does this stage SET UP or DEPEND ON "
           "the next/previous stage? What would break if stages were reordered?",
    required=False,
    columns=_COL_TRAINING_STAGE,
)

_F_INTUITIVE_ANALOGY = FieldDef(
    name="intuitive_analogy", kind="text",
    label_en="Intuitive Analogy", label_zh="直观类比",
    prompt="If the paper uses an intuitive analogy or metaphor to explain "
           "its method (e.g., 'mistake notebook', 'two-system brain', "
           "'assembly line'), capture it here. Return null if none.",
    required=False,
)

_F_DEPLOYMENT_ARCHITECTURE = FieldDef(
    name="deployment_architecture", kind="text",
    label_en="Deployment Architecture", label_zh="部署架构",
    prompt="Describe the deployment or system architecture: how is the model "
           "served? What hardware is used? What are the latency/throughput "
           "considerations? Include any system-level optimizations for "
           "real-world deployment. Return null if not discussed.",
    required=False,
)

_F_DEPLOYMENT_VALUE = FieldDef(
    name="deployment_value", kind="text",
    label_en="Practical Value", label_zh="工程落地价值",
    prompt="Analyze the practical deployment and engineering value of this work. "
           "Focus on CONCRETE benefits for real-world systems: "
           "(1) Hardware cost reduction — cheaper sensors, fewer GPUs? "
           "(2) Data cost reduction — less labeled data, cheaper annotations? "
           "(3) Scalability — does it scale to larger datasets or more scenarios? "
           "(4) Production readiness — can it be deployed on edge devices? "
           "What are the latency/throughput implications? "
           "(5) Comparison to industry baseline — what would a company "
           "currently use, and how does this improve upon it? "
           "Use **bold** for key metrics and hardware names. "
           "Return null if the paper does not discuss deployment aspects.",
    required=False,
)

_F_FIELD_EVOLUTION = FieldDef(
    name="field_evolution", kind="text",
    label_en="Technical Evolution", label_zh="技术路线演进",
    prompt="In EXACTLY 2 sentences, trace the technical arc that leads to "
           "this paper's problem. "
           "First sentence: the earliest relevant paradigm and what it "
           "ACHIEVED — then the next paradigm shift and what NEW capability "
           "it brought. Use **bold** for paradigm names. "
           "Second sentence: what SPECIFIC tension or contradiction emerged "
           "between these paradigms — the unresolved gap this paper steps into. "
           "Do NOT enumerate sub-problems (that goes in problem). "
           "Do NOT name individual paper titles or describe their failures "
           "(that goes in motivation and related_work_context). "
           "Do NOT mention this paper's method. "
           "This is PURELY a historical arc — two sentences, no more.",
    required=False,
)

_F_CORE_QUESTION = FieldDef(
    name="core_question", kind="text",
    label_en="Core Research Question", label_zh="核心科学问题",
    prompt="State the SINGLE core research question this paper tries to "
           "answer, in 1-2 sentences. Make it specific, falsifiable, and "
           "focused on the paper's central claim. Frame it as a question "
           "whose answer determines whether the approach succeeds. "
           "Use **bold** for key technical terms. "
           "Do NOT state the answer — that belongs in key_insight. "
           "This field captures the QUESTION the paper asks, not the solution.",
    required=False,
)

_F_EVALUATION_SETUP = FieldDef(
    name="evaluation_setup", kind="text",
    label_en="Evaluation Setup", label_zh="评测设置",
    prompt="Describe the evaluation setup in 2-3 sentences: "
           "(1) What benchmark(s) or dataset(s) are used for evaluation? "
           "(2) What does each key metric MEAN in plain language — how should "
           "a reader interpret a higher or lower value? "
           "(3) What value represents the human expert or industry reference "
           "baseline for each metric? "
           "Do NOT list actual results — just explain the evaluation framework "
           "so the reader can understand the numbers that follow. "
           "Return null if the paper uses standard metrics with no special "
           "explanation needed.",
    required=False,
)

_F_INDUSTRY_COMPARISON = FieldDef(
    name="industry_comparison", kind="result_table",
    label_en="Industry Comparison", label_zh="行业横向对比",
    prompt="Compare this paper's approach against the industry baseline or "
           "dominant paradigm across 4-6 key dimensions. "
           "Choose dimensions that highlight the MEANINGFUL differences — "
           "e.g., inference logic, long-tail scenario handling, "
           "interpretability, real-time performance, human interaction, "
           "performance ceiling. "
           "For each dimension: (1) what the traditional/industry approach "
           "does, (2) what THIS paper does differently, "
           "(3) the concrete advantage or trade-off. "
           "Return null if the paper does not position itself against an "
           "industry baseline or if such a comparison would be speculative.",
    required=False, columns=_COL_INDUSTRY,
)

_F_COMPONENTS = FieldDef(
    name="components", kind="component_table",
    label_en="Core Components", label_zh="核心组件",
    prompt="List EVERY major architectural component. For each: "
           "(1) NAME and PURPOSE — what does it do? "
           "(2) IMPLEMENTATION DETAILS — dimensions, layers, configurations. "
           "(3) INPUT and OUTPUT specification — what data enters and leaves? "
           "(4) REFERENCED FIGURE. "
           "(5) DESIGN RATIONALE — WHY this specific design? What problem "
           "does this component solve within the pipeline? How does it "
           "connect to upstream/downstream components? What would a "
           "simpler alternative miss?",
    required=False, columns=_COL_COMPONENT,
)

_F_FORMULAS = FieldDef(
    name="formulas", kind="formula_table",
    label_en="Key Formulas", label_zh="关键公式",
    prompt="Extract ALL key mathematical formulas. For each: "
           "(1) NAME, (2) LATEX expression, "
           "(3) EXPLANATION — what it computes and WHERE in the architecture "
           "it is used. (4) SIGNIFICANCE — WHY this specific formulation? "
           "What problem does it solve? What would be lost with a simpler "
           "formulation? Be precise about the role each formula plays in "
           "the overall method.",
    required=False, columns=_COL_FORMULA,
)

_F_TRAINING_DATA = FieldDef(
    name="training_data", kind="text", label_en="Training Data", label_zh="训练数据",
    prompt="What datasets were used for training? Describe scale and preprocessing.",
    required=False,
)

_F_LOSS_FUNCTIONS = FieldDef(
    name="loss_functions", kind="list[str]",
    label_en="Loss Functions", label_zh="损失函数",
    prompt="List all loss functions used.",
    required=False,
)

_F_OPTIMIZER = FieldDef(
    name="optimizer", kind="text", label_en="Optimizer", label_zh="优化器",
    prompt="What optimizer and learning rate schedule were used?",
    required=False,
)

_F_TRAINING_PROCEDURE = FieldDef(
    name="training_procedure", kind="text",
    label_en="Training Procedure", label_zh="训练流程",
    prompt="Describe the training procedure in thorough detail. Include: "
           "(1) The COMPLETE TRAINING LOOP — how data flows from environment/"
           "dataset to model and back, step by step. "
           "(2) Any CHECKPOINT, ROLLBACK, or STATE-SNAPSHOT mechanisms — "
           "when are they triggered? what is saved? WHY are they needed? "
           "(3) FAILURE DETECTION and HANDLING — how is failure defined? "
           "what happens when the model fails during training? "
           "(4) Any HUMAN INTERVENTION or CORRECTION mechanisms — how do "
           "humans interact with the training loop? when and how are "
           "corrections recorded? "
           "(5) DATA ORGANIZATION — how are different types of training "
           "data organized (separate buffers? prioritized sampling? "
           "paired samples?) and WHY this organization was chosen. "
           "(6) Key hyper-parameters, optimizer settings, batch sizes, "
           "and training infrastructure. "
           "For each mechanism, explain WHY it exists — what would go "
           "wrong without it.",
    required=False,
)

_F_INFERENCE_PROCEDURE = FieldDef(
    name="inference_procedure", kind="text",
    label_en="Inference Procedure", label_zh="推理流程",
    prompt="Describe the complete inference/forward-pass flow from sensor "
           "input to final output or action. Trace data through EVERY "
           "module. Include: (1) Input preprocessing and sensor configuration. "
           "(2) Each processing stage with dimensions and data formats. "
           "(3) Any post-processing, filtering, or trajectory rollout steps. "
           "(4) Real-world deployment details if available — hardware, "
           "latency, sensor setup, control interface. "
           "Explain what makes this flow deployment-viable or what gaps remain.",
    required=False,
)

_F_POST_PROCESSING = FieldDef(
    name="post_processing", kind="text",
    label_en="Post-processing", label_zh="后处理",
    prompt="Describe any post-processing steps (e.g. NMS, thresholding). "
           "Return null if none.",
    required=False,
)

_F_MAIN_RESULTS = FieldDef(
    name="main_results", kind="result_table",
    label_en="Main Results", label_zh="主要结果",
    prompt="Extract the main experimental results: dataset, metric, "
           "achieved value, and comparison to baselines. "
           "Include context: what does each metric represent in plain "
           "language, and what is the reference baseline (e.g., human "
           "performance, prior SOTA)? "
           "Include ALL rows with complete data.",
    required=False, columns=_COL_RESULT,
)

_F_ABLATION_RESULTS = FieldDef(
    name="ablation_results", kind="result_table",
    label_en="Ablation Studies", label_zh="消融实验",
    prompt="Extract ALL ablation experiments in a structured table. "
           "For each ablation row: (1) CONFIGURATION — what components or "
           "settings were varied? Name the specific configuration (e.g., "
           "'M1 only', 'M1+M2', 'full model'). (2) PERFORMANCE IMPACT — "
           "the exact metric values and the delta vs baseline. Include "
           "multiple metrics if reported (e.g., 'EPDMS 87.8, EC 84.8, D 40%'). "
           "(3) KEY INSIGHT — what does this ablation REVEAL about the method? "
           "Does it confirm a design hypothesis or reveal a surprise? "
           "Include ALL ablation experiments: component ablations, "
           "hyper-parameter sweeps, pre-training comparisons, etc.",
    required=False, columns=_COL_ABLATION,
)

_F_QUALITATIVE_RESULTS = FieldDef(
    name="qualitative_results", kind="text",
    label_en="Qualitative Analysis", label_zh="定性分析",
    prompt="Describe qualitative or visualization results. What do the "
           "visualizations SHOW about the method's behavior? What patterns "
           "or properties are visible that numbers alone don't capture? "
           "How do these results support or illustrate the design rationale? "
           "Return null if no qualitative results are presented.",
    required=False,
)

_F_CONTRIBUTIONS = FieldDef(
    name="contributions", kind="list[str]",
    label_en="Main Contributions", label_zh="主要贡献",
    prompt="List the main contributions of this paper.",
    required=True,
)

_F_LIMITATIONS = FieldDef(
    name="limitations", kind="list[str]",
    label_en="Limitations", label_zh="局限性",
    prompt="List the limitations acknowledged by the authors or obvious from the work.",
    required=True,
)

_F_SYNTHESIS = FieldDef(
    name="synthesis", kind="text",
    label_en="Synthesis & Significance", label_zh="总结与意义",
    prompt="Write a CLOSING SYNTHESIS focused on BROADER IMPLICATIONS. "
           "Do NOT repeat the contributions list or re-describe the method. "
           "Instead: (1) What is the CORE VALUE — what new capability does "
           "this unlock that was previously impossible or impractical? "
           "(2) What does this work MEAN for the broader field — what "
           "ASSUMPTIONS does it challenge? What new RESEARCH DIRECTIONS "
           "does it open? (3) What is the most important TAKEAWAY for a "
           "practitioner — if they only remember ONE thing from this paper, "
           "what should it be? "
           "Write as a concise narrative paragraph. "
           "Do NOT re-trace dependency chains (covered in Design Rationale). "
           "Focus on significance BEYOND this specific paper.\n\n"
           "END with an explicit qualitative judgment: classify this paper "
           "as one of: ★★★ MILESTONE (fundamentally shifts the field), "
           "★★ STRONG CONTRIBUTION (significant advance over SOTA), "
           "★ INCREMENTAL (solid but marginal improvement), or "
           "◆ EXPLORATORY (interesting direction, unvalidated). "
           "State the classification explicitly and justify it in one sentence.",
    required=True,
)

# ══════════════════════════════════════════════════════════════════════
# Paper type: experimental (the default for AI/ML)
# ══════════════════════════════════════════════════════════════════════

_EXPERIMENTAL_FIELDS = [
    _F_FIELD_EVOLUTION,
    _F_PROBLEM,
    _F_MOTIVATION,
    _F_CORE_QUESTION,
    _F_KEY_INSIGHT,
    _F_RELATED_WORK_CONTEXT,
    _F_ARCH_OVERVIEW,
    _F_ARCH_FIGURE,
    _F_COMPONENTS,
    _F_FORMULAS,
    _F_DESIGN_RATIONALE,
    _F_INTUITIVE_ANALOGY,
    _F_TRAINING_DATA,
    _F_DATA_ENGINEERING,
    _F_TRAINING_STAGES,
    _F_LOSS_FUNCTIONS,
    _F_OPTIMIZER,
    _F_TRAINING_PROCEDURE,
    _F_INFERENCE_PROCEDURE,
    _F_POST_PROCESSING,
    _F_DEPLOYMENT_ARCHITECTURE,
    _F_DEPLOYMENT_VALUE,
    _F_EVALUATION_SETUP,
    _F_MAIN_RESULTS,
    _F_ABLATION_RESULTS,
    _F_QUALITATIVE_RESULTS,
    _F_INDUSTRY_COMPARISON,
    _F_CONTRIBUTIONS,
    _F_LIMITATIONS,
    _F_SYNTHESIS,
]

_EXPERIMENTAL_SECTIONS = [
    # ── §1 Paper Overview ──
    SectionDef(
        name="overview", level=1,
        title_en="Paper Overview", title_zh="论文概览",
        fields=["meta:title", "meta:authors", "meta:year",
                "meta:citation_count", "meta:url", "key_insight"],
    ),
    # ── §2 Problem ──
    SectionDef(
        name="problem", level=1,
        title_en="Problem Definition & Motivation", title_zh="问题定义与动机",
        fields=["field_evolution", "problem", "motivation",
                "related_work_context", "core_question"],
    ),
    # ── §3 Method ──
    SectionDef(
        name="method", level=1,
        title_en="Method Architecture", title_zh="方法架构",
        subsections=[
            SectionDef(
                name="arch_overview", level=2,
                title_en="Overall Architecture", title_zh="整体架构",
                fields=["architecture_overview"],
            ),
            SectionDef(
                name="arch_figure", level=2,
                title_en="Architecture Diagram Explanation", title_zh="架构图详解",
                fields=["architecture_figure"],
            ),
            SectionDef(
                name="components", level=2,
                title_en="Core Components", title_zh="核心组件",
                fields=["components"],
            ),
            SectionDef(
                name="design_rationale", level=2,
                title_en="Design Rationale", title_zh="设计原理与直观理解",
                fields=["design_rationale", "intuitive_analogy"],
            ),
            SectionDef(
                name="training", level=2,
                title_en="Training Pipeline", title_zh="训练流程",
                fields=["training_data", "data_engineering", "training_stages",
                        "loss_functions", "optimizer", "training_procedure"],
            ),
            SectionDef(
                name="inference", level=2,
                title_en="Inference Pipeline", title_zh="推理流程",
                fields=["inference_procedure", "post_processing",
                        "deployment_architecture", "deployment_value"],
            ),
        ],
    ),
    # ── §4 Results ──
    SectionDef(
        name="results", level=1,
        title_en="Experimental Results", title_zh="实验结果",
        subsections=[
            SectionDef(
                name="eval_setup", level=2,
                title_en="Evaluation Setup", title_zh="评测设置",
                fields=["evaluation_setup"],
            ),
            SectionDef(
                name="main_results", level=2,
                title_en="Main Results", title_zh="主要结果",
                fields=["main_results"],
            ),
            SectionDef(
                name="ablation", level=2,
                title_en="Ablation Studies", title_zh="消融实验",
                fields=["ablation_results"],
            ),
            SectionDef(
                name="qualitative", level=2,
                title_en="Qualitative Analysis", title_zh="定性分析",
                fields=["qualitative_results"],
            ),
            SectionDef(
                name="industry_comparison", level=2,
                title_en="Industry Comparison", title_zh="行业横向对比",
                fields=["industry_comparison"],
            ),
        ],
    ),
    # ── §5 Contributions & Limitations ──
    SectionDef(
        name="contrib_limits", level=1,
        title_en="Contributions & Limitations", title_zh="贡献与局限性",
        subsections=[
            SectionDef(
                name="contributions", level=2,
                title_en="Main Contributions", title_zh="主要贡献",
                fields=["contributions"],
            ),
            SectionDef(
                name="limitations", level=2,
                title_en="Limitations", title_zh="局限性",
                fields=["limitations"],
            ),
            SectionDef(
                name="synthesis", level=2,
                title_en="Synthesis & Significance", title_zh="总结与意义",
                fields=["synthesis"],
            ),
        ],
    ),
    # ── §6 Field Technical Landscape [condition: routes] ──
    SectionDef(
        name="field_routes", level=1,
        title_en="Field Technical Landscape", title_zh="领域技术路线",
        condition="routes is not None",
    ),
    # ── §7 Comparative Analysis [condition: comparison] ──
    SectionDef(
        name="comparison", level=1,
        title_en="Comparative Analysis", title_zh="对比分析",
        condition="comparison is not None",
    ),
    # ── §8 Reference Classification [condition: references] ──
    SectionDef(
        name="references", level=1,
        title_en="Reference Classification", title_zh="参考文献分类",
        condition="references is not None",
    ),
]

EXPERIMENTAL_PROFILE = PaperTypeProfile(
    type_name="experimental",
    description="Proposes a new method/model/algorithm and validates it "
                "through experiments on standard benchmarks.",
    fields=_EXPERIMENTAL_FIELDS,
    sections=_EXPERIMENTAL_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# Additional shared fields for non-experimental types
# ══════════════════════════════════════════════════════════════════════

_F_SCOPE = FieldDef(
    name="scope", kind="text", label_en="Scope", label_zh="范围界定",
    prompt="Define the scope of this work: time range, sub-areas covered, "
           "inclusion/exclusion criteria.",
    required=True,
)

_F_DATASET_DESCRIPTION = FieldDef(
    name="dataset_description", kind="text",
    label_en="Dataset Description", label_zh="数据集描述",
    prompt="Describe the dataset in detail: data sources, scale, modalities, "
           "annotation methodology.",
    required=True,
)

_F_COLLECTION_METHODOLOGY = FieldDef(
    name="collection_methodology", kind="text",
    label_en="Collection Methodology", label_zh="采集方法",
    prompt="Describe the data collection/annotation methodology: process, "
           "quality control, ethical considerations.",
    required=True,
)

_F_DATASET_STATISTICS = FieldDef(
    name="dataset_statistics", kind="key_value_table",
    label_en="Dataset Statistics", label_zh="数据集统计",
    prompt="Provide key dataset statistics: sample count, class distribution, "
           "average length, etc.",
    required=True,
    columns=[
        ColumnDef(name="key", label_en="Statistic", label_zh="统计项"),
        ColumnDef(name="value", label_en="Value", label_zh="数值"),
    ],
)

_F_TASK_DEFINITION = FieldDef(
    name="task_definition", kind="text",
    label_en="Task Definition", label_zh="任务定义",
    prompt="Define the task: input, output, evaluation criteria.",
    required=True,
)

_F_EVALUATION_PROTOCOL = FieldDef(
    name="evaluation_protocol", kind="text",
    label_en="Evaluation Protocol", label_zh="评估协议",
    prompt="Describe the evaluation protocol: metrics, data splits, "
           "statistical significance testing.",
    required=True,
)

_F_BASELINE_METHODS = FieldDef(
    name="baseline_methods", kind="structured_list",
    label_en="Baseline Methods", label_zh="基线方法",
    prompt="List the methods used as baselines and the rationale for each.",
    required=False,
    columns=[
        ColumnDef(name="title", label_en="Method", label_zh="方法"),
        ColumnDef(name="description", label_en="Rationale", label_zh="选择理由"),
        ColumnDef(name="papers", label_en="References", label_zh="参考文献"),
    ],
)

_F_KNOWN_BIASES = FieldDef(
    name="known_biases", kind="text",
    label_en="Known Biases", label_zh="已知偏差",
    prompt="Describe any known data biases or limitations. Return null if none.",
    required=False,
)

_F_MAINTENANCE_PLAN = FieldDef(
    name="maintenance_plan", kind="text",
    label_en="Maintenance Plan", label_zh="维护计划",
    prompt="Describe long-term maintenance and versioning plans. "
           "Return null if not discussed.",
    required=False,
)

_F_SYSTEM_ARCHITECTURE = FieldDef(
    name="system_architecture", kind="text",
    label_en="System Architecture", label_zh="系统架构",
    prompt="Describe the overall system architecture: modules, data flow, "
           "deployment topology.",
    required=True,
)

_F_DESIGN_DECISIONS = FieldDef(
    name="design_decisions", kind="structured_list",
    label_en="Design Decisions", label_zh="设计决策",
    prompt="List key design decisions: what was chosen, why, and trade-off analysis.",
    required=True,
    columns=[
        ColumnDef(name="title", label_en="Decision", label_zh="决策"),
        ColumnDef(name="description", label_en="Rationale & Trade-offs",
                  label_zh="理由与权衡"),
        ColumnDef(name="papers", label_en="References", label_zh="参考"),
    ],
)

_F_API_DESIGN = FieldDef(
    name="api_design", kind="text",
    label_en="API Design", label_zh="接口设计",
    prompt="Describe the API/interface design. Return null if not applicable.",
    required=False,
)

_F_IMPLEMENTATION_STACK = FieldDef(
    name="implementation_stack", kind="text",
    label_en="Implementation Stack", label_zh="技术栈",
    prompt="Describe the implementation: languages, frameworks, dependencies. "
           "Return null if not specified.",
    required=False,
)

_F_PERFORMANCE_ENGINEERING = FieldDef(
    name="performance_engineering", kind="text",
    label_en="Performance Engineering", label_zh="性能工程",
    prompt="Describe performance optimizations: parallelism, caching, "
           "memory management, compilation tricks.",
    required=False,
)

_F_SCALABILITY_EVALUATION = FieldDef(
    name="scalability_evaluation", kind="result_table",
    label_en="Scalability Evaluation", label_zh="可扩展性评测",
    prompt="Provide scalability metrics: scale, throughput, latency data.",
    required=False, columns=_COL_RESULT,
)

_F_ECOSYSTEM_INTEGRATION = FieldDef(
    name="ecosystem_integration", kind="text",
    label_en="Ecosystem Integration", label_zh="生态集成",
    prompt="Describe interoperability with other tools in the ecosystem. "
           "Return null if not applicable.",
    required=False,
)

_F_COMPARISONS_TO_ALTERNATIVES = FieldDef(
    name="comparisons_to_alternatives", kind="result_table",
    label_en="Comparisons to Alternatives", label_zh="竞品对比",
    prompt="Compare against alternative solutions/competitors.",
    required=False, columns=_COL_RESULT,
)

_F_THEORETICAL_FRAMEWORK = FieldDef(
    name="theoretical_framework", kind="text",
    label_en="Theoretical Framework", label_zh="理论框架",
    prompt="Describe the theoretical framework: mathematical tools and "
           "analytical paradigm used.",
    required=True,
)

_F_KEY_DEFINITIONS = FieldDef(
    name="key_definitions", kind="key_value_table",
    label_en="Key Definitions", label_zh="关键定义",
    prompt="List key definitions: term → precise mathematical definition.",
    required=True,
    columns=[
        ColumnDef(name="key", label_en="Term", label_zh="术语"),
        ColumnDef(name="value", label_en="Definition", label_zh="定义"),
    ],
)

_COL_THEOREM = [
    ColumnDef(name="name", label_en="Theorem", label_zh="定理"),
    ColumnDef(name="latex", label_en="Statement", label_zh="陈述"),
    ColumnDef(name="explanation", label_en="Proof Sketch", label_zh="证明概要"),
    ColumnDef(name="significance", label_en="Significance", label_zh="意义"),
]

_F_THEOREMS = FieldDef(
    name="theorems", kind="formula_table",
    label_en="Theorems & Lemmas", label_zh="定理与引理",
    prompt="List all major theorems and lemmas: name, statement, "
           "proof sketch, and significance.",
    required=True, columns=_COL_THEOREM,
)

_F_ASSUMPTIONS = FieldDef(
    name="assumptions", kind="list[str]",
    label_en="Assumptions", label_zh="假设条件",
    prompt="List all assumptions the analysis depends on.",
    required=True,
)

_F_THEORETICAL_RESULTS = FieldDef(
    name="theoretical_results", kind="result_table",
    label_en="Theoretical Results", label_zh="理论结果",
    prompt="Summarize key theoretical results: type (bound/guarantee/complexity), "
           "statement, and conditions.",
    required=False, columns=_COL_RESULT,
)

_F_PROOF_TECHNIQUE = FieldDef(
    name="proof_technique", kind="text",
    label_en="Proof Technique", label_zh="证明技术",
    prompt="Describe the core proof technique used. Return null if not applicable.",
    required=False,
)

_F_CONNECTIONS_TO_EMPIRICAL = FieldDef(
    name="connections_to_empirical", kind="text",
    label_en="Connections to Practice", label_zh="实践联系",
    prompt="Describe how the theoretical results connect to empirical practice. "
           "Return null if not discussed.",
    required=False,
)

_F_OPEN_PROBLEMS = FieldDef(
    name="open_problems", kind="list[str]",
    label_en="Open Problems", label_zh="开放问题",
    prompt="List open problems identified by the authors.",
    required=False,
)

_F_TAXONOMY_OVERVIEW = FieldDef(
    name="taxonomy_overview", kind="text",
    label_en="Taxonomy Overview", label_zh="分类体系",
    prompt="Describe the taxonomy/organizational framework: "
           "classification dimensions and logic.",
    required=True,
)

_F_TAXONOMY_CATEGORIES = FieldDef(
    name="taxonomy_categories", kind="structured_list",
    label_en="Taxonomy Categories", label_zh="分类条目",
    prompt="List each taxonomy category: name, description, "
           "representative papers, key technical characteristics.",
    required=True,
    columns=[
        ColumnDef(name="title", label_en="Category", label_zh="分类"),
        ColumnDef(name="description", label_en="Description & Key Papers",
                  label_zh="描述与代表论文"),
        ColumnDef(name="papers", label_en="Papers", label_zh="论文"),
    ],
)

_F_HISTORICAL_EVOLUTION = FieldDef(
    name="historical_evolution", kind="text",
    label_en="Historical Evolution", label_zh="历史演进",
    prompt="Describe the historical evolution of the field.",
    required=False,
)

_F_METHOD_COMPARISON = FieldDef(
    name="method_comparison", kind="result_table",
    label_en="Method Comparison", label_zh="方法对比",
    prompt="Compare methods across key dimensions (optional).",
    required=False, columns=_COL_RESULT,
)

_F_TRENDS_AND_INSIGHTS = FieldDef(
    name="trends_and_insights", kind="text",
    label_en="Trends & Insights", label_zh="趋势与洞察",
    prompt="Summarize observed trends and deep insights from the review.",
    required=True,
)

_F_OPEN_CHALLENGES = FieldDef(
    name="open_challenges", kind="list[str]",
    label_en="Open Challenges", label_zh="开放挑战",
    prompt="List the open challenges facing the field.",
    required=True,
)

_F_FUTURE_DIRECTIONS = FieldDef(
    name="future_directions", kind="list[str]",
    label_en="Future Directions", label_zh="未来方向",
    prompt="List predicted future research directions.",
    required=False,
)

_F_POSITION_STATEMENT = FieldDef(
    name="position_statement", kind="text",
    label_en="Position Statement", label_zh="立场陈述",
    prompt="What is the core position/argument this paper advances?",
    required=True,
)

_F_ARGUMENTS = FieldDef(
    name="arguments", kind="list[str]",
    label_en="Arguments", label_zh="论点",
    prompt="List the main arguments supporting the position.",
    required=True,
)

_F_SUPPORTING_EVIDENCE = FieldDef(
    name="supporting_evidence", kind="list[str]",
    label_en="Supporting Evidence", label_zh="支持证据",
    prompt="List evidence/data cited to support the arguments.",
    required=False,
)

_F_COUNTER_ARGUMENTS = FieldDef(
    name="counter_arguments", kind="list[str]",
    label_en="Counter-arguments", label_zh="反方论点",
    prompt="List counter-arguments the paper addresses or acknowledges.",
    required=False,
)

_F_IMPLICATIONS = FieldDef(
    name="implications", kind="text",
    label_en="Implications", label_zh="影响与意义",
    prompt="What are the broader implications of this position?",
    required=False,
)

_F_CALL_TO_ACTION = FieldDef(
    name="call_to_action", kind="text",
    label_en="Call to Action", label_zh="行动呼吁",
    prompt="What action does the paper call for? Return null if not explicit.",
    required=False,
)

# ══════════════════════════════════════════════════════════════════════
# Benchmark profile
# ══════════════════════════════════════════════════════════════════════

_BENCHMARK_FIELDS = [
    _F_PROBLEM, _F_MOTIVATION, _F_KEY_INSIGHT,
    _F_DATASET_DESCRIPTION, _F_COLLECTION_METHODOLOGY,
    _F_DATASET_STATISTICS, _F_TASK_DEFINITION,
    _F_EVALUATION_PROTOCOL, _F_MAIN_RESULTS,  # main_results = baseline_results
    _F_BASELINE_METHODS, _F_KNOWN_BIASES, _F_MAINTENANCE_PLAN,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_BENCHMARK_SECTIONS = [
    SectionDef(
        name="overview", level=1,
        title_en="Paper Overview", title_zh="论文概览",
        fields=["meta:title", "meta:authors", "meta:year",
                "meta:citation_count", "meta:url", "key_insight"],
    ),
    SectionDef(
        name="problem", level=1,
        title_en="Problem Definition & Motivation", title_zh="问题定义与动机",
        fields=["problem", "motivation"],
    ),
    SectionDef(
        name="dataset", level=1,
        title_en="Dataset", title_zh="数据集",
        subsections=[
            SectionDef(name="desc", level=2,
                       title_en="Dataset Description", title_zh="数据集描述",
                       fields=["dataset_description"]),
            SectionDef(name="collection", level=2,
                       title_en="Collection Methodology", title_zh="采集方法",
                       fields=["collection_methodology"]),
            SectionDef(name="stats", level=2,
                       title_en="Dataset Statistics", title_zh="数据集统计",
                       fields=["dataset_statistics"]),
            SectionDef(name="biases", level=2,
                       title_en="Known Biases", title_zh="已知偏差",
                       fields=["known_biases"]),
            SectionDef(name="maintenance", level=2,
                       title_en="Maintenance Plan", title_zh="维护计划",
                       fields=["maintenance_plan"]),
        ],
    ),
    SectionDef(
        name="task_eval", level=1,
        title_en="Task & Evaluation", title_zh="任务与评估",
        subsections=[
            SectionDef(name="task_def", level=2,
                       title_en="Task Definition", title_zh="任务定义",
                       fields=["task_definition"]),
            SectionDef(name="eval_proto", level=2,
                       title_en="Evaluation Protocol", title_zh="评估协议",
                       fields=["evaluation_protocol"]),
            SectionDef(name="baseline_methods", level=2,
                       title_en="Baseline Methods", title_zh="基线方法",
                       fields=["baseline_methods"]),
            SectionDef(name="baseline_results", level=2,
                       title_en="Baseline Results", title_zh="基线结果",
                       fields=["main_results"]),
        ],
    ),
    SectionDef(
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
    ),
    SectionDef(name="field_routes", level=1,
               title_en="Field Technical Landscape", title_zh="领域技术路线",
               condition="routes is not None"),
    SectionDef(name="comparison", level=1,
               title_en="Comparative Analysis", title_zh="对比分析",
               condition="comparison is not None"),
    SectionDef(name="references", level=1,
               title_en="Reference Classification", title_zh="参考文献分类",
               condition="references is not None"),
]

_BENCHMARK_PROFILE = PaperTypeProfile(
    type_name="benchmark",
    description="Introduces a new dataset, benchmark suite, or evaluation protocol.",
    fields=_BENCHMARK_FIELDS,
    sections=_BENCHMARK_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# System profile
# ══════════════════════════════════════════════════════════════════════

_SYSTEM_FIELDS = [
    _F_PROBLEM, _F_MOTIVATION, _F_KEY_INSIGHT,
    _F_SYSTEM_ARCHITECTURE, _F_DESIGN_DECISIONS, _F_API_DESIGN,
    _F_IMPLEMENTATION_STACK, _F_PERFORMANCE_ENGINEERING,
    _F_SCALABILITY_EVALUATION, _F_ECOSYSTEM_INTEGRATION,
    _F_COMPARISONS_TO_ALTERNATIVES, _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_SYSTEM_SECTIONS = [
    SectionDef(
        name="overview", level=1,
        title_en="Paper Overview", title_zh="论文概览",
        fields=["meta:title", "meta:authors", "meta:year",
                "meta:citation_count", "meta:url", "key_insight"],
    ),
    SectionDef(
        name="problem", level=1,
        title_en="Problem Definition & Motivation", title_zh="问题定义与动机",
        fields=["problem", "motivation"],
    ),
    SectionDef(
        name="architecture", level=1,
        title_en="System Architecture", title_zh="系统架构",
        subsections=[
            SectionDef(name="sys_arch", level=2,
                       title_en="Overall Architecture", title_zh="整体架构",
                       fields=["system_architecture"]),
            SectionDef(name="design", level=2,
                       title_en="Design Decisions", title_zh="设计决策",
                       fields=["design_decisions"]),
            SectionDef(name="api", level=2,
                       title_en="API Design", title_zh="接口设计",
                       fields=["api_design"]),
            SectionDef(name="stack", level=2,
                       title_en="Implementation Stack", title_zh="技术栈",
                       fields=["implementation_stack"]),
            SectionDef(name="perf", level=2,
                       title_en="Performance Engineering", title_zh="性能工程",
                       fields=["performance_engineering"]),
        ],
    ),
    SectionDef(
        name="evaluation", level=1,
        title_en="Evaluation", title_zh="评测",
        subsections=[
            SectionDef(name="scalability", level=2,
                       title_en="Scalability", title_zh="可扩展性",
                       fields=["scalability_evaluation"]),
            SectionDef(name="comparisons", level=2,
                       title_en="Competitive Comparison", title_zh="竞品对比",
                       fields=["comparisons_to_alternatives"]),
            SectionDef(name="ecosystem", level=2,
                       title_en="Ecosystem Integration", title_zh="生态集成",
                       fields=["ecosystem_integration"]),
        ],
    ),
    SectionDef(
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
    ),
    SectionDef(name="field_routes", level=1,
               title_en="Field Technical Landscape", title_zh="领域技术路线",
               condition="routes is not None"),
    SectionDef(name="comparison", level=1,
               title_en="Comparative Analysis", title_zh="对比分析",
               condition="comparison is not None"),
    SectionDef(name="references", level=1,
               title_en="Reference Classification", title_zh="参考文献分类",
               condition="references is not None"),
]

_SYSTEM_PROFILE = PaperTypeProfile(
    type_name="system",
    description="Describes a software system, framework, library, or "
                "distributed infrastructure.",
    fields=_SYSTEM_FIELDS,
    sections=_SYSTEM_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# Theoretical profile
# ══════════════════════════════════════════════════════════════════════

_THEORETICAL_FIELDS = [
    _F_PROBLEM, _F_MOTIVATION, _F_KEY_INSIGHT,
    _F_THEORETICAL_FRAMEWORK, _F_KEY_DEFINITIONS, _F_ASSUMPTIONS,
    _F_THEOREMS, _F_PROOF_TECHNIQUE, _F_THEORETICAL_RESULTS,
    _F_CONNECTIONS_TO_EMPIRICAL, _F_CONTRIBUTIONS, _F_LIMITATIONS,
    _F_OPEN_PROBLEMS,
]

_THEORETICAL_SECTIONS = [
    SectionDef(
        name="overview", level=1,
        title_en="Paper Overview", title_zh="论文概览",
        fields=["meta:title", "meta:authors", "meta:year",
                "meta:citation_count", "meta:url", "key_insight"],
    ),
    SectionDef(
        name="problem", level=1,
        title_en="Problem Definition & Motivation", title_zh="问题定义与动机",
        fields=["problem", "motivation"],
    ),
    SectionDef(
        name="framework", level=1,
        title_en="Theoretical Framework", title_zh="理论框架",
        subsections=[
            SectionDef(name="framework_overview", level=2,
                       title_en="Framework Overview", title_zh="框架概述",
                       fields=["theoretical_framework"]),
            SectionDef(name="definitions", level=2,
                       title_en="Key Definitions", title_zh="关键定义",
                       fields=["key_definitions"]),
        ],
    ),
    SectionDef(
        name="assumptions", level=1,
        title_en="Assumptions", title_zh="假设条件",
        fields=["assumptions"],
    ),
    SectionDef(
        name="theorems", level=1,
        title_en="Theorems & Proofs", title_zh="定理与证明",
        subsections=[
            SectionDef(name="theorem_list", level=2,
                       title_en="Main Theorems", title_zh="主要定理",
                       fields=["theorems"]),
            SectionDef(name="proof_technique", level=2,
                       title_en="Proof Technique", title_zh="证明技术",
                       fields=["proof_technique"]),
        ],
    ),
    SectionDef(
        name="theoretical_results", level=1,
        title_en="Theoretical Results", title_zh="理论结果",
        subsections=[
            SectionDef(name="results_table", level=2,
                       title_en="Results Summary", title_zh="结果总结",
                       fields=["theoretical_results"]),
            SectionDef(name="empirical", level=2,
                       title_en="Connections to Practice", title_zh="实践联系",
                       fields=["connections_to_empirical"]),
        ],
    ),
    SectionDef(
        name="contrib_limits", level=1,
        title_en="Contributions, Limitations & Open Problems",
        title_zh="贡献、局限与开放问题",
        subsections=[
            SectionDef(name="contributions", level=2,
                       title_en="Main Contributions", title_zh="主要贡献",
                       fields=["contributions"]),
            SectionDef(name="limitations", level=2,
                       title_en="Limitations", title_zh="局限性",
                       fields=["limitations"]),
            SectionDef(name="open_problems", level=2,
                       title_en="Open Problems", title_zh="开放问题",
                       fields=["open_problems"]),
        ],
    ),
    SectionDef(name="field_routes", level=1,
               title_en="Field Technical Landscape", title_zh="领域技术路线",
               condition="routes is not None"),
    SectionDef(name="comparison", level=1,
               title_en="Comparative Analysis", title_zh="对比分析",
               condition="comparison is not None"),
    SectionDef(name="references", level=1,
               title_en="Reference Classification", title_zh="参考文献分类",
               condition="references is not None"),
]

_THEORETICAL_PROFILE = PaperTypeProfile(
    type_name="theoretical",
    description="Core contribution is a theorem, bound, or theoretical framework "
                "with mathematical proofs.",
    fields=_THEORETICAL_FIELDS,
    sections=_THEORETICAL_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# Survey profile
# ══════════════════════════════════════════════════════════════════════

_SURVEY_FIELDS = [
    _F_PROBLEM, _F_SCOPE, _F_KEY_INSIGHT,
    _F_TAXONOMY_OVERVIEW, _F_TAXONOMY_CATEGORIES,
    _F_HISTORICAL_EVOLUTION, _F_METHOD_COMPARISON,
    _F_TRENDS_AND_INSIGHTS, _F_OPEN_CHALLENGES, _F_FUTURE_DIRECTIONS,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_SURVEY_SECTIONS = [
    SectionDef(
        name="overview", level=1,
        title_en="Paper Overview", title_zh="论文概览",
        fields=["meta:title", "meta:authors", "meta:year",
                "meta:citation_count", "meta:url", "key_insight"],
    ),
    SectionDef(
        name="scope", level=1,
        title_en="Scope & Problem", title_zh="范围与问题",
        fields=["problem", "scope"],
    ),
    SectionDef(
        name="taxonomy", level=1,
        title_en="Taxonomy", title_zh="分类体系",
        subsections=[
            SectionDef(name="tax_overview", level=2,
                       title_en="Taxonomy Overview", title_zh="分类概述",
                       fields=["taxonomy_overview"]),
            SectionDef(name="tax_categories", level=2,
                       title_en="Categories", title_zh="分类条目",
                       fields=["taxonomy_categories"]),
        ],
    ),
    SectionDef(
        name="evolution", level=1,
        title_en="Historical Evolution", title_zh="历史演进",
        fields=["historical_evolution"],
    ),
    SectionDef(
        name="method_comp", level=1,
        title_en="Method Comparison", title_zh="方法对比",
        fields=["method_comparison"],
    ),
    SectionDef(
        name="trends", level=1,
        title_en="Trends & Insights", title_zh="趋势与洞察",
        subsections=[
            SectionDef(name="trends_insights", level=2,
                       title_en="Trends and Insights", title_zh="趋势与洞察",
                       fields=["trends_and_insights"]),
            SectionDef(name="challenges", level=2,
                       title_en="Open Challenges", title_zh="开放挑战",
                       fields=["open_challenges"]),
            SectionDef(name="future", level=2,
                       title_en="Future Directions", title_zh="未来方向",
                       fields=["future_directions"]),
        ],
    ),
    SectionDef(
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
    ),
    SectionDef(name="field_routes", level=1,
               title_en="Field Technical Landscape", title_zh="领域技术路线",
               condition="routes is not None"),
    SectionDef(name="comparison", level=1,
               title_en="Comparative Analysis", title_zh="对比分析",
               condition="comparison is not None"),
    SectionDef(name="references", level=1,
               title_en="Reference Classification", title_zh="参考文献分类",
               condition="references is not None"),
]

_SURVEY_PROFILE = PaperTypeProfile(
    type_name="survey",
    description="Systematic literature review, taxonomy, or trend analysis.",
    fields=_SURVEY_FIELDS,
    sections=_SURVEY_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# Position profile
# ══════════════════════════════════════════════════════════════════════

_POSITION_FIELDS = [
    _F_PROBLEM, _F_POSITION_STATEMENT, _F_MOTIVATION, _F_KEY_INSIGHT,
    _F_ARGUMENTS, _F_SUPPORTING_EVIDENCE, _F_COUNTER_ARGUMENTS,
    _F_IMPLICATIONS, _F_CALL_TO_ACTION,
    _F_CONTRIBUTIONS, _F_LIMITATIONS,
]

_POSITION_SECTIONS = [
    SectionDef(
        name="overview", level=1,
        title_en="Paper Overview", title_zh="论文概览",
        fields=["meta:title", "meta:authors", "meta:year",
                "meta:citation_count", "meta:url", "key_insight"],
    ),
    SectionDef(
        name="position", level=1,
        title_en="Position Statement", title_zh="立场陈述",
        fields=["position_statement"],
    ),
    SectionDef(
        name="problem", level=1,
        title_en="Problem & Motivation", title_zh="问题与动机",
        fields=["problem", "motivation"],
    ),
    SectionDef(
        name="arguments", level=1,
        title_en="Arguments & Evidence", title_zh="论点与证据",
        subsections=[
            SectionDef(name="main_args", level=2,
                       title_en="Arguments", title_zh="论点",
                       fields=["arguments"]),
            SectionDef(name="evidence", level=2,
                       title_en="Supporting Evidence", title_zh="支持证据",
                       fields=["supporting_evidence"]),
            SectionDef(name="counter_args", level=2,
                       title_en="Counter-arguments", title_zh="反方论点",
                       fields=["counter_arguments"]),
        ],
    ),
    SectionDef(
        name="impact", level=1,
        title_en="Implications & Call to Action", title_zh="影响与行动呼吁",
        subsections=[
            SectionDef(name="implications", level=2,
                       title_en="Implications", title_zh="影响与意义",
                       fields=["implications"]),
            SectionDef(name="call_to_action", level=2,
                       title_en="Call to Action", title_zh="行动呼吁",
                       fields=["call_to_action"]),
        ],
    ),
    SectionDef(
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
    ),
    SectionDef(name="field_routes", level=1,
               title_en="Field Technical Landscape", title_zh="领域技术路线",
               condition="routes is not None"),
    SectionDef(name="comparison", level=1,
               title_en="Comparative Analysis", title_zh="对比分析",
               condition="comparison is not None"),
    SectionDef(name="references", level=1,
               title_en="Reference Classification", title_zh="参考文献分类",
               condition="references is not None"),
]

_POSITION_PROFILE = PaperTypeProfile(
    type_name="position",
    description="Position paper, vision statement, or call to action.",
    fields=_POSITION_FIELDS,
    sections=_POSITION_SECTIONS,
)

# ══════════════════════════════════════════════════════════════════════
# Domain
# ══════════════════════════════════════════════════════════════════════

AI_ML_DOMAIN = DomainProfile(
    domain_name="ai_ml",
    domain_description="Artificial intelligence, machine learning, computer "
                       "vision, natural language processing, and related areas.",
    paper_types=[
        EXPERIMENTAL_PROFILE,
        _BENCHMARK_PROFILE,
        _SYSTEM_PROFILE,
        _THEORETICAL_PROFILE,
        _SURVEY_PROFILE,
        _POSITION_PROFILE,
    ],
    default_paper_type="experimental",
)
