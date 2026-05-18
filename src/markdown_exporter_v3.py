"""Markdown export for V3 structured analysis reports.

Generates a rich Markdown report from the seed paper's structured understanding,
technical routes, comparative analysis, and reference classification.
Supports bilingual output (EN / ZH).
"""
from __future__ import annotations

from pathlib import Path

from paper import Paper, StructuredUnderstanding, Reference, CitationType

# ── Section header translations ──
_SECTION = {
    "en": {
        "title": "Structured Understanding of",
        "overview": "Paper Overview",
        "problem": "Problem Definition & Motivation",
        "method": "Method Architecture",
        "arch_overview": "Overall Architecture",
        "arch_figure": "Architecture Diagram Explanation",
        "components": "Core Components",
        "formulas": "Key Formulas",
        "training": "Training Pipeline",
        "inference": "Inference Pipeline",
        "results": "Experimental Results",
        "main_results": "Main Results",
        "ablation": "Ablation Studies",
        "qualitative": "Qualitative Analysis",
        "contrib_limits": "Contributions & Limitations",
        "contributions": "Main Contributions",
        "limitations": "Limitations",
        "field_routes": "Field Technical Landscape",
        "route_overview": "Technical Route Overview",
        "branch": "Branch",
        "key_papers_in_branch": "Key Papers",
        "common_tech": "Common Technical Features",
        "comparison": "Comparative Analysis",
        "comp_matrix": "Design Comparison",
        "comp_narrative": "Comparison Summary",
        "unique_pos": "Unique Positioning",
        "references": "Reference Classification",
        "ref_type": "Type",
        "importance": "Importance",
        "note": "Note",
        "no_data": "(not available)",
        "component_col": "Component",
        "purpose_col": "Purpose",
        "details_col": "Implementation Details",
        "figure_col": "Figure",
        "formula_col": "Formula",
        "explanation_col": "Explanation",
        "significance_col": "Significance",
        "dataset_col": "Dataset",
        "metric_col": "Metric",
        "value_col": "Value",
        "comparison_col": "vs Baseline",
        "dimension": "Dimension",
        "seed_approach": "Seed Paper",
        "mainstream_approach": "Mainstream Approach",
        "advantage": "Advantage / Difference",
    },
    "zh": {
        "title": "结构化解读",
        "overview": "论文概览",
        "problem": "问题定义与动机",
        "method": "方法架构",
        "arch_overview": "整体架构",
        "arch_figure": "架构图详解",
        "components": "核心组件",
        "formulas": "关键公式",
        "training": "训练流程",
        "inference": "推理流程",
        "results": "实验结果",
        "main_results": "主要结果",
        "ablation": "消融实验",
        "qualitative": "定性分析",
        "contrib_limits": "贡献与局限性",
        "contributions": "主要贡献",
        "limitations": "局限性",
        "field_routes": "领域技术路线",
        "route_overview": "技术路线全景",
        "branch": "分支",
        "key_papers_in_branch": "关键论文",
        "common_tech": "共同技术特征",
        "comparison": "对比分析",
        "comp_matrix": "设计对比",
        "comp_narrative": "对比总结",
        "unique_pos": "独特定位",
        "references": "参考文献分类",
        "ref_type": "类型",
        "importance": "重要性",
        "note": "说明",
        "no_data": "（无数据）",
        "component_col": "组件",
        "purpose_col": "功能",
        "details_col": "实现细节",
        "figure_col": "对应图表",
        "formula_col": "公式",
        "explanation_col": "含义",
        "significance_col": "重要性",
        "dataset_col": "数据集",
        "metric_col": "指标",
        "value_col": "数值",
        "comparison_col": "对比Baseline",
        "dimension": "维度",
        "seed_approach": "种子论文",
        "mainstream_approach": "主流路线",
        "advantage": "优势/差异",
    },
}

_CITATION_TYPE_LABEL = {
    "en": {
        "supporting": "Supporting",
        "contrasting": "Contrasting",
        "foundational": "Foundational",
        "related_work": "Related Work",
        "not_classified": "—",
    },
    "zh": {
        "supporting": "赞同",
        "contrasting": "对比",
        "foundational": "奠基",
        "related_work": "相关",
        "not_classified": "—",
    },
}


def _t(key: str, lang: str) -> str:
    return _SECTION.get(lang, _SECTION["en"]).get(key, key)


def _ct(citation_type, lang: str) -> str:
    val = citation_type.value if hasattr(citation_type, "value") else str(citation_type)
    return _CITATION_TYPE_LABEL.get(lang, _CITATION_TYPE_LABEL["en"]).get(val, val)


def _importance_stars(is_key: bool) -> str:
    return "★★★" if is_key else "★★"


def _escape_md(text: str) -> str:
    """Escape pipe characters in markdown table cells."""
    return (text or "").replace("|", "\\|").replace("\n", " ")


def export_markdown(
    seed_paper: Paper,
    routes: dict | None,
    comparison: dict | None,
    references: list[Reference] | None,
    output_path: str | Path,
    lang: str = "en",
) -> None:
    """Generate the full structured analysis markdown report."""
    lines = []
    s = seed_paper.structured

    # ── Title ──
    title = seed_paper.title or "Unknown Paper"
    lines.append(f"# {_t('title', lang)}: {title}")
    lines.append("")

    # ── Paper Overview ──
    lines.append(f"## 1. {_t('overview', lang)}")
    lines.append("")
    lines.append(f"- **Title**: {title}")
    authors_str = ", ".join(seed_paper.authors) if seed_paper.authors else _t("no_data", lang)
    lines.append(f"- **Authors**: {authors_str}")
    lines.append(f"- **Year**: {seed_paper.year or _t('no_data', lang)}")
    lines.append(f"- **Citations**: {seed_paper.citation_count}")
    lines.append(f"- **URL**: {seed_paper.url or _t('no_data', lang)}")
    lines.append("")

    if s and s.key_insight:
        lines.append(f"**{s.key_insight}**")
        lines.append("")

    # ── Problem & Motivation ──
    lines.append(f"## 2. {_t('problem', lang)}")
    lines.append("")
    if s:
        if s.problem:
            lines.append(f"**Problem**: {s.problem}")
            lines.append("")
        if s.motivation:
            lines.append(f"**Motivation**: {s.motivation}")
            lines.append("")

    # ── Method Architecture ──
    lines.append(f"## 3. {_t('method', lang)}")
    lines.append("")

    if s:
        # 3.1 Overall architecture
        lines.append(f"### 3.1 {_t('arch_overview', lang)}")
        lines.append("")
        lines.append(s.architecture_overview or _t("no_data", lang))
        lines.append("")

        # 3.2 Architecture figure
        if s.architecture_figure:
            lines.append(f"### 3.2 {_t('arch_figure', lang)}")
            lines.append("")
            lines.append(s.architecture_figure)
            lines.append("")

        # 3.3 Components
        if s.components:
            lines.append(f"### 3.3 {_t('components', lang)}")
            lines.append("")
            lines.append(
                f"| {_t('component_col', lang)} | {_t('purpose_col', lang)} "
                f"| {_t('details_col', lang)} | {_t('figure_col', lang)} |"
            )
            lines.append("|---|---|---|---|")
            for c in s.components:
                name = _escape_md(c.name)
                purpose = _escape_md(c.purpose)
                details = _escape_md(c.details or _t("no_data", lang))
                fig = c.referenced_figure or "—"
                lines.append(f"| {name} | {purpose} | {details} | {fig} |")
            lines.append("")

        # 3.4 Formulas
        if s.formulas:
            lines.append(f"### 3.4 {_t('formulas', lang)}")
            lines.append("")
            lines.append(
                f"| {_t('formula_col', lang)} | {_t('explanation_col', lang)} "
                f"| {_t('significance_col', lang)} |"
            )
            lines.append("|---|---|---|")
            for f in s.formulas:
                name = _escape_md(f.name)
                expl = _escape_md(f.explanation)
                sig = _escape_md(f.significance)
                if f.latex:
                    name = f"${f.latex}$ — {name}"
                lines.append(f"| {name} | {expl} | {sig} |")
            lines.append("")

        # 3.5 Training
        lines.append(f"### 3.5 {_t('training', lang)}")
        lines.append("")
        if s.training_data:
            lines.append(f"- **Data**: {s.training_data}")
        if s.loss_functions:
            losses = ", ".join(f"`{l}`" for l in s.loss_functions)
            lines.append(f"- **Loss Functions**: {losses}")
        if s.optimizer:
            lines.append(f"- **Optimizer**: {s.optimizer}")
        if s.training_procedure:
            lines.append(f"- **Procedure**: {s.training_procedure}")
        lines.append("")

        # 3.6 Inference
        lines.append(f"### 3.6 {_t('inference', lang)}")
        lines.append("")
        if s.inference_procedure:
            lines.append(s.inference_procedure)
        if s.post_processing:
            lines.append(f"- **Post-processing**: {s.post_processing}")
        lines.append("")

    # ── Experimental Results ──
    lines.append(f"## 4. {_t('results', lang)}")
    lines.append("")

    if s:
        if s.main_results:
            lines.append(f"### 4.1 {_t('main_results', lang)}")
            lines.append("")
            lines.append(
                f"| {_t('dataset_col', lang)} | {_t('metric_col', lang)} "
                f"| {_t('value_col', lang)} | {_t('comparison_col', lang)} |"
            )
            lines.append("|---|---|---|---|")
            for r in s.main_results:
                dataset = _escape_md(r.dataset)
                metric = _escape_md(r.metric)
                value = _escape_md(r.value)
                comp = _escape_md(r.comparison or "—")
                lines.append(f"| {dataset} | {metric} | {value} | {comp} |")
            lines.append("")

        if s.ablation_results:
            lines.append(f"### 4.2 {_t('ablation', lang)}")
            lines.append("")
            for a in s.ablation_results:
                lines.append(f"- {a}")
            lines.append("")

        if s.qualitative_results:
            lines.append(f"### 4.3 {_t('qualitative', lang)}")
            lines.append("")
            lines.append(s.qualitative_results)
            lines.append("")

    # ── Contributions & Limitations ──
    lines.append(f"## 5. {_t('contrib_limits', lang)}")
    lines.append("")

    if s:
        if s.contributions:
            lines.append(f"### 5.1 {_t('contributions', lang)}")
            lines.append("")
            for c in s.contributions:
                lines.append(f"- {c}")
            lines.append("")

        if s.limitations:
            lines.append(f"### 5.2 {_t('limitations', lang)}")
            lines.append("")
            for lim in s.limitations:
                lines.append(f"- {lim}")
            lines.append("")

    # ── Field Technical Landscape ──
    lines.append("---")
    lines.append("")
    lines.append(f"## 6. {_t('field_routes', lang)}")
    lines.append("")

    if routes:
        overview = routes.get("overview", "")
        if overview:
            lines.append(f"### 6.1 {_t('route_overview', lang)}")
            lines.append("")
            lines.append(overview)
            lines.append("")

        for i, branch in enumerate(routes.get("branches", []), 1):
            name = branch.get("name", f"Branch {i}")
            is_ms = " ★" if branch.get("is_mainstream") else ""
            lines.append(f"### 6.{i + 1} {name}{is_ms}")
            lines.append("")
            if branch.get("description"):
                lines.append(branch["description"])
                lines.append("")

            tags = branch.get("common_technical_tags", [])
            if tags:
                lines.append(f"**{_t('common_tech', lang)}**: {', '.join(tags)}")
                lines.append("")

    # ── Comparative Analysis ──
    if comparison:
        lines.append("---")
        lines.append("")
        lines.append(f"## 7. {_t('comparison', lang)}")
        lines.append("")

        matrix = comparison.get("comparison_matrix", [])
        if matrix:
            lines.append(f"### 7.1 {_t('comp_matrix', lang)}")
            lines.append("")
            lines.append(
                f"| {_t('dimension', lang)} | {_t('seed_approach', lang)} "
                f"| {_t('mainstream_approach', lang)} | {_t('advantage', lang)} |"
            )
            lines.append("|---|---|---|---|")
            for row in matrix:
                dim = _escape_md(row.get("dimension", ""))
                seed_val = _escape_md(row.get("seed_paper", ""))
                main_val = _escape_md(row.get("mainstream_approach", ""))
                adv = _escape_md(row.get("advantage", ""))
                lines.append(f"| {dim} | {seed_val} | {main_val} | {adv} |")
            lines.append("")

        narrative = comparison.get("narrative", "")
        if narrative:
            lines.append(f"### 7.2 {_t('comp_narrative', lang)}")
            lines.append("")
            lines.append(narrative)
            lines.append("")

        positioning = comparison.get("unique_positioning", "")
        if positioning:
            lines.append(f"### 7.3 {_t('unique_pos', lang)}")
            lines.append("")
            lines.append(positioning)
            lines.append("")

    # ── Reference Classification ──
    if references:
        lines.append("---")
        lines.append("")
        lines.append(f"## 8. {_t('references', lang)}")
        lines.append("")
        lines.append(
            f"| # | Paper | {_t('ref_type', lang)} "
            f"| {_t('importance', lang)} | {_t('note', lang)} |"
        )
        lines.append("|---|---|---|---|---|")
        for i, ref in enumerate(references, 1):
            title = _escape_md(ref.paper_title or ref.paper_id or "?")
            ct_label = _ct(ref.citation_type, lang)
            stars = _importance_stars(ref.is_key_reference)
            note = _escape_md((ref.context or "")[:100])
            lines.append(f"| {i} | {title} | {ct_label} | {stars} | {note} |")
        lines.append("")

    # ── Write file ──
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
