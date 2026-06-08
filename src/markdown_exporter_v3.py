"""Markdown export for V3 structured analysis reports.

Generates a rich Markdown report from the seed paper's structured understanding,
technical routes, comparative analysis, and reference classification.
Supports bilingual output (EN / ZH) via post-hoc LLM translation.

Rendering is driven by a ``PaperTypeProfile`` schema when provided (see
``src/domains/``), or falls back to the hardcoded experimental layout.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

from openai import OpenAI

from paper import Paper, StructuredUnderstanding, Reference, CitationType
from domains.base import PaperTypeProfile, FieldDef, SectionDef

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
        "design_rationale": "Design Rationale & Intuition",
        "training": "Training Pipeline",
        "inference": "Inference Pipeline",
        "results": "Experimental Results",
        "main_results": "Main Results",
        "ablation": "Ablation Studies",
        "qualitative": "Qualitative Analysis",
        "contrib_limits": "Contributions & Limitations",
        "contributions": "Main Contributions",
        "limitations": "Limitations",
        "synthesis": "Synthesis & Significance",
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
        "empty_section": "(Not discussed in the paper)",
        "meta_title": "Title",
        "meta_authors": "Authors",
        "meta_year": "Year",
        "meta_citation_count": "Citations",
        "meta_url": "URL",
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
        "design_rationale": "设计原理与直观理解",
        "training": "训练流程",
        "inference": "推理流程",
        "results": "实验结果",
        "main_results": "主要结果",
        "ablation": "消融实验",
        "qualitative": "定性分析",
        "contrib_limits": "贡献与局限性",
        "contributions": "主要贡献",
        "limitations": "局限性",
        "synthesis": "总结与意义",
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
        "empty_section": "（论文未讨论）",
        "meta_title": "标题",
        "meta_authors": "作者",
        "meta_year": "年份",
        "meta_citation_count": "引用次数",
        "meta_url": "URL",
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
    *,
    profile: Optional[PaperTypeProfile] = None,
) -> None:
    """Generate the full structured analysis markdown report.

    When *profile* is given, section structure and rendering are driven by
    the schema.  Otherwise the hardcoded experimental layout is used.
    """
    if profile is not None:
        return _export_schema_driven(
            seed_paper, routes, comparison, references,
            output_path, lang, profile,
        )

    # ── Hardcoded path (backward compat) ──
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

    # ── Field Technical Landscape [仅完整分析模式] ──
    if routes:
        lines.append("---")
        lines.append("")
        lines.append(f"## 6. {_t('field_routes', lang)}")
        lines.append("")

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

    # ── Comparative Analysis [仅完整分析模式] ──
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


# ──────────────────────────────────────────────────────────────────────
# Schema-driven export
# ──────────────────────────────────────────────────────────────────────

def _is_empty(value) -> bool:
    """Check whether a field value is semantically empty."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _render_field_value(value, field: FieldDef, lang: str) -> list[str]:
    """Render a single field value to one or more markdown lines."""
    lines: list[str] = []

    if field.kind == "text":
        text = str(value).strip()
        if text and field.name in ("intuitive_analogy", "core_question"):
            lines.append(f"> {text}")
        else:
            import re
            # (1) Split mid-line (N) items with double newline for
            # markdown paragraph breaks
            text = re.sub(r'(?<!\n)(?<!\A)\((\d+)\)\s', r'\n\n(\1) ', text)
            # (2) Ensure blank lines between consecutive (N) items that
            # ended up on adjacent lines with only a single \n between them
            text = re.sub(r'(\(\d+\)\s+[^\n]+)\n(\(\d+\)\s+)', r'\1\n\n\2', text)
            # (3) Add paragraph break before bold sub-headers after ". ",
            # but NOT after digit-period (preserve "1. **Title**" intact)
            text = re.sub(r'(?<=[a-zA-Z一-鿿])\. (\*\*[^*]{2,80}\*\*)',
                         r'\n\n\1', text)
            # Clean up excessive blank lines
            text = re.sub(r'\n{4,}', '\n\n\n', text)
            lines.append(text)
    elif field.kind == "list[str]":
        if isinstance(value, list):
            for item in value:
                lines.append(f"- {item}")
    elif field.kind == "formula_table":
        if not isinstance(value, list) or not value:
            return lines
        # 3-column format matching legacy: Formula ($latex$ — name) | Explanation | Significance
        headers = [
            _t("formula_col", lang),
            _t("explanation_col", lang),
            _t("significance_col", lang),
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|---|---|---|")
        for row in value:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", ""))
            latex = str(row.get("latex", ""))
            formula_cell = f"${latex}$ — {name}" if latex else name
            expl = _escape_md(str(row.get("explanation", "")))
            sig = _escape_md(str(row.get("significance", "")))
            lines.append(f"| {formula_cell} | {expl} | {sig} |")
    elif field.kind in ("component_table", "result_table", "key_value_table"):
        if not isinstance(value, list) or not value:
            return lines
        cols = field.columns
        if cols:
            headers = [c.label_en if lang == "en" else c.label_zh for c in cols]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            for row in value:
                if not isinstance(row, dict):
                    continue
                cells = [_escape_md(str(row.get(c.name, ""))) for c in cols]
                lines.append("| " + " | ".join(cells) + " |")
    elif field.kind == "structured_list":
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                title = item.get("title", "")
                desc = item.get("description", "")
                papers = item.get("papers", [])
                lines.append(f"- **{title}**: {desc}")
                if papers:
                    lines.append(f"  — {', '.join(papers)}")

    return lines


def _render_meta_field(field_name: str, paper: Paper, lang: str) -> str:
    """Render a meta: field from the Paper object's metadata with label."""
    meta_key = field_name[len("meta:"):]
    label = _t(f"meta_{meta_key}", lang)
    if meta_key == "title":
        val = paper.title or _t("no_data", lang)
    elif meta_key == "authors":
        val = ", ".join(paper.authors) if paper.authors else _t("no_data", lang)
    elif meta_key == "year":
        val = str(paper.year) if paper.year else _t("no_data", lang)
    elif meta_key == "citation_count":
        val = str(paper.citation_count)
    elif meta_key == "url":
        val = paper.url or _t("no_data", lang)
    else:
        return ""
    return f"- **{label}**: {val}"


def _eval_condition(condition: str, routes, comparison, references) -> bool:
    """Evaluate a simple SectionDef condition expression."""
    if condition is None:
        return True
    # Whitelist-based evaluation: only allow specific expressions
    allowed = {
        "routes": routes,
        "comparison": comparison,
        "references": references,
    }
    try:
        return bool(eval(condition, {"__builtins__": {}}, allowed))
    except Exception:
        return True  # if eval fails, show the section


def _render_schema_section(
    section: SectionDef,
    paper: Paper,
    structured: dict,
    routes,
    comparison,
    references,
    lang: str,
    profile: PaperTypeProfile,
    counters: list[int],
) -> list[str]:
    """Render one section (and its subsections) to markdown lines."""
    lines: list[str] = []

    # Check section-level condition
    if section.condition and not _eval_condition(
        section.condition, routes, comparison, references
    ):
        return lines

    # ── Section numbering ──
    level = section.level
    while len(counters) <= level:
        counters.append(0)
    counters[level] += 1
    # Reset deeper counters
    for i in range(level + 1, len(counters)):
        counters[i] = 0

    num_parts = [str(counters[i]) for i in range(1, level + 1)]
    section_num = ".".join(num_parts)

    # Collect rendered field blocks
    rendered_blocks: list[list[str]] = []

    # ── Handle special sections by name ──
    if section.name == "field_routes" and routes:
        rendered_blocks.append(_render_routes_section(routes, lang, counters[1]))
    elif section.name == "comparison" and comparison:
        rendered_blocks.append(_render_comparison_section(comparison, lang, counters[1]))
    elif section.name == "references" and references:
        rendered_blocks.append(_render_references_section(references, lang))
    elif section.fields:
        # ── Regular section with structured fields ──
        for field_name in section.fields:
            if field_name.startswith("meta:"):
                val = _render_meta_field(field_name, paper, lang)
                if val:
                    rendered_blocks.append([val])
            else:
                val = structured.get(field_name) if structured else None
                if _is_empty(val):
                    continue
                # Look up the FieldDef for rendering
                fd = profile.get_field(field_name)
                if fd:
                    rendered = _render_field_value(val, fd, lang)
                    # Bold key_insight text
                    if field_name == "key_insight" and rendered:
                        rendered = [f"**{rendered[0]}**"]
                    if rendered:
                        rendered_blocks.append(rendered)

    # ── Render subsections ──
    for sub in section.subsections:
        sub_lines = _render_schema_section(
            sub, paper, structured, routes, comparison, references, lang,
            profile, counters,
        )
        if sub_lines:
            rendered_blocks.append(sub_lines)

    # ── Handle empty sections ──
    if not rendered_blocks:
        if section.always_show:
            pass  # force render even when empty
        elif section.level >= 2 and section.fields:
            # For subsections with defined fields, show placeholder to preserve numbering
            placeholder = _t("empty_section", lang)
            rendered_blocks.append([placeholder])
        else:
            return lines  # skip top-level empty sections entirely

    # ── Emit heading ──
    heading = "#" * (level + 1)
    title = section.title_en if lang == "en" else section.title_zh
    num_display = f"{section_num}." if level == 1 else section_num
    lines.append(f"{heading} {num_display} {title}")
    lines.append("")

    for block in rendered_blocks:
        lines.extend(block)
        lines.append("")

    return lines


def _render_routes_section(routes: dict, lang: str, parent_num: int) -> list[str]:
    """Render the field technical landscape section."""
    lines: list[str] = []
    overview = routes.get("overview", "")
    if overview:
        lines.append(f"### {parent_num}.1 {_t('route_overview', lang)}")
        lines.append("")
        lines.append(overview)
        lines.append("")

    for i, branch in enumerate(routes.get("branches", []), 1):
        name = branch.get("name", f"Branch {i}")
        is_ms = " ★" if branch.get("is_mainstream") else ""
        lines.append(f"### {parent_num}.{i + 1} {name}{is_ms}")
        lines.append("")
        if branch.get("description"):
            lines.append(branch["description"])
            lines.append("")
        tags = branch.get("common_technical_tags", [])
        if tags:
            lines.append(f"**{_t('common_tech', lang)}**: {', '.join(tags)}")
            lines.append("")
    return lines


def _render_comparison_section(comparison: dict, lang: str, parent_num: int) -> list[str]:
    """Render the comparative analysis section."""
    lines: list[str] = []
    matrix = comparison.get("comparison_matrix", [])
    if matrix:
        lines.append(f"### {parent_num}.1 {_t('comp_matrix', lang)}")
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
        lines.append(f"### {parent_num}.2 {_t('comp_narrative', lang)}")
        lines.append("")
        lines.append(narrative)
        lines.append("")

    positioning = comparison.get("unique_positioning", "")
    if positioning:
        lines.append(f"### {parent_num}.3 {_t('unique_pos', lang)}")
        lines.append("")
        lines.append(positioning)
        lines.append("")
    return lines


def _render_references_section(references: list, lang: str) -> list[str]:
    """Render the reference classification section."""
    lines: list[str] = []
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
    return lines


def _export_schema_driven(
    seed_paper: Paper,
    routes: dict | None,
    comparison: dict | None,
    references: list[Reference] | None,
    output_path: str | Path,
    lang: str,
    profile: PaperTypeProfile,
) -> None:
    """Schema-driven markdown export — rendering order from PaperTypeProfile."""
    lines: list[str] = []

    # ── Title ──
    title = seed_paper.title or "Unknown Paper"
    lines.append(f"# {_t('title', lang)}: {title}")
    lines.append("")

    # ── Convert structured to dict ──
    s = seed_paper.structured
    structured: dict = {}
    if s is not None:
        if isinstance(s, StructuredUnderstanding):
            structured = s.to_dict()
        elif isinstance(s, dict):
            structured = s

    # ── Render sections ──
    counters: list[int] = [0]  # index by level
    for section in profile.sections:
        section_lines = _render_schema_section(
            section, seed_paper, structured, routes, comparison, references, lang,
            profile, counters,
        )
        if section_lines:
            lines.extend(section_lines)
            lines.append("---")
            lines.append("")

    # ── Write file ──
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


# ── Post-hoc LLM translation ──

def translate_markdown_to_zh(
    markdown_text: str,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> str:
    """Translate an English V3 markdown report to Simplified Chinese via LLM.

    Preserves markdown structure, tables, LaTeX formulas, code blocks, paper
    titles, author names, and technical terms.  Returns the original text
    unchanged on any failure (graceful degradation).
    """
    if client is None:
        return markdown_text

    from llm_analyzer import _resolve_model
    model_name = _resolve_model(model)
    if not model_name:
        return markdown_text

    # Truncate if extremely long (> 100K chars) to avoid token limits
    text = markdown_text
    if len(text) > 100000:
        text = text[:100000] + "\n\n[Content truncated for translation]"

    prompt = textwrap.dedent(f"""\
        Translate the following academic research analysis report from English
        to Simplified Chinese.

        CRITICAL RULES:
        1. Preserve ALL markdown formatting — headers (# ## ###), tables (|...|),
           bold (**...**), inline code (`...`), list markers (-), separators (---)
        2. Keep ALL LaTeX formulas EXACTLY as-is ($...$ or $$...$$)
        3. Keep ALL paper titles in their original English form
        4. Keep ALL author names in their original English form
        5. Keep ALL of the following technical terms in English (DO NOT translate):
           - RL/ML terms: actor, critic, rollout, replay buffer, SAC, policy,
             Q-value, reward, episode, checkpoint, embedding, token, BEV,
             autoencoder, encoder, decoder, MLP, ViT, LSS, feature, batch
           - Method names: CIL-SERL, 3DGS, NMPC, HIL-SERL, REAP-SAC, ParkingHIL,
             E2EParking, ParkingE2E, Lift-Splat-Shoot
           - Dataset & metric names: NAVSIM, Bench2Drive, PDMS, EPDMS, mAP, NDS,
             PSR, PCR, PTR, PBR, NGS
           - Other: IoU, NUC, ROS, IMU, LiDAR, RGB, fps, Hz, GPU, CPU
        6. Keep ALL URLs, DOIs, and links unchanged
        7. Keep ALL model names and hardware names in English (e.g., "XGRIDS Lixel
           K1", "NVIDIA RTX 4090", "Chang'an CS55", "3DRealCar")
        8. Translate all OTHER text naturally to Simplified Chinese:
           - Section content / descriptions / explanations
           - Table cell content (except technical terms as above)
           - Narrative paragraphs, analysis, comparisons
        9. Do NOT add any extra commentary or markdown fences.
           Output ONLY the translated markdown.

        MARKDOWN TO TRANSLATE:

        {text}""")

    try:
        from config import LLM_ANALYZER_TIMEOUT_SEC
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=16000,
            timeout=LLM_ANALYZER_TIMEOUT_SEC * 3,
        )
        raw = response.choices[0].message.content or ""
        if raw.strip():
            return raw.strip() + "\n"
    except Exception as exc:
        print(f"Markdown translation failed: {exc}")

    return markdown_text
