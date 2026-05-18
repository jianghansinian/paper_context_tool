from pathlib import Path

# ── Section header translations ──

_SECTION = {
    "en": {
        "title": "Field",
        "overview": "Overview",
        "cross_branch_relationships": "Cross-Branch Relationships",
        "evolutionary_timeline": "Evolutionary Timeline",
        "evolution_narrative": "Evolution Narrative",
        "paradigm_shifts": "Paradigm Shifts",
        "technical_forks": "Technical Forks",
        "key_papers": "Key Papers",
        "timeline": "Timeline",
        "quality_review": "Quality Review",
        "quality_score": "Quality Score",
        "triggered_by": "triggered by",
    },
    "zh": {
        "title": "研究领域",
        "overview": "领域概述",
        "cross_branch_relationships": "分支间关系",
        "evolutionary_timeline": "演化时间线",
        "evolution_narrative": "发展脉络",
        "paradigm_shifts": "技术范式转变",
        "technical_forks": "技术路线分叉",
        "key_papers": "关键论文",
        "timeline": "时间线",
        "quality_review": "质量审查",
        "quality_score": "质量评分",
        "triggered_by": "触发论文",
    },
}

_REL_TYPE = {
    "en": {
        "precursor_to": "Precursor → Successor",
        "technical_fork": "Technical Fork",
        "parallel_development": "Parallel Development",
        "application_area": "Application Area",
    },
    "zh": {
        "precursor_to": "先驱 → 后继",
        "technical_fork": "技术分叉",
        "parallel_development": "并行发展",
        "application_area": "应用领域",
    },
}

_SEVERITY = {
    "en": {
        "warning": "WARNING",
        "error": "ERROR",
        "info": "INFO",
    },
    "zh": {
        "warning": "警告",
        "error": "错误",
        "info": "提示",
    },
}


def _t(section: str, lang: str) -> str:
    return _SECTION.get(lang, _SECTION["en"]).get(section, section)


def _rel_type(rel: str, lang: str) -> str:
    return _REL_TYPE.get(lang, _REL_TYPE["en"]).get(rel, rel.replace("_", " ").title())


def _sev(severity: str, lang: str) -> str:
    return _SEVERITY.get(lang, _SEVERITY["en"]).get(severity, severity.upper())


def export_markdown(field_map: dict, output_path, lang: str = "en"):
    lines = []

    # ── Title ──
    lines.append(f"# {_t('title', lang)}: {field_map.get('field', 'Unknown')}")
    lines.append("")

    # ── Overview ──
    overview = field_map.get("overview", "")
    if overview:
        lines.append(f"## {_t('overview', lang)}")
        lines.append("")
        for para in overview.split("\n\n"):
            lines.append(para.strip())
            lines.append("")

    # ── Cross-Branch Relationships ──
    relationships = field_map.get("cross_branch_relationships", [])
    if relationships:
        lines.append(f"## {_t('cross_branch_relationships', lang)}")
        lines.append("")
        for rel in relationships:
            branches_list = rel.get("branches", [])
            rel_type = _rel_type(rel.get("relationship", ""), lang)
            desc = rel.get("description", "")
            branch_str = " → ".join(branches_list) if branches_list else "?"
            lines.append(f"- **{branch_str}** — *{rel_type}*")
            if desc:
                lines.append(f"  {desc}")
        lines.append("")

    # ── Temporal ordering ──
    temporal = field_map.get("temporal_ordering", [])
    if temporal:
        lines.append(f"## {_t('evolutionary_timeline', lang)}")
        lines.append("")
        for i, branch_name in enumerate(temporal, 1):
            lines.append(f"{i}. {branch_name}")
        lines.append("")

    # ── Per-branch sections ──
    for branch in field_map.get("branches", []):
        branch_name = branch.get("branch_name", "Unknown")
        lines.append(f"## {branch_name}")
        lines.append("")

        narrative = branch.get("narrative", "")
        if narrative:
            lines.append(f"### {_t('evolution_narrative', lang)}")
            lines.append("")
            for para in narrative.split("\n\n"):
                lines.append(para.strip())
                lines.append("")

        paradigm_shifts = branch.get("paradigm_shifts", [])
        if paradigm_shifts:
            lines.append(f"### {_t('paradigm_shifts', lang)}")
            lines.append("")
            for shift in paradigm_shifts:
                from_ = shift.get("from_approach", "?")
                to_ = shift.get("to_approach", "?")
                trigger = shift.get("trigger_paper", "")
                line = f"- **{from_}** → **{to_}**"
                if trigger:
                    line += f" ({_t('triggered_by', lang)}: {trigger})"
                lines.append(line)
            lines.append("")

        technical_forks = branch.get("technical_forks", [])
        if technical_forks:
            lines.append(f"### {_t('technical_forks', lang)}")
            lines.append("")
            for fork in technical_forks:
                desc = fork.get("description", "")
                papers = fork.get("representative_papers", [])
                lines.append(f"- **{desc}**")
                for p in papers:
                    lines.append(f"  - {p}")
            lines.append("")

        lines.append(f"### {_t('key_papers', lang)}")
        lines.append("")
        lines.append("| # | Paper | Year | Significance |")
        lines.append("|---|---|---|---|")
        key_papers = branch.get("key_papers", [])
        for rank, paper in enumerate(key_papers, 1):
            title = paper.get("title", "?")
            year = paper.get("year", "?")
            significance = (paper.get("significance", "") or "")[:120]
            lines.append(f"| {rank} | {title} | {year} | {significance} |")
        lines.append("")

        lines.append(f"### {_t('timeline', lang)}")
        lines.append("")
        for item in branch.get("timeline", []):
            title = item.get("title", "?")
            year = item.get("year", "?")
            link = item.get("link", "")
            if link:
                lines.append(f"- **{year}** → [{title}]({link})")
            else:
                lines.append(f"- **{year}** → {title}")
        lines.append("")

    # ── Quality review (non‑blocking validation appendix) ──
    validation = field_map.get("_validation", {})
    if validation:
        score = validation.get("quality_score")
        issues = validation.get("issues", [])
        if score or issues:
            lines.append("---")
            lines.append(f"## {_t('quality_review', lang)}")
            lines.append("")
            if score is not None:
                lines.append(f"**{_t('quality_score', lang)}**: {score}/10")
                lines.append("")
            for issue in issues:
                sev_label = _sev(issue.get("severity", "info"), lang)
                desc = issue.get("description", "")
                loc = issue.get("location", "")
                tag = f"[{sev_label}]"
                parts = [f"- {tag}"]
                parts.append(desc)
                if loc:
                    parts.append(f"(*{loc}*)")
                lines.append(" ".join(parts))
            lines.append("")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")