from pathlib import Path


def export_markdown(field_map: dict, output_path):
    lines = [f"# Field: {field_map.get('field', 'Unknown')}", ""]

    for branch in field_map.get("branches", []):
        lines.append(f"## {branch['branch_name']}")
        lines.append("")
        lines.append("Key Papers")
        for paper in branch.get("key_papers", []):
            score = paper.get("score")
            if score is None:
                lines.append(f"- {paper['title']} ({paper['year']}) [link]({paper['link']})")
            else:
                lines.append(
                    f"- {paper['title']} ({paper['year']}) [score={score}] [link]({paper['link']})"
                )
        lines.append("")
        lines.append("Timeline")
        for item in branch.get("timeline", []):
            lines.append(f"{item['year']} → {item['title']}")
        lines.append("")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")
