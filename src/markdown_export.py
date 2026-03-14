from pathlib import Path


def export_markdown(field_map, output_path):
    md = ""

    for field in field_map:
        md += f"# Field: {field}\n\n"

        for branch in field_map[field]:
            md += f"## Branch: {branch}\n\n"
            md += "Key Papers\n"

            for paper in field_map[field][branch]["key_papers"]:
                md += f"- {paper['title']} ({paper['year']}) [link]({paper['link']})\n"

            md += "\nTimeline\n"

            for paper in field_map[field][branch]["timeline"]:
                md += f"{paper['year']} → {paper['title']}\n"

            md += "\n"

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        file.write(md)
