import json
from urllib.parse import urlparse

import requests
from requests import RequestException

from src.classifier import build_branch_embeddings, classify_paper
from src.embedding import build_embedding_client
from src.timeline import build_timeline
from src.config import (
    BRANCHES_PATH,
    EMBEDDING_BASE_URL,
    ENABLE_LOCAL_EMBEDDING_FALLBACK,
    OUTPUT_MARKDOWN_PATH,
    PAPERS_PATH,
    TOP_K_PAPERS,
)


def load_data():
    with PAPERS_PATH.open(encoding="utf-8") as f:
        papers = json.load(f)

    with BRANCHES_PATH.open(encoding="utf-8") as f:
        branches = json.load(f)

    return papers, branches


def build_field_map(papers, branches, client):
    field_map = {}
    branch_embeddings = build_branch_embeddings(branches, client)

    for paper in papers:
        branch = classify_paper(paper, branches, branch_embeddings, client)
        if branch is None:
            continue

        field = branch["field"]
        branch_name = branch["branch"]

        if field not in field_map:
            field_map[field] = {}

        if branch_name not in field_map[field]:
            field_map[field][branch_name] = {"papers": []}

        field_map[field][branch_name]["papers"].append(paper)

    for field in field_map:
        for branch in field_map[field]:
            papers_in_branch = field_map[field][branch]["papers"]

            papers_sorted = sorted(
                papers_in_branch,
                key=lambda x: x["citation_count"],
                reverse=True,
            )

            key_papers = papers_sorted[:TOP_K_PAPERS]
            timeline = build_timeline(key_papers)

            field_map[field][branch]["key_papers"] = key_papers
            field_map[field][branch]["timeline"] = timeline

    return field_map


def _convert_for_export(field_map):
    """Convert V1 nested-dict field_map to V2 list-of-branches format."""
    result = []
    for field_name, branches in field_map.items():
        branch_list = []
        order = 1
        for branch_name, data in branches.items():
            branch_list.append({
                "branch_id": order,
                "branch_name": branch_name,
                "keywords": [],
                "paper_count": len(data["papers"]),
                "key_papers": data["key_papers"],
                "timeline": data["timeline"],
            })
            order += 1
        result.append({"field": field_name, "branches": branch_list})
    return result


def check_embedding_api_reachable(base_url):
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return False, "EMBEDDING_BASE_URL 格式无效"

    try:
        response = requests.get(base_url, timeout=3)
        return True, f"HTTP {response.status_code}"
    except RequestException as exc:
        return False, str(exc)


def main():
    papers, branches = load_data()
    client = build_embedding_client()
    if client is not None:
        reachable, detail = check_embedding_api_reachable(EMBEDDING_BASE_URL)
        if not reachable:
            print(f"Embedding API 不可达: {detail}")
            print("请配置代理后重试（例如设置 HTTPS_PROXY/HTTP_PROXY）。")
            if ENABLE_LOCAL_EMBEDDING_FALLBACK:
                print("已自动切换到本地 embedding fallback。")
                client = None
            else:
                return

    try:
        field_map = build_field_map(papers, branches, client)
    except ValueError as exc:
        print(f"运行失败: {exc}")
        print("请检查 EMBEDDING_BASE_URL、EMBEDDING_MODEL，或配置代理。")
        return

    output_path = OUTPUT_MARKDOWN_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_list = _convert_for_export(field_map)
    parts = []
    for entry in export_list:
        lines = [f"# Field: {entry.get('field', 'Unknown')}", ""]
        for branch in entry.get("branches", []):
            lines.append(f"## {branch['branch_name']}")
            lines.append("")
            lines.append("Key Papers")
            for paper in branch.get("key_papers", []):
                lines.append(
                    f"- {paper['title']} ({paper['year']}) [link]({paper['link']})"
                )
            lines.append("")
            lines.append("Timeline")
            for item in branch.get("timeline", []):
                lines.append(f"{item['year']} -> {item['title']}")
            lines.append("")
        parts.append("\n".join(lines).rstrip())

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(parts) + "\n")

    print(f"Done. Output saved to {output_path}")


if __name__ == "__main__":
    main()