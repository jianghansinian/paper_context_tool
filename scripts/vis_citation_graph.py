"""Visualize the citation graph as an interactive HTML.

Usage:
  python scripts/vis_citation_graph.py <output_dir> [top_n=50] [--dark]

Generates citation_graph.html in the output directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from pyvis.network import Network


def load_json(output_dir: Path, pattern: str) -> list | dict | None:
    paths = sorted(output_dir.glob(pattern))
    if not paths:
        return None
    with open(paths[0]) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_dir> [top_n=50] [--dark]")
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    top_n = 50
    dark_mode = False
    for arg in sys.argv[2:]:
        if arg == "--dark":
            dark_mode = True
        else:
            try:
                top_n = int(arg)
            except ValueError:
                pass

    ranked = load_json(out_dir, "step_4_ranked_all*")
    seeds = load_json(out_dir, "step_2_resolved_seeds*")
    edges = load_json(out_dir, "step_3_graph_edges*")

    if not ranked:
        print("No ranked papers found in", out_dir)
        sys.exit(1)

    # Build title→ranked paper lookup
    seed_titles = {s["title"] for s in seeds} if seeds else set()
    top = ranked[:top_n]
    top_titles = {p["title"] for p in top}

    # Filter edges to only those between visible nodes
    edge_map = {}
    if edges:
        for e in edges:
            u = e.get("source_title", "")
            v = e.get("target_title", "")
            if u in top_titles and v in top_titles:
                key = (u, v)
                edge_map[key] = edge_map.get(key, 0) + 1

    # Build graph
    bg = "#1a1a2e" if dark_mode else "#ffffff"
    fg = "#e0e0e0" if dark_mode else "#333333"

    net = Network(height="800px", width="100%", directed=True, bgcolor=bg, font_color=fg)

    # Physics: stronger gravity to pull clusters together
    net.set_options("""{
      "physics": {
        "stabilization": {"iterations": 150},
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 180,
          "springConstant": 0.05,
          "damping": 0.5
        }
      }
    }""")

    for p in top:
        title = p["title"]
        is_seed = p.get("is_seed", False) or title in seed_titles
        cit = p.get("citation_count", 0) or 0
        year = p.get("year", "?") or "?"
        score = p.get("graph_score", 0) or 0
        cp = p.get("coupling_degree", 0) or 0

        # Size: log scale, clamp to [15, 70]
        size = max(15, min(70, int(cit ** 0.4 * 3)))

        if is_seed:
            color = "#ff6b6b" if not dark_mode else "#ff6b6b"
            border = "#cc4444"
            shape = "star"
        else:
            color = "#4ecdc4" if not dark_mode else "#45b7aa"
            border = "#36a89f"
            shape = "dot"

        hover = (
            f"<b>{title}</b><br>"
            f"Year: {year} | Citations: {cit:,}<br>"
            f"Score: {score:.3f} | Coupling: {cp:.2f}<br>"
            f"{'★ SEED' if is_seed else '○ discovered'}"
        )

        label = title[:35] + "…" if len(title) > 35 else title
        net.add_node(title, label=label, size=size, color=color,
                     border=border, title=hover, shape=shape,
                     font={"size": 10, "color": fg})

    for (u, v), weight in edge_map.items():
        w = min(weight, 5)
        net.add_edge(u, v, arrows="to", color="#888888" if not dark_mode else "#666666",
                     width=w, title=f"{weight} citation(s)")

    html_path = out_dir / "citation_graph.html"
    net.save_graph(str(html_path))
    print(f"Saved: {html_path}")
    print(f"  Nodes: {len(top):,}  Edges: {len(edge_map):,}  (top-{top_n})")
    print(f"  Seeds: {sum(1 for p in top if p.get('is_seed') or p['title'] in seed_titles)}")
    print(f"  Open {html_path} in a browser to interact")


if __name__ == "__main__":
    main()