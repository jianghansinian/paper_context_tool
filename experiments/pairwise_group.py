"""Pairwise similarity grouping: LLM outputs paper-pair relationships,
then deterministic clustering produces groups.

Key insight: LLMs agree on pairwise relationships (A is similar to B) but
disagree on where to draw group boundaries. So:
  Step 1: LLM judges all paper pairs (1 call)
  Step 2: Deterministic clustering from similarity matrix

Usage:
    cd /home/pnc/ws/paper_context_tool
    python experiments/pairwise_group.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_analyzer import build_analyzer_client, _extract_json_object, _resolve_model
from config import LLM_ANALYZER_TIMEOUT_SEC

INPUT_NARRATIVE = "output/v8_mvp_069/narrative.json"
OUTPUT_DIR = Path("output/pairwise_group")
N_RUNS = 5

USER_TEMPLATE = """Below are claims from {n_papers} papers in the field of {field}.

{claims_text}

For EVERY pair of papers below, judge whether they address the SAME specific
research problem AND share the same innovation approach. Return "same" if they
are close enough to belong in the same group, "different" if not.

Be STRICT: two papers are "same" only if they address the same narrow question
with the same core approach. If they address different sub-problems or take
fundamentally different approaches, they are "different".

Return ONLY a valid JSON object:
{{
  "pairs": [
    {{"paper_a": "title1", "paper_b": "title2", "judgment": "same" or "different", "reason": "brief reason"}}
  ]
}}
"""


def load_claims():
    data = json.load(open(INPUT_NARRATIVE))
    claims = data["claims"]
    field = data.get("field_name", "BEV perception")
    by_paper: dict[str, dict] = {}
    for c in claims:
        pid = c["paper_id"]
        if pid not in by_paper:
            by_paper[pid] = {"title": c["paper_title"], "year": c["year"], "month": c["month"], "claims": []}
        by_paper[pid]["claims"].append(c["statement"])
    papers = sorted(by_paper.values(), key=lambda p: (p["year"], p["month"]))
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"[{i}] {p['title']} ({p['year']}-{p['month']:02d})")
        for c in p["claims"]:
            lines.append(f"  Claim: {c}")
        lines.append("")
    return papers, "\n".join(lines), field


def run_once(client, claims_text: str, field: str, n_papers: int) -> dict | None:
    user_prompt = USER_TEMPLATE.format(n_papers=n_papers, field=field, claims_text=claims_text)
    try:
        resp = client.chat.completions.create(
            model=_resolve_model(None),
            messages=[{"role": "user", "content": user_prompt}],
            temperature=1.0,
            timeout=LLM_ANALYZER_TIMEOUT_SEC,
        )
        text = resp.choices[0].message.content or ""
        return _extract_json_object(text)
    except Exception as e:
        print(f"  FAIL: {e}")
        return None


def build_adjacency(result: dict, all_titles: list[str]) -> dict[str, set[str]]:
    """Build paper -> set of same-group neighbors from pairwise judgments."""
    adj: dict[str, set[str]] = {t: set() for t in all_titles}
    for pair in result.get("pairs", []):
        a = pair.get("paper_a", "")
        b = pair.get("paper_b", "")
        if pair.get("judgment") == "same":
            if a in adj and b in adj:
                adj[a].add(b)
                adj[b].add(a)
    return adj


def cluster_connected(adj: dict[str, set[str]]) -> list[set[str]]:
    """Simple connected-components clustering from adjacency."""
    visited = set()
    clusters = []
    for node in adj:
        if node in visited:
            continue
        cluster = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            cluster.add(n)
            for neighbor in adj[n]:
                if neighbor not in visited:
                    stack.append(neighbor)
        clusters.append(cluster)
    return clusters


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    papers, claims_text, field = load_claims()
    n_papers = len(papers)
    all_titles = [p["title"] for p in papers]
    n_pairs = n_papers * (n_papers - 1) // 2
    print(f"{n_papers} papers, {n_pairs} pairs, field={field}")

    client = build_analyzer_client()
    if not client:
        print("ERROR: no LLM client")
        sys.exit(1)

    results = []
    for i in range(1, N_RUNS + 1):
        t0 = time.time()
        result = run_once(client, claims_text, field, n_papers)
        dt = time.time() - t0
        if result is None:
            print(f"  run {i}: FAILED ({dt:.1f}s)")
            continue

        pairs = result.get("pairs", [])
        n_judged = len(pairs)
        n_same = sum(1 for p in pairs if p.get("judgment") == "same")
        n_diff = sum(1 for p in pairs if p.get("judgment") == "different")

        adj = build_adjacency(result, all_titles)
        clusters = cluster_connected(adj)
        sizes = sorted([len(c) for c in clusters], reverse=True)

        print(f"  run {i}: {n_judged} pairs judged ({n_same} same, {n_diff} diff), "
              f"→ {len(clusters)} clusters, sizes={sizes} ({dt:.1f}s)")

        out_path = OUTPUT_DIR / f"run_{i}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        results.append((i, result, adj, clusters))
        time.sleep(2)

    if len(results) < 2:
        print("\nNot enough runs.")
        return

    # ── Pairwise agreement across runs ──
    print(f"\n{'='*70}")
    print("PAIRWISE JUDGMENT STABILITY")
    print(f"{'='*70}")

    # Count how many runs said "same" for each pair
    pair_same_count: dict[tuple[str, str], int] = defaultdict(int)
    pair_reasons: dict[tuple[str, str], list[str]] = defaultdict(list)
    for _, result, _, _ in results:
        for pair in result.get("pairs", []):
            a = pair.get("paper_a", "")
            b = pair.get("paper_b", "")
            key = tuple(sorted([a, b]))
            if pair.get("judgment") == "same":
                pair_same_count[key] += 1
            pair_reasons[key].append(pair.get("reason", ""))

    # Show all pairs sorted by agreement
    print(f"\n{'pair':<60} {'same/runs':>9}  {'agreement'}")
    print("-" * 85)
    sorted_pairs = sorted(pair_same_count.keys(), key=lambda k: -pair_same_count[k])
    for key in sorted_pairs:
        a_short = key[0].split(":")[0][:28]
        b_short = key[1].split(":")[0][:28]
        n_same = pair_same_count[key]
        if n_same == len(results):
            marker = "ALWAYS SAME"
        elif n_same == 0:
            marker = "ALWAYS DIFF"
        elif n_same > len(results) / 2:
            marker = "MOSTLY SAME"
        else:
            marker = "MOSTLY DIFF"
        print(f"  {a_short:<28} + {b_short:<28} {n_same}/{len(results)}  {marker}")

    # ── Cluster stability ──
    print(f"\n{'='*70}")
    print("CLUSTER STABILITY (connected components from pairwise same-judgments)")
    print(f"{'='*70}")

    partitions = []
    for i, _, _, clusters in results:
        partition = frozenset(frozenset(c) for c in clusters)
        partitions.append(partition)
        print(f"\n  run {i} ({len(clusters)} clusters):")
        for c in sorted(clusters, key=lambda x: -len(x)):
            titles = [t.split(":")[0][:40] for t in c]
            print(f"    [{len(c)}] {', '.join(titles)}")

    unique_partitions = set(partitions)
    print(f"\n  Unique partitions: {len(unique_partitions)} / {N_RUNS}")

    # Jaccard
    print(f"\n  PAIRWISE JACCARD:")
    print("      " + "  ".join(f"r{i}" for i, _, _, _ in results))
    for i, p1 in enumerate(partitions):
        row = []
        for j, p2 in enumerate(partitions):
            if p1 == p2:
                row.append("1.00")
            elif not p1 or not p2:
                row.append(" ?? ")
            else:
                inter = len(p1 & p2)
                union = len(p1 | p2)
                row.append(f"{inter/union:.2f}")
        print(f"  r{i+1}  " + "  ".join(row))

    # ── Consensus matrix ──
    print(f"\n{'='*70}")
    print("CONSENSUS GROUPING (majority vote: same if ≥50% runs agree)")
    print(f"{'='*70}")
    consensus_adj: dict[str, set[str]] = {t: set() for t in all_titles}
    threshold = len(results) / 2
    for key in pair_same_count:
        if pair_same_count[key] >= threshold:
            consensus_adj[key[0]].add(key[1])
            consensus_adj[key[1]].add(key[0])

    consensus_clusters = cluster_connected(consensus_adj)
    print(f"\n  Consensus: {len(consensus_clusters)} clusters")
    for c in sorted(consensus_clusters, key=lambda x: -len(x)):
        titles = [t.split(":")[0][:40] for t in c]
        print(f"    [{len(c)}] {', '.join(titles)}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
