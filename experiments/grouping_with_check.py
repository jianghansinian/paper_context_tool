"""Grouping with deterministic cross-check against problem extraction.

Pipeline:
  Step 1: per-paper problem extraction (1 LLM call) — semantic anchor
  Step 2: top-down grouping (5 LLM calls) — candidate groupings
  Step 3: deterministic cross-check — validate each grouping against problems

Cross-check catches catastrophic grouping errors (e.g. Sparse4D + SparseDrive
in same group when their problems are fundamentally different).

Usage:
    cd /home/pnc/ws/paper_context_tool
    python experiments/grouping_with_check.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_analyzer import build_analyzer_client, _extract_json_object, _resolve_model
from config import LLM_ANALYZER_TIMEOUT_SEC

INPUT_NARRATIVE = "output/v8_mvp_069/narrative.json"
OUTPUT_DIR = Path("output/grouping_with_check")
N_GROUP_RUNS = 5

# ── Step 1: Problem extraction prompt ──
PROBLEM_USER = """Below are claims from {n_papers} papers in the field of {field}.

{claims_text}

For each paper, state the ONE research problem it addresses. A research problem
is the question the paper tries to answer — not its method.

Return ONLY valid JSON, no markdown:
{{
  "papers": [
    {{"paper": "full paper title as shown above", "problem": "one-sentence research problem"}}
  ]
}}
"""

# ── Step 2: Grouping prompt ──
GROUP_USER = """Below are claims from {n_papers} papers in the field of {field}.

{claims_text}

Group these papers into research problem groups. Papers that address the SAME
research problem should be in the SAME group. Papers that address DIFFERENT
research problems should be in DIFFERENT groups.

Each paper must appear in exactly one group.

Return ONLY a valid JSON object, no markdown fences, no other text:
{{
  "groups": [
    {{
      "problem": "one sentence: what research problem does this group address?",
      "papers": ["full paper title as it appears above"]
    }}
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


def llm_call(client, prompt: str) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=_resolve_model(None),
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            timeout=LLM_ANALYZER_TIMEOUT_SEC,
        )
        text = resp.choices[0].message.content or ""
        return _extract_json_object(text)
    except Exception as e:
        print(f"  LLM FAIL: {e}")
        return None


# ── Step 3: Deterministic cross-check ──

def cross_check(grouping: dict, problems: dict[str, str]) -> dict:
    """Check grouping against per-paper problem labels.

    Returns a report with:
      - intra_violations: pairs of papers in the same group whose problems differ
      - inter_violations: pairs of papers in different groups whose problems are similar
      - score: 0-1, higher = more consistent
      - flagged: list of group indices with violations
    """
    # build paper -> group_idx mapping
    paper_to_group: dict[str, int] = {}
    group_problems: dict[int, list[str]] = defaultdict(list)
    groups = grouping.get("groups", [])

    for gi, g in enumerate(groups):
        for p in g.get("papers", []):
            paper_to_group[p] = gi
            if p in problems:
                group_problems[gi].append(problems[p])

    # Intra-group check: papers in same group should have similar problems
    # Simple heuristic: within each group, check if problem keywords overlap
    intra_violations = []
    for gi, g in enumerate(groups):
        papers_in = g.get("papers", [])
        probs = [problems.get(p, "") for p in papers_in]
        # check pairwise keyword overlap
        for i in range(len(probs)):
            for j in range(i + 1, len(probs)):
                sim = _keyword_similarity(probs[i], probs[j])
                if sim < 0.15:  # very low overlap = different problems in same group
                    intra_violations.append({
                        "group": gi,
                        "group_label": g.get("problem", "")[:60],
                        "paper_a": papers_in[i][:60],
                        "problem_a": probs[i][:80],
                        "paper_b": papers_in[j][:60],
                        "problem_b": probs[j][:80],
                        "similarity": round(sim, 3),
                    })

    # Inter-group check: groups should have distinct problems
    inter_violations = []
    for gi in range(len(groups)):
        for gj in range(gi + 1, len(groups)):
            # compare group problem labels
            label_i = groups[gi].get("problem", "")
            label_j = groups[gj].get("problem", "")
            sim = _keyword_similarity(label_i, label_j)
            if sim > 0.5:  # high overlap = groups too similar
                inter_violations.append({
                    "group_a": gi,
                    "label_a": label_i[:80],
                    "group_b": gj,
                    "label_b": label_j[:80],
                    "similarity": round(sim, 3),
                })

    # Score: penalize intra violations heavily (wrong grouping),
    # penalize inter violations mildly (redundant groups)
    n_pairs = sum(len(g.get("papers", [])) * (len(g.get("papers", [])) - 1) // 2 for g in groups)
    n_pairs = max(n_pairs, 1)
    intra_penalty = len(intra_violations) / n_pairs
    inter_penalty = len(inter_violations) * 0.1 / max(len(groups), 1)
    score = max(0, 1.0 - intra_penalty - inter_penalty)

    flagged_groups = set()
    for v in intra_violations:
        flagged_groups.add(v["group"])
    for v in inter_violations:
        flagged_groups.add(v["group_a"])
        flagged_groups.add(v["group_b"])

    return {
        "score": round(score, 3),
        "intra_violations": intra_violations,
        "inter_violations": inter_violations,
        "flagged_groups": sorted(flagged_groups),
        "n_intra": len(intra_violations),
        "n_inter": len(inter_violations),
    }


def _keyword_similarity(a: str, b: str) -> float:
    """Simple keyword overlap similarity. No LLM, no embeddings."""
    if not a or not b:
        return 0.0
    stop = {"a", "an", "the", "in", "on", "for", "of", "to", "and", "or", "is", "are",
            "how", "can", "from", "by", "with", "be", "that", "this", "which", "it",
            "as", "do", "does", "not", "while", "using", "based", "via", "into",
            "its", "their", "has", "have", "been", "was", "were", "will", "would",
            "should", "could", "may", "might", "than", "more", "such", "these",
            "those", "each", "per", "among", "between", "through", "without",
            "no", "only", "own", "same", "so", "up", "out", "also", "just"}
    wa = set(a.lower().replace("?", "").replace(",", "").replace(".", "").split()) - stop
    wb = set(b.lower().replace("?", "").replace(",", "").replace(".", "").split()) - stop
    if not wa or not wb:
        return 0.0
    inter = wa & wb
    union = wa | wb
    return len(inter) / len(union)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    papers, claims_text, field = load_claims()
    n_papers = len(papers)
    print(f"{n_papers} papers, {sum(len(p['claims']) for p in papers)} claims\n")

    client = build_analyzer_client()
    if not client:
        print("ERROR: no LLM client")
        sys.exit(1)

    # ── Step 1: Problem extraction (1 call, semantic anchor) ──
    print("=" * 60)
    print("STEP 1: Per-paper problem extraction (anchor)")
    print("=" * 60)
    prob_prompt = PROBLEM_USER.format(n_papers=n_papers, field=field, claims_text=claims_text)
    prob_result = llm_call(client, prob_prompt)
    if not prob_result:
        print("FATAL: problem extraction failed")
        sys.exit(1)

    # Build paper_title -> problem mapping
    problems: dict[str, str] = {}
    for p in prob_result.get("papers", []):
        problems[p.get("paper", "")] = p.get("problem", "")

    print(f"  Extracted {len(problems)} problems")
    for title, prob in sorted(problems.items(), key=lambda x: x[0][:30]):
        print(f"    {title[:50]}: {prob[:70]}")

    # Save anchor
    (OUTPUT_DIR / "problem_anchor.json").write_text(
        json.dumps(prob_result, indent=2, ensure_ascii=False))

    # ── Step 2: Grouping (5 calls) ──
    print(f"\n{'=' * 60}")
    print(f"STEP 2: Top-down grouping ({N_GROUP_RUNS} runs)")
    print("=" * 60)
    group_prompt = GROUP_USER.format(n_papers=n_papers, field=field, claims_text=claims_text)

    groupings = []
    for i in range(1, N_GROUP_RUNS + 1):
        t0 = time.time()
        result = llm_call(client, group_prompt)
        dt = time.time() - t0
        if result is None:
            print(f"  run {i}: FAILED ({dt:.1f}s)")
            continue
        ng = len(result.get("groups", []))
        sizes = [len(g.get("papers", [])) for g in result.get("groups", [])]
        print(f"  run {i}: {ng} groups, sizes={sizes} ({dt:.1f}s)")
        out_path = OUTPUT_DIR / f"group_run_{i}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        groupings.append((i, result))
        time.sleep(2)

    # ── Step 3: Cross-check ──
    print(f"\n{'=' * 60}")
    print("STEP 3: Cross-check (deterministic)")
    print("=" * 60)

    reports = []
    for i, grouping in groupings:
        report = cross_check(grouping, problems)
        reports.append((i, report))
        status = "PASS" if report["score"] >= 0.8 else "WARN" if report["score"] >= 0.5 else "FAIL"
        print(f"\n  run {i}: score={report['score']:.3f} [{status}]")
        print(f"    intra_violations={report['n_intra']}, inter_violations={report['n_inter']}, "
              f"flagged_groups={report['flagged_groups']}")

        if report["intra_violations"]:
            print(f"    INTRA violations (papers in same group, different problems):")
            for v in report["intra_violations"]:
                print(f"      {v['paper_a'][:40]} vs {v['paper_b'][:40]} (sim={v['similarity']})")
                print(f"        A: {v['problem_a'][:70]}")
                print(f"        B: {v['problem_b'][:70]}")

        if report["inter_violations"]:
            print(f"    INTER violations (groups too similar):")
            for v in report["inter_violations"]:
                print(f"      group {v['group_a']} vs {v['group_b']} (sim={v['similarity']})")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'run':>4}  {'groups':>6}  {'score':>6}  {'intra':>5}  {'inter':>5}  {'status':>6}")
    print("-" * 45)
    for i, report in reports:
        status = "PASS" if report["score"] >= 0.8 else "WARN" if report["score"] >= 0.5 else "FAIL"
        ng = len(groupings[i-1][1].get("groups", [])) if i-1 < len(groupings) else "?"
        print(f"  {i:>2}  {ng:>6}  {report['score']:>6.3f}  {report['n_intra']:>5}  {report['n_inter']:>5}  {status:>6}")

    # Show grouping detail for each run
    print(f"\n{'=' * 60}")
    print("GROUPING DETAIL")
    print("=" * 60)
    for i, grouping in groupings:
        report = dict(reports)[i] if i in dict(reports) else None
        groups = grouping.get("groups", [])
        flag_marker = " ***FLAGGED***" if report and report["score"] < 0.8 else ""
        print(f"\n  run {i}{flag_marker}")
        for gi, g in enumerate(groups):
            prob = g.get("problem", "??")
            papers_in = g.get("papers", [])
            flag = " ⚠" if report and gi in report.get("flagged_groups", []) else ""
            print(f"    [{len(papers_in)}] {prob[:70]}{flag}")
            for p in papers_in:
                short = p.split(":")[0][:45]
                print(f"        - {short}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
