"""
Evaluation: Sample 35 edges across all layers for manual inspection.
Writes reports/eval_sample.md.

Also computes Precision@K using hand-labeled gold pairs (src, dst, relation).
Each gold pair is labeled as: concept-concept (L1/L3) or file-file (L2).
For each gold pair (A, B), Precision@K = 1 if B appears in A's top-K neighbors, else 0.
Aggregated over all gold pairs at each K.

Sampling plan (35 edges):
  - 10 random L3 typed concept-concept edges
  - 10 random L2 K-NN file pairs
  - 10 cross-domain K-NN edges (src.domain != dst.domain)
  - 5 random L1 PMI edges (sanity)
"""

from pathlib import Path
import random
import json

import pandas as pd
import networkx as nx
import pickle

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)

random.seed(42)

# ── Hand-labeled Gold Pairs for Precision@K ───────────────────────────────────
# Format: (src, dst, label) — label is 'concept' or 'file'
# Concepts: pairs from domain-set analysis and known L3 relations
# Files: same-domain pairs (math/math or econ/econ) from high-confidence K-NN
GOLD_PAIRS = [
    # Concept-concept pairs (L1 PMI verified, domain-related)
    ("微積分", "微分", "concept"),
    ("微積分", "函數", "concept"),
    ("期望值", "機率", "concept"),
    ("期望值", "隨機", "concept"),
    ("機率", "隨機", "concept"),
    ("利率", "通貨膨脹", "concept"),
    ("通貨膨脹", "貨幣", "concept"),
    ("銀行", "貨幣", "concept"),
    ("投資", "交易", "concept"),
    ("套利", "期望值", "concept"),
    ("矩陣", "函數", "concept"),
    ("微分", "矩陣", "concept"),
    ("計量", "投資", "concept"),
    ("數據", "通貨膨脹", "concept"),
    ("隨機", "函數", "concept"),
    # Cross-domain concept pairs (high PMI, different ratios)
    ("微積分", "交易", "concept"),
    ("期望值", "投資", "concept"),
    ("函數", "貨幣", "concept"),
    ("機率", "套利", "concept"),
    ("矩陣", "銀行", "concept"),
    # L3 typed relation pairs (defines, is-a, special-case-of, part-of)
    ("皮耶·波爾", "布勞威爾不動點定理", "concept"),
    ("角穀靜夫定理", "布勞威爾不動點定理", "concept"),
    ("布勞威爾不動點定理", "代數拓撲", "concept"),
    ("美聯儲", "貝爾斯登", "concept"),
    ("格林斯潘", "信貸危機", "concept"),
    ("聯邦房屋金融委員會", "聯邦房屋貸款銀行系統", "concept"),
    # File-file pairs (L2 K-NN, same-domain, high cosine)
    # Math-math high similarity pairs
    (4545, 100410, "file"),
    (100410, 5290611, "file"),
    # Econ-econ high similarity pairs
    (1057405, 590120, "file"),  # "天領" vs "本間宗久" - cross-domain, keep for variety
    # Math-math average similarity
    (101161, 20811, "file"),
    (101161, 5290611, "file"),
    # More concept-concept from L3 relations
    ("布勞威爾不動點定理", "連續函數", "concept"),
    ("狄利克雷問題", "偏微分方程", "concept"),
    ("法伊特 - 湯普森定理", "奇階單群", "concept"),
    ("美聯儲", "定期招標工具", "concept"),
    ("美聯儲", "聯邦基金利率", "concept"),
    ("房利美", "MBS", "concept"),
    ("高盛", "MBS", "concept"),
    ("摩根大通", "貝爾斯登", "concept"),
    ("信貸危機", "瑞銀", "concept"),
    ("伯南克", "美聯儲", "concept"),
    # File-file cross-domain (lower cos but meaningful)
    # These are intentionally cross-domain to test cross-domain recall
]


def compute_precision_at_k(G, gold_pairs, k_values=[1, 3, 5, 10, 20, 50]):
    """
    Compute Precision@K for gold pairs.
    For concept pair (A, B): use L1 PMI neighbors of A (sorted by PMI desc)
    For file pair (A, B): use L2 K-NN neighbors of A (sorted by cos_sim desc)
    Precision@K for a pair = 1 if B in top-K neighbors of A, else 0.
    Returns dict: k -> (precision, count)
    """
    results = {k: [] for k in k_values}

    for src, dst, ptype in gold_pairs:
        # Resolve file IDs to node IDs
        if ptype == "file":
            src_id = str(int(src)) if isinstance(src, (int, float)) else src
            dst_id = str(int(dst)) if isinstance(dst, (int, float)) else dst
        else:
            src_id = src
            dst_id = dst

        # Skip if nodes not in graph
        if src_id not in G:
            continue

        neighbors = []
        if ptype == "file":
            # L2 K-NN neighbors — sort by cos_sim desc
            for _, nbr, d in G.edges(src_id, data=True):
                if d.get("source") == "knn":
                    neighbors.append((nbr, d.get("cos_sim", 0)))
        else:
            # L1 PMI neighbors — sort by PMI desc
            for _, nbr, d in G.edges(src_id, data=True):
                if d.get("source") == "pmi":
                    neighbors.append((nbr, d.get("pmi", 0)))
                elif d.get("source") == "graphrag":
                    neighbors.append((nbr, d.get("weight", 0)))

        neighbors.sort(key=lambda x: x[1], reverse=True)
        top_k = [n for n, _ in neighbors[:max(k_values)]]

        for k in k_values:
            if dst_id in top_k[:k]:
                results[k].append(1)
            else:
                results[k].append(0)

    # Compute precision for each k
    out = {}
    for k in k_values:
        vals = results[k]
        if vals:
            out[k] = (sum(vals) / len(vals), len(vals))
        else:
            out[k] = (0.0, 0)
    return out

# ── Load unified graph ─────────────────────────────────────────────────────────

print("Loading unified graph...")

with open(ROOT / "analysis" / "unified_graph.pkl", "rb") as f:
    G = pickle.load(f)

print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# ── Load supporting data ────────────────────────────────────────────────────────

docs = pd.read_csv(ROOT / "processed" / "docs.csv", dtype={"id": str}, keep_default_na=False)
l3_entities = pd.read_parquet(ROOT / "graphrag_project" / "output" / "entities.parquet")
l3_relations = pd.read_parquet(ROOT / "graphrag_project" / "output" / "relationships.parquet")
sample = pd.read_csv(ROOT / "processed" / "sample_100.csv", dtype={"id": str}, keep_default_na=False)

# id → title lookup (for file nodes)
id_to_title = dict(zip(docs["id"], docs["title"]))
# id → domain lookup
id_to_domain = dict(zip(docs["id"], docs["domain"]))

# L3 entity id → title lookup
entity_id_to_title = dict(zip(l3_entities["id"].astype(str), l3_entities["title"]))

# ── Helper: file node info ─────────────────────────────────────────────────────

def file_info(node_id):
    title = id_to_title.get(node_id, node_id)
    domain = id_to_domain.get(node_id, "?")
    return f"{title} ({domain})"

# ── Helper: concept node info ───────────────────────────────────────────────────

def concept_info(node_id):
    attrs = G.nodes[node_id]
    source = attrs.get("source", "?")
    if source == "graphrag":
        return f"{node_id[:16]}... [L3 entity: {attrs.get('title', attrs.get('entity_type', 'unknown'))}]"
    else:
        return f"{node_id} [L1 concept]"

# ── Sample L3 typed edges ──────────────────────────────────────────────────────

l3_edges = [
    (u, v, d) for u, v, d in G.edges(data=True)
    if d.get("source") == "graphrag" and d.get("relation") is not None
]
print(f"  L3 typed edges: {len(l3_edges)}")

l3_sample = random.sample(l3_edges, min(10, len(l3_edges)))

# ── Sample L2 K-NN edges ───────────────────────────────────────────────────────

l2_edges = [
    (u, v, d) for u, v, d in G.edges(data=True)
    if d.get("source") == "knn"
]
print(f"  L2 K-NN edges: {len(l2_edges)}")

l2_sample = random.sample(l2_edges, min(10, len(l2_edges)))

# ── Sample cross-domain K-NN edges ──────────────────────────────────────────────

cross_edges = [
    (u, v, d) for u, v, d in l2_edges
    if G.nodes[u].get("domain") != G.nodes[v].get("domain")
]
print(f"  Cross-domain K-NN edges: {len(cross_edges)}")

cross_sample = random.sample(cross_edges, min(10, len(cross_edges)))

# ── Sample L1 PMI edges ─────────────────────────────────────────────────────────

l1_edges = [
    (u, v, d) for u, v, d in G.edges(data=True)
    if d.get("source") == "pmi"
]
print(f"  L1 PMI edges: {len(l1_edges)}")

l1_sample = random.sample(l1_edges, min(5, len(l1_edges)))

# ── Build report ────────────────────────────────────────────────────────────────

lines = [
    "# Evaluation Sample — 35 Edges Across All Layers\n",
    "Generated by `src/eval.py` from `analysis/unified_graph.pkl`.\n",
    "---\n",
    "\n",
    "## Sampling Methodology\n",
    "\n",
    "Four layers sampled:\n",
    "- **L3 typed edges** (10): random from L3 typed concept-concept edges (graphrag source, has relation)\n",
    "- **L2 K-NN edges** (10): random from L2 file-file edges (knn source)\n",
    "- **Cross-domain K-NN edges** (10): L2 edges where src.domain ≠ dst.domain\n",
    "- **L1 PMI edges** (5): random from L1 untyped concept edges (pmi source)\n",
    "\n",
    "---\n",
]

# ── L3 typed edges ──────────────────────────────────────────────────────────────
lines.append("## L3 Typed Edges (10)\n\n")
lines.append("| # | Entity A | Entity B | Relation | Weight | Combined Degree |\n")
lines.append("|---:|---|---|---|---:|---:|\n")
for i, (u, v, d) in enumerate(l3_sample, 1):
    u_title = entity_id_to_title.get(u, u[:16] + "...")
    v_title = entity_id_to_title.get(v, v[:16] + "...")
    rel = d.get("relation", "related-to")
    wt = d.get("weight", 0)
    deg = d.get("combined_degree", 0)
    lines.append(f"| {i} | {u_title} | {v_title} | {rel} | {wt:.3f} | {deg} |\n")
lines.append("\n")

# ── L2 K-NN edges ───────────────────────────────────────────────────────────────
lines.append("## L2 K-NN File Edges (10)\n\n")
lines.append("| # | File A | File B | Cosine Sim | Domain A | Domain B | Community Overlap |\n")
lines.append("|---:|---|---|---:|---|---|---:|\n")
for i, (u, v, d) in enumerate(l2_sample, 1):
    u_info = file_info(u)
    v_info = file_info(v)
    cos = d.get("cos_sim", 0)
    comm_ov = d.get("community_overlap", 0)
    dom_u = G.nodes[u].get("domain", "?")
    dom_v = G.nodes[v].get("domain", "?")
    lines.append(f"| {i} | {u_info} | {v_info} | {cos:.4f} | {dom_u} | {dom_v} | {comm_ov:.3f} |\n")
lines.append("\n")

# ── Cross-domain K-NN edges ─────────────────────────────────────────────────────
lines.append("## Cross-Domain K-NN Edges (10)\n\n")
lines.append("| # | File A | File B | Cosine Sim | Domain A | Domain B | Community Overlap |\n")
lines.append("|---:|---|---|---:|---|---|---:|\n")
for i, (u, v, d) in enumerate(cross_sample, 1):
    u_info = file_info(u)
    v_info = file_info(v)
    cos = d.get("cos_sim", 0)
    comm_ov = d.get("community_overlap", 0)
    dom_u = G.nodes[u].get("domain", "?")
    dom_v = G.nodes[v].get("domain", "?")
    lines.append(f"| {i} | {u_info} | {v_info} | {cos:.4f} | {dom_u} | {dom_v} | {comm_ov:.3f} |\n")
lines.append("\n")

# ── L1 PMI edges ───────────────────────────────────────────────────────────────
lines.append("## L1 PMI Edges (5) — Sanity\n\n")
lines.append("| # | Concept A | Concept B | PMI | Co-doc Count | A Ratio (數學) | B Ratio (數學) |\n")
lines.append("|---:|---|---|---:|---:|---:|---:|\n")

def ratio_math(concept):
    attrs = G.nodes.get(concept, {})
    return attrs.get("domain_ratio_math", 0.5)

for i, (u, v, d) in enumerate(l1_sample, 1):
    pmi = d.get("pmi", 0)
    co_doc = d.get("co_doc_count", 0)
    ra = ratio_math(u)
    rb = ratio_math(v)
    lines.append(f"| {i} | {u} | {v} | {pmi:.3f} | {co_doc} | {ra:.2f} | {rb:.2f} |\n")

lines.append("\n---\n")
lines.append(f"\n*Total graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges*\n")

# ── Write ───────────────────────────────────────────────────────────────────────

out_path = OUT_DIR / "eval_sample.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"\nWritten: {out_path}")
print(f"  L3: {len(l3_sample)} | L2: {len(l2_sample)} | Cross: {len(cross_sample)} | L1: {len(l1_sample)}")

# ── Precision@K ────────────────────────────────────────────────────────────────

print("\nComputing Precision@K...")
pk_results = compute_precision_at_k(G, GOLD_PAIRS, k_values=[1, 3, 5, 10, 20, 50])

precision_lines = [
    "\n---\n\n",
    "## Precision@K Evaluation\n\n",
    "Hand-labeled gold pairs used to evaluate retrieval quality across layers.\n",
    "For each gold pair (A, B), Precision@K = 1 if B appears in A's top-K neighbors, else 0.\n\n",
    "| K | Precision | # Evaluated |\n",
    "|---:|---:|---:|\n",
]
for k, (prec, count) in sorted(pk_results.items()):
    precision_lines.append(f"| {k} | {prec:.3f} | {count} |\n")

# Also write detailed per-pair results
detail_lines = [
    "\n### Gold Pair Details\n\n",
    "| Source | Target | Type | In Graph? | Notes |\n",
    "|---|---|---|---|---|\n",
]
for src, dst, ptype in GOLD_PAIRS:
    src_id = str(int(src)) if ptype == "file" and isinstance(src, (int, float)) else src
    dst_id = str(int(dst)) if ptype == "file" and isinstance(dst, (int, float)) else dst
    in_graph = src_id in G
    detail_lines.append(f"| {src_id} | {dst_id} | {ptype} | {in_graph} | |\n")

print("Precision@K results:")
for k, (prec, count) in sorted(pk_results.items()):
    print(f"  K={k:2d}: precision={prec:.3f} (n={count})")

# Append to report
with open(out_path, "a", encoding="utf-8") as f:
    f.writelines(precision_lines)
    f.writelines(detail_lines)

print(f"\nPrecision@K appended to {out_path}")