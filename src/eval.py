"""
Evaluation: Sample 35 edges across all layers for manual inspection.
Writes reports/eval_sample.md.

Also computes Precision@K using hand-labeled gold pairs (src, dst, relation).
Each gold pair is labeled as: concept or file
For each gold pair (A, B), Precision@K = 1 if B appears in A's top-K neighbors, else 0.
Aggregated over all gold pairs at each K.

Gold pair type semantics:
  - 'pmi-concept': use L1 PMI neighbors only (sorted by PMI desc)
  - 'l3-concept': use L3 graphrag typed neighbors only (sorted by combined_degree desc)
  - 'file': use L2 K-NN neighbors only (sorted by cos_sim desc)

Sampling plan (35 edges):
  - 10 random L3 typed concept-concept edges
  - 10 random L2 K-NN file pairs
  - 10 cross-domain K-NN edges (src.domain != dst.domain)
  - 5 random L1 PMI edges (sanity)
"""

from pathlib import Path
import random
import pickle

import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)

random.seed(42)

# ── Hand-labeled Gold Pairs for Precision@K ───────────────────────────────────
# Format: (src, dst, label)
# 'pmi-concept': L1/L2 concept-concept verified by PMI (domain set + cross-domain)
# 'l3-concept': L3 typed relation pairs — use Chinese entity titles (fuse_graph maps title→UUID)
# 'file': same-domain K-NN file pairs
GOLD_PAIRS = [
    # PMI concept-concept pairs (verified L1 neighbors)
    ("微積分", "微分", "pmi-concept"),
    ("微積分", "函數", "pmi-concept"),
    ("期望值", "機率", "pmi-concept"),
    ("期望值", "隨機", "pmi-concept"),
    ("機率", "隨機", "pmi-concept"),
    ("利率", "通貨膨脹", "pmi-concept"),
    ("通貨膨脹", "貨幣", "pmi-concept"),
    ("銀行", "貨幣", "pmi-concept"),
    ("投資", "交易", "pmi-concept"),
    ("套利", "期望值", "pmi-concept"),
    ("矩陣", "函數", "pmi-concept"),
    ("微分", "矩陣", "pmi-concept"),
    ("計量", "投資", "pmi-concept"),
    ("數據", "通貨膨脹", "pmi-concept"),
    ("隨機", "函數", "pmi-concept"),
    # Cross-domain PMI pairs
    ("微積分", "交易", "pmi-concept"),
    ("期望值", "投資", "pmi-concept"),
    ("函數", "貨幣", "pmi-concept"),
    ("機率", "套利", "pmi-concept"),
    ("矩陣", "銀行", "pmi-concept"),
    # L3 typed relation pairs (Chinese entity titles → UUID via fuse_graph reverse map)
    ("皮耶·波爾", "布勞威爾不動點定理", "l3-concept"),
    ("角穀靜夫定理", "布勞威爾不動點定理", "l3-concept"),
    ("布勞威爾不動點定理", "代數拓撲", "l3-concept"),
    ("美聯儲", "貝爾斯登", "l3-concept"),
    ("格林斯潘", "信貸危機", "l3-concept"),
    ("聯邦房屋金融委員會", "聯邦房屋貸款銀行系統", "l3-concept"),
    ("布勞威爾不動點定理", "連續函數", "l3-concept"),
    ("狄利克雷問題", "偏微分方程", "l3-concept"),
    ("法伊特 - 湯普森定理", "奇階單群", "l3-concept"),
    ("美聯儲", "定期招標工具", "l3-concept"),
    ("美聯儲", "聯邦基金利率", "l3-concept"),
    ("房利美", "MBS", "l3-concept"),
    ("高盛", "MBS", "l3-concept"),
    ("摩根大通", "貝爾斯登", "l3-concept"),
    ("信貸危機", "瑞銀", "l3-concept"),
    ("伯南克", "美聯儲", "l3-concept"),
    # File-file pairs (L2 K-NN, same-domain, high cosine)
    (4545, 100410, "file"),
    (100410, 5290611, "file"),
    (101161, 20811, "file"),
    (101161, 5290611, "file"),
]


def compute_precision_at_k(G, gold_pairs, k_values=[1, 3, 5, 10, 20, 50]):
    """
    Compute Precision@K for gold pairs.
    For PMI-concept pair: use L1 PMI neighbors only (sorted by PMI desc).
    For L3-concept pair: use L3 graphrag typed neighbors only (sorted by combined_degree desc).
    For file pair: use L2 K-NN neighbors (sorted by cos_sim desc).
    Precision@K = 1 if B in top-K neighbors of A, else 0.
    Returns dict: k -> (precision, count)
    """
    # Build L3 title → entity_id reverse map from graph node attributes
    l3_title_to_eid: dict[str, str] = {}
    for nid in G.nodes:
        attrs = G.nodes[nid]
        if attrs.get("source") == "graphrag" and attrs.get("title"):
            l3_title_to_eid[attrs["title"]] = nid

    results = {k: [] for k in k_values}

    for s_raw, d_raw, ptype in gold_pairs:
        src_id: str
        dst_id: str

        if ptype == "file":
            src_id = str(int(float(s_raw))) if isinstance(s_raw, (int, float)) else str(s_raw)
            dst_id = str(int(float(d_raw))) if isinstance(d_raw, (int, float)) else str(d_raw)
        elif ptype == "l3-concept":
            # Chinese entity title → UUID via reverse map (fuse_graph built this)
            src_id = l3_title_to_eid.get(str(s_raw), str(s_raw))
            dst_id = l3_title_to_eid.get(str(d_raw), str(d_raw))
        else:
            # 'pmi-concept': use title as-is
            src_id = str(s_raw)
            dst_id = str(d_raw)

        if src_id not in G:
            continue

        nbrs: list[tuple[str, float]] = []
        if ptype == "file":
            for _, n, d in G.edges(src_id, data=True):
                if d.get("source") == "knn":
                    nbrs.append((n, d.get("cos_sim", 0)))
        elif ptype == "l3-concept":
            for _, n, d in G.edges(src_id, data=True):
                if d.get("source") == "graphrag" and d.get("relation") is not None:
                    nbrs.append((n, d.get("combined_degree", d.get("weight", 0))))
        else:
            # L1 PMI concept neighbors ONLY — no graphrag mixing
            for _, n, d in G.edges(src_id, data=True):
                if d.get("source") == "pmi":
                    nbrs.append((n, d.get("pmi", 0)))

        nbrs.sort(key=lambda x: x[1], reverse=True)
        top_k = [n for n, _ in nbrs[:max(k_values)]]

        for k in k_values:
            results[k].append(1 if dst_id in top_k[:k] else 0)

    out = {}
    for k in k_values:
        vals = results[k]
        out[k] = (sum(vals) / len(vals), len(vals)) if vals else (0.0, 0)
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

# id → title lookup (for file nodes)
id_to_title = dict(zip(docs["id"], docs["title"]))
# id → domain lookup
id_to_domain = dict(zip(docs["id"], docs["domain"]))

# L3 entity id → title lookup (for display)
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

# ── Sample cross-domain K-NN edges ─────────────────────────────────────────────

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

# ── Build report ───────────────────────────────────────────────────────────────

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

# ── Precision@K ───────────────────────────────────────────────────────────────

print("\nComputing Precision@K...")
pk_results = compute_precision_at_k(G, GOLD_PAIRS, k_values=[1, 3, 5, 10, 20, 50])

precision_lines = [
    "\n---\n\n",
    "## Precision@K Evaluation\n\n",
    "Hand-labeled gold pairs used to evaluate retrieval quality across layers.\n",
    "Types: 'pmi-concept' → L1 PMI neighbors only; 'l3-concept' → L3 graphrag typed neighbors only; 'file' → L2 K-NN neighbors only.\n",
    "For each gold pair (A, B), Precision@K = 1 if B appears in A's top-K neighbors, else 0.\n\n",
    "| K | Precision | # Evaluated |\n",
    "|---:|---:|\n",
]
for k, (prec, count) in sorted(pk_results.items()):
    precision_lines.append(f"| {k} | {prec:.3f} | {count} |\n")

# Per-pair detail
detail_lines = [
    "\n### Gold Pair Details\n\n",
    "| Source | Target | Type | In Graph? | Top-10 Neighbors (truncated) |\n",
    "|---|---|---|---|---|\n",
]
for src_raw, dst_raw, ptype in GOLD_PAIRS:
    if ptype == "file":
        src_id = str(int(float(src_raw))) if isinstance(src_raw, (int, float)) else str(src_raw)
        dst_id = str(int(float(dst_raw))) if isinstance(dst_raw, (int, float)) else str(dst_raw)
    elif ptype == "l3-concept":
        l3_title_to_eid = {
            attrs["title"]: nid
            for nid in G.nodes
            if (attrs := G.nodes[nid]).get("source") == "graphrag" and attrs.get("title")
        }
        src_id = l3_title_to_eid.get(str(src_raw), str(src_raw))
        dst_id = l3_title_to_eid.get(str(dst_raw), str(dst_raw))
    else:
        src_id = str(src_raw)
        dst_id = str(dst_raw)
    in_graph = src_id in G

    # Collect top-10 neighbors for debug
    nbrs_sample = []
    if src_id in G:
        if ptype == "file":
            edges = [(n, d.get("cos_sim", 0)) for _, n, d in G.edges(src_id, data=True) if d.get("source") == "knn"]
        elif ptype == "l3-concept":
            edges = [(n, d.get("combined_degree", 0)) for _, n, d in G.edges(src_id, data=True) if d.get("source") == "graphrag" and d.get("relation") is not None]
        else:
            edges = [(n, d.get("pmi", 0)) for _, n, d in G.edges(src_id, data=True) if d.get("source") == "pmi"]
        edges.sort(key=lambda x: x[1], reverse=True)
        nbrs_sample = [n for n, _ in edges[:10]]
        nbrs_sample = nbrs_sample[:5]  # truncate for table width

    detail_lines.append(f"| {src_id} | {dst_id} | {ptype} | {in_graph} | {nbrs_sample} |\n")

print("Precision@K results:")
for k, (prec, count) in sorted(pk_results.items()):
    print(f"  K={k:2d}: precision={prec:.3f} (n={count})")

# Append to report
with open(out_path, "a", encoding="utf-8") as f:
    f.writelines(precision_lines)
    f.writelines(detail_lines)

print(f"\nPrecision@K appended to {out_path}")