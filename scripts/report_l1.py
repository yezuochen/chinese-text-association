"""
Report L1: PMI Concept Graph — Domain-Set Analysis
Generates tables and figures for REPORT.md
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

# Chinese font setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "MingLiU"]
plt.rcParams["axes.unicode_minus"] = False

# ── Load data ──────────────────────────────────────────────────────────────
pmi = pd.read_parquet(ROOT / "analysis" / "pmi_graph.parquet")
import pyarrow.parquet as pq
table = pq.read_table(ROOT / "analysis" / "concept_domains.parquet")
dom = table.to_pandas()
dom.columns = ["concept", "doc_freq", "docs_經濟金融", "docs_數學", "cross_domain_count"]
docs = pd.read_csv(ROOT / "processed" / "docs.csv", keep_default_na=False, dtype={"id": str})

# PMI degree per concept (count of unique neighbors)
pmi_deg = pmi.groupby("src").size().reset_index(name="n_edges_src")
pmi_deg_dst = pmi.groupby("dst").size().reset_index(name="n_edges_dst")
pmi_deg = pmi_deg.merge(pmi_deg_dst, left_on="src", right_on="dst", how="outer")
pmi_deg = pmi_deg.fillna(0)
pmi_deg["pmi_degree"] = pmi_deg["n_edges_src"] + pmi_deg["n_edges_dst"]
pmi_deg = pmi_deg.set_index("src")["pmi_degree"].to_dict()

# Also compute degree count (number of distinct connected edges per concept)
deg_count = pmi["src"].value_counts().add(pmi["dst"].value_counts(), fill_value=0).to_dict()

# ── Domain-set concepts (15 words) ───────────────────────────────────────
DOMAIN_SET = [
    "微積分", "期望值", "計量", "通貨膨脹", "數據",
    "貨幣", "銀行", "機率", "隨機", "微分",
    "函數", "矩陣", "投資", "套利", "交易", "利率"
]

# Filter to concepts that actually exist in L1
vocab = set(dom["concept"].tolist())
domain_set = [w for w in DOMAIN_SET if w in vocab]
missing = [w for w in DOMAIN_SET if w not in vocab]
if missing:
    print(f"[WARN] Domain-set words not in L1 vocabulary: {missing}")

# Build co-occurrence subgraph among domain-set
subset = pmi[pmi["src"].isin(domain_set) & pmi["dst"].isin(domain_set)].copy()
subset = subset[subset["src"] != subset["dst"]]

def domain_ratio(concept):
    row = dom[dom["concept"] == concept]
    if len(row) == 0:
        return 0.5
    m = row["docs_數學"].values[0]
    e = row["docs_經濟金融"].values[0]
    total = m + e
    if total == 0:
        return 0.5
    return m / total

# ── Table 1.1: Domain-Set Concept Statistics ───────────────────────────────
concept_stats = []
for w in domain_set:
    m_val = dom[dom["concept"] == w]["docs_數學"].values
    e_val = dom[dom["concept"] == w]["docs_經濟金融"].values
    m = int(m_val[0]) if len(m_val) else 0
    e = int(e_val[0]) if len(e_val) else 0
    deg = int(pmi_deg.get(w, 0))
    edge_n = int(deg_count.get(w, 0))
    total = m + e
    ratio = m / total if total > 0 else 0.5
    concept_stats.append({
        "concept": w,
        "pmi_degree": deg,
        "edge_count": edge_n,
        "docs_數學": m,
        "docs_經濟金融": e,
        "ratio_math": ratio
    })

stats_df = pd.DataFrame(concept_stats).sort_values("pmi_degree", ascending=False).reset_index(drop=True)

table_path = OUT / "table1_1.md"
with open(table_path, "w", encoding="utf-8") as f:
    f.write("| Rank | Concept | PMI Degree | Edge Count | 數學 Docs | 經濟金融 Docs | Ratio (數學) |\n")
    f.write("|---:|---|---:|---:|---:|---:|---:|\n")
    for i, row in stats_df.iterrows():
        f.write(f"| {i+1} | {row['concept']} | {row['pmi_degree']} | {row['edge_count']} | {row['docs_數學']} | {row['docs_經濟金融']} | {row['ratio_math']:.2f} |\n")
print(f"Saved: {table_path}")

# ── Table 1.2: Top 10 Edges within Domain Set ───────────────────────────────
# Also include domain-set x non-domain-set top edges for richness
top_edges = subset.nlargest(10, "pmi")[["src", "dst", "pmi", "co_doc_count"]].reset_index(drop=True)

table_path2 = OUT / "table1_2.md"
with open(table_path2, "w", encoding="utf-8") as f:
    f.write("| Rank | Concept A | Concept B | PMI | Co-doc Freq | A Ratio (數學) | B Ratio (數學) |\n")
    f.write("|---:|---|---|---:|---:|---:|---:|\n")
    for i, row in top_edges.iterrows():
        ra = domain_ratio(row["src"])
        rb = domain_ratio(row["dst"])
        f.write(f"| {i+1} | {row['src']} | {row['dst']} | {row['pmi']:.2f} | {row['co_doc_count']} | {ra:.2f} | {rb:.2f} |\n")
print(f"Saved: {table_path2}")

# ── Figure 1.1: Domain-Set PMI Network (NetworkX spring layout) ─────────
try:
    import matplotlib.colors as mcolors
    import numpy as np
    import networkx as nx

    G_nodes = domain_set
    edges_for_g = subset[subset["pmi"] >= 2.0].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Build NetworkX graph for spring layout
    G = nx.Graph()
    G.add_nodes_from(G_nodes)
    for _, row in edges_for_g.iterrows():
        G.add_edge(row["src"], row["dst"], pmi=row["pmi"])

    pos = nx.spring_layout(G, k=1.5, iterations=150, seed=42)

    ratios = {w: domain_ratio(w) for w in G_nodes}
    node_colors = [plt.cm.GnBu(ratios[w]) for w in G_nodes]

    # Draw edges using NetworkX
    for (src, dst, data) in G.edges(data=True):
        if src in pos and dst in pos:
            lw = min(data.get("pmi", 2) / 2, 3)
            ax.annotate("", xy=pos[dst], xytext=pos[src],
                        arrowprops=dict(arrowstyle="-", color="gray",
                                        linewidth=lw, alpha=0.5))

    # Draw nodes with proper annotation
    for i, w in enumerate(G_nodes):
        ax.scatter(pos[w][0], pos[w][1], c=[node_colors[i]],
                   s=1200, zorder=2, edgecolors="black", linewidths=1.0)
        ax.annotate(w, (pos[w][0], pos[w][1]),
                    fontsize=11, ha="center", va="center",
                    fontweight="bold", zorder=3)

    ax.set_title("L1: Domain-Set PMI Network", fontsize=13, fontweight="bold")
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap="GnBu", norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    ax2 = fig.add_axes([0.82, 0.15, 0.04, 0.3])
    cbar = plt.colorbar(sm, cax=ax2)
    cbar.set_label("數學 Ratio", fontsize=8)

    fig.savefig(OUT / "fig1_1_pminetwork.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT / 'fig1_1_pminetwork.png'}")
except Exception as e:
    import traceback
    print(f"[WARN] Could not generate figure: {e}")
    traceback.print_exc()

# Summary
total_docs = len(docs)
vocab_size = len(dom)
print(f"\nL1 Summary:")
print(f"  Total documents: {total_docs}")
print(f"  Vocabulary (doc_freq>=5): {vocab_size}")
print(f"  PMI edges: {len(pmi):,}")
print(f"  數學 docs: {len(docs[docs['domain']=='數學'])} | 經濟金融 docs: {len(docs[docs['domain']=='經濟金融'])}")
print(f"  Domain-set found in L1: {len(domain_set)}/{len(DOMAIN_SET)}")