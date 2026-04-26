"""
Report L3: GraphRAG Visualization — Entities, Relations, Communities
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

# ── Load GraphRAG data ─────────────────────────────────────────────────────
ent = pd.read_parquet(ROOT / "graphrag_project" / "output" / "entities.parquet")
rel = pd.read_parquet(ROOT / "graphrag_project" / "output" / "relationships.parquet")
com = pd.read_parquet(ROOT / "graphrag_project" / "output" / "communities.parquet")

print(f"Entities: {ent.shape}")
print(f"Relations: {rel.shape}")
print(f"Communities: {com.shape}")
print(f"Relation types: {rel['description'].value_counts().to_dict()}")

# ── Table 3.1: Entity Type Distribution ───────────────────────────────────
ent_type_counts = ent["type"].value_counts().head(10)
table_path = OUT / "table3_1.md"
with open(table_path, "w", encoding="utf-8") as f:
    f.write("| Entity Type | Count |\n")
    f.write("|---|---:|\n")
    for t, c in ent_type_counts.items():
        f.write(f"| {t} | {c} |\n")
print(f"Saved: {table_path}")

# ── Table 3.2: Leiden Communities — ALL entities listed ─────────────────
id_to_title = dict(zip(ent["id"], ent["title"]))

com_rows = []
for _, row in com.sort_values("human_readable_id").iterrows():
    cid = row["human_readable_id"]
    eids = row["entity_ids"]
    if hasattr(eids, "__len__") and not isinstance(eids, str):
        members = [id_to_title.get(eid, eid[:8]) for eid in eids if eid in id_to_title]
        size = len(eids)
    else:
        members = []
        size = 0
    com_rows.append({
        "community_id": cid,
        "size": size,
        "members": members,
        "level": int(row["level"]),
    })

table_path2 = OUT / "table3_2.md"
with open(table_path2, "w", encoding="utf-8") as f:
    f.write("| Community | Size | Entities |\n")
    f.write("|---:|---:|---|\n")
    for row in com_rows:
        # List ALL entities
        members_str = ", ".join(row["members"])
        f.write(f"| {row['community_id']} | {row['size']} | {members_str} |\n")
print(f"Saved: {table_path2}")

# ── Table 3.3: Representative Relations per Type ───────────────────────────
rel_types = rel["description"].value_counts().index.tolist()

table_path3 = OUT / "table3_3.md"
with open(table_path3, "w", encoding="utf-8") as f:
    f.write("| Relation Type | Entity A | Entity B | Weight | Combined Degree |\n")
    f.write("|---|---|---|---:|---:|\n")
    for rtype in rel_types:
        subset = rel[rel["description"] == rtype]
        top = subset.nlargest(1, "combined_degree").iloc[0]
        title_a = id_to_title.get(top["source"], top["source"][:20])
        title_b = id_to_title.get(top["target"], top["target"][:20])
        f.write(f"| {rtype} | {title_a[:40]} | {title_b[:40]} | {top['weight']:.3f} | {int(top['combined_degree'])} |\n")
print(f"Saved: {table_path3}")

# ── Figure 3.2: Entity-Relation Graph (Community 0 & 5, Top 30 Edges) ──────
try:
    import numpy as np

    # Get entity IDs from community 0 and 5
    target_communities = [0, 5]
    community_entity_ids = set()
    for cid in target_communities:
        row = com[com["human_readable_id"] == cid]
        if len(row) > 0:
            eids = row.iloc[0]["entity_ids"]
            if hasattr(eids, "__len__") and not isinstance(eids, str):
                community_entity_ids.update(eids)
    print(f"Community 0 & 5 entities: {len(community_entity_ids)}")

    # Build title-to-id and id-to-title maps
    title_to_id = dict(zip(ent["title"], ent["id"]))
    id_to_title = dict(zip(ent["id"], ent["title"]))
    title_to_type = dict(zip(ent["title"], ent["type"]))

    # Build edge list filtered to community 0 & 5 entities
    edge_list = []
    for _, row in rel.iterrows():
        src_title = row["source"]
        tgt_title = row["target"]
        src_id = title_to_id.get(src_title)
        tgt_id = title_to_id.get(tgt_title)
        if src_id and tgt_id and src_id in community_entity_ids and tgt_id in community_entity_ids:
            edge_list.append((src_id, tgt_id, src_title, tgt_title, row["description"], row["weight"]))

    # Keep top 30 edges by weight
    edge_list.sort(key=lambda x: x[5], reverse=True)
    edge_list = edge_list[:30]
    print(f"Top 30 edges among community 0 & 5 entities: {len(edge_list)}")

    # node_ids are UUIDs, node_titles are Chinese strings
    nodes = list(set(e[0] for e in edge_list) | set(e[1] for e in edge_list))
    node_titles_map = {e[0]: e[2] for e in edge_list}
    node_titles_map.update({e[1]: e[3] for e in edge_list})
    node_titles = [node_titles_map[n][:20] for n in nodes]
    node_types = [title_to_type.get(node_titles_map[n], "概念") for n in nodes]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Spring layout
    try:
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for s, t, stitle, ttitle, desc, w in edge_list:
            G.add_edge(s, t, description=desc, weight=w)
        pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
    except Exception as e:
        print(f"[WARN] NetworkX layout failed: {e}")
        np.random.seed(42)
        angle = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
        pos = {n: (float(np.cos(a)), float(np.sin(a))) for n, a in zip(nodes, angle)}

    type_colors = {
        "概念": "steelblue", "定理": "coral", "公式": "lightgreen",
        "人物": "gold", "地點": "orange", "組織": "violet",
        "金融機構": "darkgreen", "經濟概念": "darkred",
        "學科領域": "purple", "數學物件": "teal",
    }
    node_colors = [type_colors.get(t, "gray") for t in node_types]

    # Draw edges with relation type labels
    for e in edge_list:
        src_id, tgt_id, src_title, tgt_title, desc, w = e
        if src_id in pos and tgt_id in pos:
            # Draw edge line
            ax.plot([pos[src_id][0], pos[tgt_id][0]], [pos[src_id][1], pos[tgt_id][1]],
                    color="gray", linewidth=min(w * 0.3, 1.5), alpha=0.6, zorder=1)
            # Add relation type label at midpoint
            mid_x = (pos[src_id][0] + pos[tgt_id][0]) / 2
            mid_y = (pos[src_id][1] + pos[tgt_id][1]) / 2
            ax.annotate(desc, (mid_x, mid_y), fontsize=6, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
                        zorder=4)

    # Draw nodes
    x = [pos[n][0] for n in nodes]
    y = [pos[n][1] for n in nodes]
    ax.scatter(x, y, c=node_colors, s=600, zorder=2, edgecolors="black", linewidths=0.8)

    for n, title in zip(nodes, node_titles):
        ax.annotate(title, (pos[n][0], pos[n][1]),
                    fontsize=8, ha="center", va="bottom", zorder=3)

    ax.set_title("L3: GraphRAG Entity-Relation Graph (Community 0 & 5, Top 30 Edges)", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Legend for node types
    unique_types = sorted(set(node_types))
    handles = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=type_colors.get(t, "gray"),
                           label=t, markersize=8) for t in unique_types if t in type_colors]
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=7)

    fig.savefig(OUT / "fig3_2_entity_relation_graph.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT / 'fig3_2_entity_relation_graph.png'}")
    print(f"  nodes={len(nodes)}, edges={len(edge_list)}")
except Exception as e:
    import traceback
    print(f"[WARN] Could not generate entity graph: {e}")
    traceback.print_exc()

print("\nDone.")