"""
Report L2: K-NN Document Graph — Distribution, Examples & Clustering
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
knn = pd.read_parquet(ROOT / "analysis" / "knn_graph.parquet")
docs = pd.read_csv(ROOT / "processed" / "docs.csv", keep_default_na=False, dtype={"id": str})
emb = pd.read_parquet(ROOT / "analysis" / "embeddings.parquet")

docs["id_int"] = docs["id"].astype(int)
knn["src_int"] = knn["src"].astype(int)
knn["dst_int"] = knn["dst"].astype(int)

# Filter self-loops
knn = knn[knn["src"] != knn["dst"]].copy()

# ── Stats ───────────────────────────────────────────────────────────────────
total_edges = len(knn)
cos_min = knn["cos_sim"].min()
cos_max = knn["cos_sim"].max()
cos_mean = knn["cos_sim"].mean()
cos_std = knn["cos_sim"].std()
cos_median = knn["cos_sim"].median()

print(f"Total edges: {total_edges:,}")
print(f"Cos sim range: {cos_min:.4f} – {cos_max:.4f}")
print(f"Mean: {cos_mean:.4f}, Std: {cos_std:.4f}, Median: {cos_median:.4f}")

# ── Figure 2.1: Density plot only ──────────────────────────────────────────
try:
    import numpy as np
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(cos_min, cos_max, 300)
    density = gaussian_kde(knn["cos_sim"].values)
    ax.fill_between(x, density(x), alpha=0.5, color="steelblue")
    ax.plot(x, density(x), color="steelblue", linewidth=1.5)
    ax.axvline(cos_mean, linestyle="--", color="red", alpha=0.8, label=f"Mean={cos_mean:.3f}")
    ax.axvline(cos_median, linestyle=":", color="darkorange", alpha=0.8, label=f"Median={cos_median:.3f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("K-NN Cosine Similarity Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_1.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT / 'fig2_1.png'}")
except Exception as e:
    print(f"[WARN] Could not generate density plot: {e}")

# ── Table 2.1: Three-level K-NN examples with proper thresholds ─────────────
doc_info = docs.drop_duplicates("id_int").set_index("id_int")[["title", "domain", "text"]].to_dict("index")

def make_summary(text, max_chars=120):
    s = text.replace("\n", " ").replace("=", " ").replace("|", " ").replace("{", " ").replace("}", " ")
    s = " ".join(s.split())
    return s[:max_chars] + "..." if len(s) > max_chars else s

def is_meaningful(title, text):
    """Reject pure template/boilerplate pages."""
    t_lower = title.lower() if isinstance(title, str) else ""
    if t_lower in ("userbox", "userbox 2", "infobox_officeholder", "infobox writer",
                   "infobox person", "infobox scientist", "noteta"):
        return False
    if t_lower.startswith("infobox_"):
        return False
    if not isinstance(text, str):
        return False
    if "本分類收錄與" in text or "如果要將條目加入本分類" in text:
        return False
    template_chars = sum(1 for c in text if c in "{}|=")
    if len(text) > 0 and template_chars / len(text) > 0.15:
        return False
    if len(text.strip()) < 200:
        return False
    zh_count = text.count(" zh ")
    if len(text) > 0 and zh_count / len(text) > 0.004:
        return False
    if "無綫劇集" in title or "談情說案" in title or "純熟意外" in title:
        return False
    return True

def get_doc_info(doc_id):
    info = doc_info.get(doc_id, {})
    return info.get("title", ""), info.get("text", "")

# Thresholds: High > 0.9, Average 0.6–0.8, Low < 0.5
# Strategy: find cross-domain pairs first (more meaningful), then same-domain
import random
random.seed(99)

examples = []

# High: cos > 0.9 — prefer same-domain (both math or both econ)
high_pool = knn[knn["cos_sim"] > 0.9].copy()
high_pool = high_pool.merge(docs[["id_int","domain"]], left_on="src_int", right_on="id_int").rename(columns={"domain":"dom_src"}).drop("id_int", axis=1)
high_pool = high_pool.merge(docs[["id_int","domain"]], left_on="dst_int", right_on="id_int").rename(columns={"domain":"dom_dst"}).drop("id_int", axis=1)
# Same domain first, then cross
pools = [high_pool[high_pool["dom_src"] == high_pool["dom_dst"]], high_pool[high_pool["dom_src"] != high_pool["dom_dst"]]]
for pool in pools:
    for _, row in pool.iterrows():
        t1, tx1 = get_doc_info(row["src_int"])
        t2, tx2 = get_doc_info(row["dst_int"])
        if is_meaningful(t1, tx1) and is_meaningful(t2, tx2):
            examples.append(("High", row, t1, t2, tx1, tx2))
            break
    if len(examples) and examples[-1][0] == "High":
        break

# Average: 0.6 <= cos <= 0.8 — prefer same-domain (math-math or econ-econ)
avg_pool = knn[(knn["cos_sim"] >= 0.6) & (knn["cos_sim"] <= 0.8)].copy()
avg_pool = avg_pool.merge(docs[["id_int","domain"]], left_on="src_int", right_on="id_int").rename(columns={"domain":"dom_src"}).drop("id_int", axis=1)
avg_pool = avg_pool.merge(docs[["id_int","domain"]], left_on="dst_int", right_on="id_int").rename(columns={"domain":"dom_dst"}).drop("id_int", axis=1)
pools = [avg_pool[avg_pool["dom_src"] == avg_pool["dom_dst"]], avg_pool[avg_pool["dom_src"] != avg_pool["dom_dst"]]]
for pool in pools:
    for _, row in pool.iterrows():
        t1, tx1 = get_doc_info(row["src_int"])
        t2, tx2 = get_doc_info(row["dst_int"])
        if is_meaningful(t1, tx1) and is_meaningful(t2, tx2):
            examples.append(("Average", row, t1, t2, tx1, tx2))
            break
    if len(examples) > 1 and examples[-1][0] == "Average":
        break

# Low: cos < 0.5 — prefer cross-domain
low_pool = knn[knn["cos_sim"] < 0.5].copy()
low_pool = low_pool.merge(docs[["id_int","domain"]], left_on="src_int", right_on="id_int").rename(columns={"domain":"dom_src"}).drop("id_int", axis=1)
low_pool = low_pool.merge(docs[["id_int","domain"]], left_on="dst_int", right_on="id_int").rename(columns={"domain":"dom_dst"}).drop("id_int", axis=1)
pools = [low_pool[low_pool["dom_src"] != low_pool["dom_dst"]], low_pool[low_pool["dom_src"] == low_pool["dom_dst"]]]
for pool in pools:
    for _, row in pool.iterrows():
        t1, tx1 = get_doc_info(row["src_int"])
        t2, tx2 = get_doc_info(row["dst_int"])
        if is_meaningful(t1, tx1) and is_meaningful(t2, tx2):
            examples.append(("Low", row, t1, t2, tx1, tx2))
            break
    if len(examples) > 2 and examples[-1][0] == "Low":
        break

print(f"\nCollected {len(examples)} examples: {[e[0] for e in examples]}")

# Write table
table_path = OUT / "table2_1.md"
with open(table_path, "w", encoding="utf-8") as f:
    f.write("| Level | Doc1 Name | Doc2 Name | Doc1 Title | Doc2 Title | Doc1 Summary | Doc2 Summary | Doc1 Domain | Doc2 Domain | Cos Sim | Cos Sim (%) |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---:|\n")
    for label, row, t1, t2, tx1, tx2 in examples:
        d1 = f"{row['src_int']}.txt"
        d2 = f"{row['dst_int']}.txt"
        cos_pct = f"{row['cos_sim']*100:.2f}"
        f.write(f"| {label} | {d1} | {d2} | "
                f"{t1[:60]} | {t2[:60]} | "
                f"{make_summary(tx1)[:120]} | {make_summary(tx2)[:120]} | "
                f"{row['dom_src']} | {row['dom_dst']} | {row['cos_sim']:.4f} | {cos_pct} |\n")
print(f"Saved: {table_path}")

# ── Figure 2.2: K-NN Clustering with labels and xlim ───────────────────────
try:
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    emb_df = emb.copy()
    emb_df["doc_id"] = emb_df["id"].astype(int)
    docs_emb = emb_df.drop_duplicates("doc_id").merge(docs[["id_int"]], left_on="doc_id", right_on="id_int", how="inner")
    mat = np.vstack(docs_emb["embedding"].values)
    doc_titles = docs_emb["title"].values
    doc_domains = docs_emb["domain"].values

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    mat_2d = tsne.fit_transform(mat)

    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(mat)
    cluster_colors = plt.cm.tab10.colors[:n_clusters]

    # Assign labels to clusters: find the most frequent domain in each cluster
    cluster_labels = {}
    for cid in range(n_clusters):
        mask = labels == cid
        doms_in_cluster = doc_domains[mask]
        # Count domains
        math_count = sum(1 for d in doms_in_cluster if d == "數學")
        econ_count = sum(1 for d in doms_in_cluster if d == "經濟金融")
        titles_in_cluster = doc_titles[mask]
        # Get mode domain
        mode_dom = "數學" if math_count >= econ_count else "經濟金融"
        count = max(math_count, econ_count)
        cluster_labels[cid] = f"Cluster {cid} ({mode_dom}, n={count})"

    fig, ax = plt.subplots(figsize=(8, 6))
    for cid in range(n_clusters):
        mask = labels == cid
        ax.scatter(mat_2d[mask, 0], mat_2d[mask, 1],
                   c=[cluster_colors[cid]], label=cluster_labels[cid], alpha=0.4, s=8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Document Embeddings — t-SNE + K-Means (K={n_clusters})", fontsize=11, fontweight="bold")
    ax.legend(markerscale=4, fontsize=8, loc="upper right")

    fig.savefig(OUT / "fig2_2.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT / 'fig2_2.png'}")
    print("Cluster labels:", cluster_labels)
except Exception as e:
    print(f"[WARN] Could not generate clustering plot: {e}")

print("\nDone.")