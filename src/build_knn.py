"""
Stage 2b: Build K-NN graph over full-corpus BGE-M3 embeddings via FAISS.

Input:  analysis/embeddings.parquet  (id, domain, title, embedding)
Output: analysis/knn_graph.parquet   (src, dst, cos_sim)
"""

import faiss
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBED_IN = ROOT / "analysis" / "embeddings.parquet"
KNN_OUT = ROOT / "analysis" / "knn_graph.parquet"


def run() -> pd.DataFrame:
    emb = pd.read_parquet(EMBED_IN)
    print(f"Loaded {len(emb):,} embeddings from {EMBED_IN}")

    X = np.stack(emb["embedding"].apply(np.asarray).values).astype("float32")
    faiss.normalize_L2(X)  # cosine similarity via inner product

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    k = 16  # +1 for self
    D, I = index.search(X, k=k)

    edges = []
    for s, (dsts, sims) in enumerate(zip(I, D)):
        src_id = emb["id"].iloc[s]
        for d, sim in zip(dsts[1:], sims[1:]):  # skip self
            edges.append((src_id, emb["id"].iloc[d], float(sim)))

    df = pd.DataFrame(edges, columns=["src", "dst", "cos_sim"])
    (ROOT / "analysis").mkdir(exist_ok=True)
    df.to_parquet(KNN_OUT)
    print(f"Written {len(df):,} edges ({k-1}-NN per doc) to {KNN_OUT}")
    return df


if __name__ == "__main__":
    run()
