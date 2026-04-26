"""
Stage 2a: Embed all files.

Provider selection via EMBED_PROVIDER env var:
  - "local" (default): BGE-M3 via FlagEmbedding (requires torch CUDA)
  - "openai": OpenAI text-embedding-3-small via OPENAI_API_KEY in .env

Output: analysis/embeddings.parquet  (id, domain, title, embedding)
"""

import os
import time
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PROVIDER = os.getenv("EMBED_PROVIDER", "local").lower()
EMBED_OUT = ROOT / "analysis" / "embeddings.parquet"

# ── OpenAI provider ────────────────────────────────────────────────────────────

def run_openai() -> pd.DataFrame:
    from openai import OpenAI, RateLimitError

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    docs = pd.read_csv(ROOT / "processed" / "docs.csv", keep_default_na=False)
    n = len(docs)
    print(f"Embedding {n:,} documents with OpenAI {model}...")

    t0 = monotonic()
    all_vecs = []

    for i in range(0, n, 100):
        batch_texts = []
        for text in docs["text"].iloc[i : i + 100].tolist():
            # Truncate to 3000 chars (~4000 tokens for Chinese, well under 8192 limit)
            if len(text) > 3000:
                text = text[:3000]
            batch_texts.append(text)

        for attempt in range(5):
            try:
                resp = client.embeddings.create(input=batch_texts, model=model)
                break
            except RateLimitError:
                if attempt < 4:
                    wait = min(2 ** attempt + 0.5, 10)
                    print(f"\n  Rate limit, waiting {wait:.1f}s... (attempt {attempt+1}/5)")
                    time.sleep(wait)
                else:
                    raise

        vecs = [np.array(d.embedding, dtype="float32") for d in resp.data]
        all_vecs.extend(vecs)
        pct = min(100, (i + 100) * 100 // n)
        print(f"\r  Progress: {min(i + 100, n):,} / {n:,} ({pct}%)", end="", flush=True)

    elapsed = monotonic() - t0
    print()

    out = pd.DataFrame({
        "id": docs["id"],
        "domain": docs["domain"],
        "title": docs["title"],
        "embedding": [v.tolist() for v in all_vecs],
    })

    (ROOT / "analysis").mkdir(exist_ok=True)
    out.to_parquet(EMBED_OUT)
    print(f"Written {len(out):,} rows to {EMBED_OUT} in {elapsed:.0f}s (~{elapsed/60:.1f} min)")
    print(f"Embedding dimension: {len(out['embedding'].iloc[0])}")

    _sanity_check(out)
    return out

# ── Local BGE-M3 provider ──────────────────────────────────────────────────────

def run_local() -> pd.DataFrame:
    # Apply local environment patches BEFORE any other imports.
    from scripts.local_workaround import apply_patches
    apply_patches()

    import torch
    from FlagEmbedding import BGEM3FlagModel
    from tqdm import tqdm

    if torch.cuda.is_available():
        print(f"GPU confirmed: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available — model running on CPU!")

    docs = pd.read_csv(ROOT / "processed" / "docs.csv", keep_default_na=False)
    n = len(docs)
    print(f"Embedding {n:,} documents with BGE-M3 (FP16, batch=4, max_length=4096)...")

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    warmup_vec = model.encode(["GPU warmup test"], batch_size=1)["dense_vecs"]
    del warmup_vec
    texts = docs["text"].tolist()
    batch_size = 4

    t0 = monotonic()
    all_vecs = []
    for batch in tqdm(
        [texts[i : i + batch_size] for i in range(0, n, batch_size)],
        desc="Embedding",
        unit="batch",
    ):
        all_vecs.extend(model.encode(batch, batch_size=batch_size, max_length=4096)["dense_vecs"])

    elapsed = monotonic() - t0
    vecs = np.array(all_vecs, dtype="float32")

    out = pd.DataFrame({
        "id": docs["id"],
        "domain": docs["domain"],
        "title": docs["title"],
        "embedding": [v.tolist() for v in vecs],
    })

    (ROOT / "analysis").mkdir(exist_ok=True)
    out.to_parquet(EMBED_OUT)
    print(f"\nWritten {len(out):,} rows to {EMBED_OUT} in {elapsed:.0f}s (~{elapsed/60:.1f} min)")
    print(f"Embedding dimension: {len(out['embedding'].iloc[0])}")

    _sanity_check(out)
    return out

# ── Sanity check ───────────────────────────────────────────────────────────────

def _sanity_check(out: pd.DataFrame) -> None:
    vecs_array = np.array(out["embedding"].tolist(), dtype="float32")
    rounded = np.round(vecs_array, 4)
    unique_vecs, inverse_counts = np.unique(rounded, axis=0, return_counts=True)
    dup_mask = inverse_counts > 1
    if dup_mask.any():
        dup_count = int(dup_mask.sum())
        dup_doc_count = int(inverse_counts[dup_mask].sum())
        print(f"\n[SANITY] Embedding collapse: {dup_count} unique vectors across {dup_doc_count} docs")
        dup_vectors = unique_vecs[dup_mask]
        for i, vec in enumerate(dup_vectors):
            mask = np.all(rounded == vec, axis=1)
            dup_ids = out.loc[mask, "id"].tolist()
            print(f"  Group {i+1}: {dup_ids}")
        dup_df = pd.DataFrame({
            "embedding_vector": [str(v) for v in dup_vectors],
            "doc_count": inverse_counts[dup_mask].tolist(),
            "doc_ids": [", ".join(str(x) for x in out.loc[np.all(rounded == v, axis=1), "id"].tolist()) for v in dup_vectors],
        })
        dup_df.to_parquet(ROOT / "analysis" / "embedding_collapse.parquet", index=False)
        print(f"  Written collapse report to analysis/embedding_collapse.parquet")
    else:
        print("\n[SANITY] No embedding collapse detected — all vectors unique.")

# ── Entry point ────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    if PROVIDER == "openai":
        return run_openai()
    return run_local()


if __name__ == "__main__":
    run()