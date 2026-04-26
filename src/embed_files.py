"""
Stage 2a: Embed all files with BGE-M3 via FlagEmbedding (local, no server).

Uses the same BGE-M3 model that would run on an infinity server.
Output: analysis/embeddings.parquet  (id, domain, title, embedding)
"""

# Apply local environment patches BEFORE any other imports.
# See scripts/local_workaround.py for details on what/why.
from scripts.local_workaround import apply_patches
apply_patches()

from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import torch  # noqa: E402  imported after all patches

from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
EMBED_OUT = ROOT / "analysis" / "embeddings.parquet"


def run() -> pd.DataFrame:
    docs = pd.read_csv(ROOT / "processed" / "docs.csv", keep_default_na=False)
    n = len(docs)
    print(f"Embedding {n:,} documents with BGE-M3 (FP16, batch=4, max_length=4096)...")

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    # Warmup: encode one short text to force GPU initialization
    warmup_vec = model.encode(["GPU warmup test"], batch_size=1)["dense_vecs"]
    if torch.cuda.is_available():
        print(f"GPU confirmed: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available — model running on CPU!")
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
    return out


if __name__ == "__main__":
    run()
