"""
Stage 2a: Embed all files with BGE-M3 via FlagEmbedding (local, no server).

Uses the same BGE-M3 model that would run on an infinity server.
Output: analysis/embeddings.parquet  (id, domain, title, embedding)
"""

# MUST be first — clear any previously-cached broken transformers/accelerate
# modules (they crashed at import time with None torch version and are now
# permanently broken in sys.modules). Removing them lets Python re-import
# fresh copies.
import sys

for mod in list(sys.modules.keys()):
    if "transformers" in mod or "accelerate" in mod:
        del sys.modules[mod]

# Patch importlib.metadata.version FIRST, before any other imports.
# On this system, importlib.metadata.version('torch') returns None because
# torch's RECORD file is broken (OneDrive + Python 3.11 interaction).
# transformers and accelerate call version.parse(version("torch")) internally,
# which crashes on None unless we handle it.
import importlib.metadata

_dists_map = {d.name: d for d in importlib.metadata.distributions()}
_orig_version = importlib.metadata.version


def _patched_version(name: str):
    if name in _dists_map:
        return _dists_map[name].version
    return _orig_version(name)


importlib.metadata.version = _patched_version

# Also patch packaging.version.parse to handle None gracefully.
# This is the second crash site: version.parse(None) raises TypeError.
import packaging.version

_orig_parse = packaging.version.parse


def _patched_parse(version_string):
    if version_string is None:
        return packaging.version.Version("0.0.0")
    return _orig_parse(version_string)


packaging.version.parse = _patched_parse

# Patch check_torch_load_is_safe to not raise — we use weights_only=True
# but transformers 4.57 requires torch>=2.6 for torch.load safety.
# We pass weights_only=True and don't load any untrusted checkpoints,
# so this check is unnecessary for our use case.
# Note: modeling_utils does `from .utils.import_utils import check_torch_load_is_safe`
# which creates a local binding, so we must patch it in both modules.
import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

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
