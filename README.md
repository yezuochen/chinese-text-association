# 關聯分析 — Chinese Text Association Analysis Pipeline

A three-layer pipeline that computes concept↔concept and file↔file associations over a corpus of ~9,943 Traditional Chinese Wikipedia-style articles in two domains (數學, 經濟金融).

## What It Produces

| Artifact | Description |
|---|---|
| `processed/docs.csv` | Full corpus, OpenCC-normalized, 9,943 rows |
| `analysis/pmi_graph.parquet` | Concept co-occurrence edges (untyped, PMI≥2.0) |
| `analysis/knn_graph.parquet` | File↔file similarity edges (K=15, cosine) |
| `graphrag_project/output/` | Typed knowledge graph on 100-file sample (entities, relations, communities) |
| `analysis/unified_graph.pkl` | NetworkX graph fusing all layers |

## Architecture

```
data/ (immutable)
    │
    ▼ OpenCC s2tw + scrub
processed/docs.csv
    │
    ├── L1: jieba + PMI ──────────────────────────→ pmi_graph.parquet
    │     (full 10K, free, local CPU)                + concept_domains.parquet
    │
    ├── L2: BGE-M3 embed + FAISS K-NN ────────────→ embeddings.parquet
    │     (full 10K, free, local GPU)                 + knn_graph.parquet
    │
    └── sample_100.py ─────────────────────────────→ processed/sample_100.csv
          (data-driven cross-domain seeds)               │
                                                       ▼
                                            L3: GraphRAG index
                                            (DeepSeek-V3 via NIM +
                                             text-embedding-3-small via OpenAI)
                                                       │
                                        graphrag_project/output/*.parquet
                                                       │
                               ┌────────────────────────┴───────────────────┐
                               ▼                                            ▼
                    analysis/unified_graph.pkl                    reports/eval_sample.md
```

**Three layers:**

- **L1 — PMI concept graph** (full 10K, free, local CPU, untyped)
- **L2 — BGE-M3 K-NN file graph** (full 10K, free, local GPU)
- **L3 — GraphRAG** on 100-file sample (~US$0.05 OpenAI + NIM free)

L1 + L2 provide full-corpus coverage cheaply; L3 provides typed relations + communities on a concept-seeded sample.

## Quick Start

```powershell
# 0. Setup
uv sync
bash setup_cuda_torch.sh        # run after every uv sync

# 1. Preprocess
uv run python src/preprocess.py  # ~1 min

# 2. L2 — full corpus embeddings (local GPU, ~80 min)
#    ⚠️ Run in a new terminal:
uv run python src/embed_files.py
uv run python src/build_knn.py    # <1 min

# 3. L1 — PMI concept graph (local CPU, ~3 min)
uv run python src/build_pmi.py

# 4. L3a — 100-file sample
uv run python src/sample_100.py   # <1 min

# 5. GraphRAG setup (one-off)
uv run graphrag init --root .\graphrag_project
uv run graphrag prompt-tune --root .\graphrag_project `
  --domain "Chinese wiki articles in math and economics" `
  --language "Traditional Chinese" `
  --output .\graphrag_project\prompts
# then manually lock 6-relation taxonomy in prompts/entity_extraction.txt

# 6. L3b — GraphRAG index (paid + slow)
#    ⚠️ Run in a new terminal (~45 min, ~US$0.05):
uv run graphrag index --root .\graphrag_project

# 7. Fuse + evaluate
uv run python src/fuse_graph.py   # ~5 min
uv run python src/eval.py         # <1 min → reports/eval_sample.md
```

## Key Design Decisions

- **Two disjoint embedding spaces.** L2 uses local BGE-M3 (in-process); L3 uses paid OpenAI `text-embedding-3-small`. They are never compared directly.
- **`encoding_format: null` bug sidestepped, not patched.** L3 embeddings route to OpenAI (which accepts `null`) rather than NIM (which rejects it).
- **No preprocessing guardrails.** Stubs and the 3.5 MB outlier pass through — intentional.
- **Folder name is document metadata, not a concept node.** Domain attributes are *computed* by aggregation, not hand-coded.
- **Directional PMI** is computed as a parallel research artifact (forward-window, not fused into the unified graph).

## Documents

- [MEMO.md](MEMO.md) — technique selection rationale
- [PLAN.md](PLAN.md) — build plan and module specs
- [DESIGN.md](DESIGN.md) — detailed algorithm design
- [CUDA_TORCH_UV_NOTE.md](CUDA_TORCH_UV_NOTE.md) — PyTorch CUDA installation note

## Budget

| Operation | Cost |
|---|---|
| L1 + L2 (full 10K) | Free (local GPU + CPU) |
| L3 GraphRAG embeddings | ~US$0.05 (100 files, OpenAI `text-embedding-3-small`) |
| L3 chat (DeepSeek-V3 via NIM) | Free (40 RPM, no charge) |