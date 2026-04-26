# Plan — Build the 關聯分析 Pipeline (Three-Layer Architecture)

## Context

Technique-selection memo: [MEMO.md](MEMO.md). Stack is fixed there:

- **Three-layer architecture:**
  - **L1 — PMI concept graph** over the full 10K corpus (free, local CPU, untyped edges)
  - **L2 — BGE-M3 K-NN file graph + traditional RAG** over the full 10K (free, local GPU)
  - **L3 — GraphRAG demo** on a concept-seeded 100-file sample (~US$0.05 OpenAI + NIM free)
- **LLM (L3 only):** DeepSeek-V3 on NVIDIA NIM (free, 40 RPM), OpenAI-compatible endpoint
- **GraphRAG internal embeddings (L3 only):** paid OpenAI `text-embedding-3-small`
- **Our K-NN embeddings (L2 only):** local BGE-M3 FP16 via `FlagEmbedding`, in-process (no server)
- **Chinese variant:** normalize to **Traditional (Taiwan standard)** via OpenCC `s2tw`
- **No preprocessing guardrails** (stubs + 3.5 MB outlier pass through)
- **Folder name** → document `domain` metadata; entity / community domain attributes are *computed*, not hand-coded
- **Preprocessing output:** `processed/docs.csv` (and a derived `processed/sample_100.csv` for L3)
- **Package management:** `uv`

This plan describes **what code to write, in what order, to run the full pipeline end-to-end** on Windows 11 with an RTX 3050 (6 GB), an NVIDIA NIM API key, and an OpenAI API key.

### End-state deliverables (runnable)

1. `processed/docs.csv` — cleaned corpus, 9,943 rows (L0 output)
2. `processed/sample_100.csv` — concept-seeded stratified sample for L3
3. `analysis/embeddings.parquet` — BGE-M3 file-level embeddings (L2)
4. `analysis/knn_graph.parquet` — file ↔ file K-NN, K = 15 (L2)
5. `analysis/pmi_graph.parquet` — concept ↔ concept PMI edges (L1)
6. `graphrag_project/output/*.parquet` — entities, typed relationships, communities, reports (L3)
7. `analysis/unified_graph.pkl` — NetworkX graph fusing L1 + L2 + L3
8. `src/concept_sim.py`, `src/file_sim.py` — query APIs
9. `reports/eval_sample.md` — 35-edge qualitative eval sheet
10. Working `uv run graphrag query --root .\graphrag_project --method local  "..."` on the 100-file sample

---

## Project Layout

```
hw/
├── data/                           # immutable input (數學/, 經濟金融/)
├── processed/
│   ├── docs.csv                    # L0 output (full 10K)
│   └── sample_100.csv              # L3 input (concept-seeded sample)
├── graphrag_project/
│   ├── settings.yaml               # GraphRAG config (NIM for chat, OpenAI for embed)
│   ├── prompts/                    # tuned Traditional Chinese prompts
│   └── output/                     # GraphRAG artifacts (parquet)
├── analysis/
│   ├── embeddings.parquet          # L2
│   ├── knn_graph.parquet           # L2
│   ├── pmi_graph.parquet           # L1
│   ├── concept_domains.parquet      # L1 (concept domain distributions)
│   ├── directional_pmi.parquet      # L1b (directional PMI, parallel artifact)
│   └── unified_graph.pkl           # Fusion
├── src/
│   ├── preprocess.py               # 10K → docs.csv
│   ├── sample_100.py               # docs.csv → sample_100.csv
│   ├── embed_files.py              # BGE-M3 on full 10K (local, no server)
│   ├── build_knn.py                # FAISS K-NN over 10K
│   ├── build_pmi.py                # jieba + PMI over 10K
│   ├── build_directional_pmi.py   # forward-window directional PMI (parallel L1)
│   ├── fuse_graph.py               # L1 + L2 + L3 → unified_graph.pkl
│   ├── concept_sim.py              # query API
│   ├── file_sim.py                 # query API
│   └── eval.py                     # 35-edge sample → reports/eval_sample.md
├── reports/
│   └── eval_sample.md
├── pyproject.toml                  # uv-managed
├── uv.lock
├── .env.example
└── README.md
```

**What was deleted vs rev. 2:**

- `scripts/start_embed_server.ps1` — infinity server no longer used
- `src/embed_server.py` — local OpenAI-compatible server wrapper, not needed (BGE-M3 runs in-process)
- `src/embed_proxy.py` — NIM-proxy shim that stripped `encoding_format: null`; moot because L3 embeddings route to OpenAI directly
- `processed/docs_smoke.csv` — artifact of the retired "smoke test on 100-file subset" step. The 100-file sample is now the canonical L3 run.
- Separate "GraphRAG smoke test" step — retired (see above)

---

## Environment Setup (uv)

**Install uv once:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Sync dependencies:**

```powershell
uv sync
```

**Add new deps (one-off if editing `pyproject.toml` by hand is undesired):**

```powershell
# Already in pyproject.toml: flagembedding, graphrag, jieba, opencc-python-reimplemented, pandas, pyarrow, python-dotenv
uv add faiss-cpu networkx python-igraph leidenalg openai httpx
uv add torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1 wheel for RTX 3050
```

**Run commands via uv:**

```powershell
uv run python src/preprocess.py
uv run graphrag index --root .\graphrag_project
```

**`.env.example`:**

```
# L3 LLM — chat via NIM free tier
NIM_API_KEY=nvapi-...
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_MODEL=deepseek-ai/deepseek-v3

# L3 embeddings — OpenAI paid (~US$0.05 for 100 files)
OPENAI_API_KEY=sk-...
OPENAI_EMBED_MODEL=text-embedding-3-small
```

No `EMBED_SERVER_URL`, no `EMBED_MODEL` env var — L2's BGE-M3 is hard-coded in `src/embed_files.py` since it runs in-process.

---

## Module-by-Module Build

### L0. `src/preprocess.py` — shared for all layers

**In:** `data/**/*.txt`
**Out:** `processed/docs.csv` (columns: `id, domain, title, text, char_count`)

Steps per file:

1. Read UTF-8
2. **OpenCC convert Simplified → Traditional (Taiwan std)** with `s2tw` — normalizes the minority Simplified content
3. Regex scrub:
   - Strip HTML / wiki-template lines (`^border \d+ cellpadding.*`, CSS tokens, `style .*`)
   - Strip broken LaTeX (`\\\w+`, stray braces)
   - Collapse whitespace
4. Pick title (first non-empty non-template line; fall back to `id`)
5. Emit row; `domain` = parent folder name (`數學` | `經濟金融`)

```python
import opencc
cc = opencc.OpenCC('s2tw')
text = cc.convert(raw)
```

Write with `pandas.DataFrame.to_csv(..., quoting=csv.QUOTE_ALL, encoding='utf-8-sig')` so Excel and GraphRAG both parse it cleanly.

### L2a. `src/embed_files.py` — full-corpus BGE-M3 embeddings

> ⚠️ **Do NOT run this from Claude Code.** This takes ~80 min on GPU. Announce the command and instruct the user to run it in a new terminal.

Local, in-process, no server. Same model that would run under `infinity`, but called directly via `FlagEmbedding`.

```python
from FlagEmbedding import BGEM3FlagModel
import pandas as pd, numpy as np

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
docs = pd.read_csv('processed/docs.csv', keep_default_na=False)
vecs = model.encode(docs['text'].tolist(),
                    batch_size=4, max_length=8192)['dense_vecs']
pd.DataFrame({
    'id': docs['id'],
    'domain': docs['domain'],
    'title': docs['title'],
    'embedding': [v.tolist() for v in np.array(vecs, dtype='float32')],
}).to_parquet('analysis/embeddings.parquet')
```

**VRAM note:** BGE-M3 FP16 + batch 4 + `max_length=8192` ≈ 4.5 GB peak. Drop `max_length` to 4096 if OOM. Expected throughput on RTX 3050: ~2 docs/s → ~80 min for 10K files.

### L2b. `src/build_knn.py` — FAISS K-NN

```python
import faiss, numpy as np, pandas as pd

emb = pd.read_parquet('analysis/embeddings.parquet')
X = np.stack(emb['embedding'].apply(np.asarray).values).astype('float32')
faiss.normalize_L2(X)                        # cosine via IP
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
D, I = index.search(X, k=16)                 # +1 for self

edges = [(emb['id'].iloc[s], emb['id'].iloc[d], float(sim))
         for s, (dsts, sims) in enumerate(zip(I, D))
         for d, sim in zip(dsts[1:], sims[1:])]
pd.DataFrame(edges, columns=['src', 'dst', 'cos_sim']).to_parquet(
    'analysis/knn_graph.parquet')
```

### L1. `src/build_pmi.py` — PMI concept graph

**In:** `processed/docs.csv`
**Out:** `analysis/pmi_graph.parquet` + `analysis/concept_domains.parquet`

Steps:

1. Load `docs.csv`.
2. Tokenize each doc with `jieba.cut`; keep `len(tok) ≥ 2`; drop stopwords.
3. Treat each doc as a bag of unique tokens; track per-token domain counts.
4. Build `doc_freq[t]` and `pair_freq[(t₁, t₂)]` for pairs where both have `doc_freq ≥ 5`.
5. `PMI(t₁, t₂) = log(p(t₁, t₂) / (p(t₁) · p(t₂)))` with probabilities over documents.
6. Keep edges with `PMI ≥ 2.0` and `pair_freq ≥ 3`.
7. Write `pmi_graph.parquet` (src, dst, pmi, co_doc_count).
8. Write `concept_domains.parquet` (concept, doc_freq, docs_數學, docs_經濟金融, cross_domain_count).

### L1b. `src/build_directional_pmi.py` — forward-window directional PMI (parallel L1)

**In:** `processed/docs.csv`
**Out:** `analysis/directional_pmi.parquet`

Steps:

1. Load `processed/docs.csv`.
2. Tokenize each doc with `jieba.cut`; keep `len(tok) ≥ 2`; drop stopwords (same filter as L1).
3. For each doc, track token **positions** (ordered list, not bag).
4. Sliding forward window (window_size=50 tokens): for each occurrence of t1 at position i, count t2 occurrences at positions j where i < j ≤ i + window_size.
5. Build `forward_count[t1→t2]` (directed pair count) across all docs.
6. Compute `PMI(t1→t2) = log( forward_count[t1→t2] / doc_freq[t1] / (doc_freq[t2] / total_docs) )` — conditional probability of t2 given t1 divided by marginal.
7. Keep directed edges with `forward_count >= 3` and `PMI(t1→t2) >= 1.5` (lower threshold than symmetric L1 since directional signal is noisier).
8. Write `directional_pmi.parquet` with columns: `src`, `dst`, `pmi_forward` (A→B), `pmi_reverse` (B→A), `co_doc_count`, `asymmetry = PMI(A→B) - PMI(B→A)`.

This is a **research artifact only** — not fused into the unified graph, not exposed via query APIs.

### L3a. `src/sample_100.py` — data-driven cross-domain seeded sample

**In:** `processed/docs.csv`, `analysis/pmi_graph.parquet`, `analysis/concept_domains.parquet`
**Out:** `processed/sample_100.csv` (same schema as `docs.csv`)

Algorithm:

1. Load `concept_domains.parquet` + `pmi_graph.parquet`.
2. Define 10 math-specific anchors (函數, 定理, 證明, 方程式, 代數, 幾何, 導數, 積分, 機率, 統計) and 10 econ-specific anchors (貨幣, 利率, 投資, 風險, 價格, 市場, 交易, 銀行, 貿易, 通貨膨脹).
3. For each anchor, collect cross-domain PMI neighbors (co-occur in ≥2 domains, ≥5 docs/domain). Accumulate `anchor_PMI_sum` per neighbor.
4. Rank neighbors by `anchor_PMI_sum` (primary) × `pmi_degree` (secondary).
5. Top-8 → hard seeds; find at least one 數學 and one 經濟金融 file per seed.
6. Stratified fill to 50/50 per domain, weighted ∝ char_count^0.5.
7. Hard cap 100 rows.

### L3b. `graphrag_project/` — GraphRAG on the 100-file sample

**One-off init (if not already initialized):**

```powershell
uv run graphrag init --root .\graphrag_project
```

**Edit `settings.yaml`** (key sections):

```yaml
input:
  type: file
  file_type: csv
  base_dir: ../processed
  file_pattern: "sample_100.csv"        # 100-file sample, not full docs.csv
  source_column: text
  title_column: title
  metadata: [id, domain]

llm:                                     # chat → DeepSeek-V3 via NIM (free)
  api_key: ${NIM_API_KEY}
  type: openai_chat
  model: deepseek-ai/deepseek-v3
  api_base: ${NIM_BASE_URL}
  requests_per_minute: 35                # under NIM's 40 ceiling
  max_retries: 10                        # exponential backoff on 429/5xx
  concurrent_requests: 4

embeddings:                              # GraphRAG internal → OpenAI paid
  llm:
    api_key: ${OPENAI_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small
    # default api_base (OpenAI) — no NIM, no infinity, no proxy

entity_extraction:
  prompt: prompts/entity_extraction.txt
  entity_types: [概念, 定理, 公式, 人物, 地點, 組織]
  max_gleanings: 1

summarize_descriptions:
  prompt: prompts/summarize_descriptions.txt
  max_length: 300

community_report:
  prompt: prompts/community_report.txt
  max_length: 2000
```

**Why OpenAI for embeddings.** `fnllm` serializes `encoding_format: null`; NIM's strict validator rejects it. OpenAI accepts it. Cost for 100 files × ~3 chunks × ~2 embed calls = ~US$0.05. See MEMO §5.

**Rate-limit safety.** GraphRAG's LLM worker uses token-bucket throttling + `tenacity` backoff. `requests_per_minute: 35` + `max_retries: 10` absorbs NIM 429 bursts.

**Tune prompts for Chinese (one-off, ~5 min):**

```powershell
uv run graphrag prompt-tune --root .\graphrag_project `
  --domain "Chinese wiki articles in math and economics" `
  --language "Traditional Chinese" `
  --output .\graphrag_project\prompts
```

Then **manually edit** `prompts/entity_extraction.txt` to constrain relations to the fixed set: `is-a, part-of, defines, special-case-of, used-in, related-to`.

**Run index (~45 min on 100 files at 35 RPM):**

> ⚠️ **Do NOT run this from Claude Code.** This takes ~45 min and costs ~US$0.05 on OpenAI embeddings. Announce the command and instruct the user to run it in a new terminal.

```powershell
uv run graphrag index --root .\graphrag_project
```

No separate smoke-test step — this run *is* the smoke test. If outputs look wrong (English entities, wrong relation types), adjust `prompts/entity_extraction.txt` and re-run.

**Query (demo):**

```powershell
uv run graphrag query --root .\graphrag_project --method local  "什麼是期望值?"
uv run graphrag query --root .\graphrag_project --method global "數學與經濟金融的共通概念?"
```

**Outputs under `graphrag_project/output/`:**

- `create_final_entities.parquet` — entities + descriptions + embeddings
- `create_final_relationships.parquet` — typed edges
- `create_final_communities.parquet` — community membership
- `create_final_community_reports.parquet` — LLM summaries
- `create_final_text_units.parquet` — chunk-level with `document_ids` (links entities ↔ files)

### Fusion. `src/fuse_graph.py`

Merges L1 + L2 + L3 into one `networkx.Graph` → `analysis/unified_graph.pkl`.

Algorithm:

1. Load `create_final_entities`, `create_final_relationships`, `create_final_communities`, `create_final_text_units` (L3).
2. Load `knn_graph.parquet` (L2), `pmi_graph.parquet` (L1).
3. Build `doc_to_communities`: for each L3-covered file id, the set of community ids touched by entities appearing in its text units. Files outside the 100-sample get an empty set.
4. **File nodes** (all 9,943): `type='file'`, `domain`.
5. **Concept nodes:**
   - L3 entities: `type='concept'`, `source='graphrag'`, `entity_type`, `description`.
   - L1 PMI nodes not in L3: `type='concept'`, `source='pmi'`, `domain_distribution`.
6. **Concept ↔ concept edges:**
   - L3 typed: `relation ∈ {is-a, part-of, defines, special-case-of, used-in, related-to}`, `source='graphrag'`.
   - L1 untyped: `pmi`, `co_doc_count`, `source='pmi'`.
7. **File ↔ file edges (L2):**
   `community_overlap = Jaccard(doc_to_communities[src], doc_to_communities[dst])`
   `weight = 0.7 · cos_sim + 0.3 · community_overlap`
8. **Concept ↔ file edges (L3):** attach each L3 concept to the files it appeared in.
9. Pickle to `analysis/unified_graph.pkl`.

### `src/concept_sim.py` and `src/file_sim.py`

```python
# concept_sim.py
def concept_similarity(c1, c2, G, entity_emb) -> dict:
    cos = cosine(entity_emb[c1], entity_emb[c2])  # L3 entity embeddings
    try:
        path_len = nx.shortest_path_length(G, c1, c2)
    except nx.NetworkXNoPath:
        path_len = None
    jacc = jaccard(set(G.neighbors(c1)), set(G.neighbors(c2)))
    composite = 0.5*cos + 0.3*(0 if path_len is None else 1/path_len) + 0.2*jacc
    return {'cosine': cos, 'path_len': path_len,
            'neighbor_jaccard': jacc, 'composite': composite}
```

If one concept is outside L3 (PMI-only), `entity_emb` lookup fails → fall back to `path_len + jacc` using PMI-weighted edges.

```python
# file_sim.py
def similar_files(file_id, G, k=10):
    neighbors = [(n, G[file_id][n]['weight'])
                 for n in G.neighbors(file_id) if G.nodes[n]['type'] == 'file']
    return sorted(neighbors, key=lambda x: -x[1])[:k]
```

### `src/eval.py`

Samples 35 edges for manual inspection into `reports/eval_sample.md`:

- 10 random L3 typed concept–concept edges (head · relation · tail · source-file snippet)
- 10 random L2 K-NN file pairs (titles + cos_sim + shared communities if any)
- 10 **cross-domain bridges** (K-NN edges where `src.domain ≠ dst.domain`)
- 5 random L1 PMI edges (for sanity)

---

## Execution Order

> ⚠️ **Long-running / paid steps are NOT auto-executed by Claude Code.** Steps 2 and 6 below (`embed_files.py` and `graphrag index`) must be run by the user in a new terminal. Claude Code will announce which command to run but will not execute it itself.

```
0. uv sync                                                 # env ready
   ⚠️ then: bash setup_cuda_torch.sh                      # install CUDA torch (run after every uv sync)
1. uv run python src/preprocess.py                         # ~1 min → docs.csv
2. uv run python src/embed_files.py                        # ⚠️ ~80 min on GPU → embeddings.parquet
                                                             Run in new terminal
3. uv run python src/build_knn.py                          # <1 min → knn_graph.parquet
4. uv run python src/build_pmi.py                          # ~3 min → pmi_graph.parquet
4b. uv run python src/build_directional_pmi.py             # ~3 min → directional_pmi.parquet
5. uv run python src/sample_100.py                         # <1 min → sample_100.csv
6. uv run graphrag init --root .\graphrag_project          # one-off
7. uv run graphrag prompt-tune ...                         # one-off, ~5 min
8. (manually lock relation taxonomy in prompts/entity_extraction.txt)
9. uv run graphrag index --root .\graphrag_project         # ⚠️ ~45 min, ~US$0.05 OpenAI embed
                                                             Run in new terminal
10. uv run python src/fuse_graph.py                        # ~5 min
11. uv run python src/eval.py                              # <1 min
```

Steps 1–5 are free (local GPU + CPU); step 9 consumes NIM free RPM + ~US$0.05 of OpenAI embed credits. No long-running background servers — nothing to start, nothing to tear down.

---

## Critical Files to Create / Modify

- `src/preprocess.py`
- `src/sample_100.py` *(new)*
- `src/embed_files.py` (already exists; fix the minor `ROOT / "analysis".mkdir(...)` operator-precedence bug if not already fixed)
- `src/build_knn.py` *(new)*
- `src/build_pmi.py` *(new)*
- `src/build_directional_pmi.py` *(new)*
- `src/fuse_graph.py` *(new)*
- `src/concept_sim.py` *(new)*
- `src/file_sim.py` *(new)*
- `src/eval.py` *(new)*
- `graphrag_project/settings.yaml`
- `graphrag_project/prompts/entity_extraction.txt` (manually edited for fixed relation schema)
- `pyproject.toml` (add `faiss-cpu`, `networkx`, `python-igraph`, `leidenalg`, `openai`, `httpx`, `torch`)
- `.env.example` *(new — drop `EMBED_SERVER_URL` + `EMBED_MODEL`; add `OPENAI_API_KEY`)*

**To delete:**

- `src/embed_server.py`
- `src/embed_proxy.py`
- `scripts/start_embed_server.ps1`
- `scripts/` folder if empty after deletion
- `processed/docs_smoke.csv`

---

## Verification

Smoke tests at each stage:

1. **Preprocessing (L0):** `processed/docs.csv` has **9,943** rows; `domain.value_counts() == {數學: 2523, 經濟金融: 7420}`; no NaN in `text`. Spot-check: any file that had simplified characters should now be fully Traditional.
2. **L2 embeddings:** `analysis/embeddings.parquet` has 9,943 rows; each `embedding` is length 1024.
3. **K-NN sanity:** `similar_files('100410')` (延森不等式) — top-K are other inequality / probability articles from 數學.
4. **L1 PMI sanity:** top-PMI neighbors of `期望值` include 隨機變數, 變異數, 機率分布, 條件期望.
4b. **Directional PMI sanity:** `directional_pmi.parquet` has asymmetric entries — at least some pairs where `PMI(A→B) ≠ PMI(B→A)`; top forward-PMI pairs differ from top reverse-PMI pairs; fewer edges than `pmi_graph.parquet` (more selective due to window constraint).
5. **Sample (L3a):** `processed/sample_100.csv` has 100 rows; contains at least one file for each of the 5 hard seeds; domain split is ~50/50.
6. **L3 GraphRAG run:** `create_final_entities.parquet` non-empty; entity names are Traditional Chinese; `create_final_relationships.parquet.relation` ⊂ `{is-a, part-of, defines, special-case-of, used-in, related-to}`.
7. **L3 query:** `uv run graphrag query --root .\graphrag_project --method local "什麼是期望值?"` returns a grounded Chinese answer citing files from `sample_100.csv`.
8. **Fusion:** `unified_graph.pkl` loads; number of `type='file'` nodes == 9943; concept nodes include both `source='graphrag'` and `source='pmi'`.
9. **Cross-domain bridge:** concept `期望值` — its L3 source docs span both folders; one of its L3 communities has mixed `domain_distribution`.
10. **Eval artifact:** `reports/eval_sample.md` renders; 35 rows populated (10 L3 + 10 L2 + 10 cross-domain + 5 L1).

---

## Open Choices (defaults picked; revisit after Stage 5 eval)

### ⚠️ uv sync + PyTorch CUDA workaround

`uv sync` always resolves `torch` from PyPI (CPU-only wheel, currently `2.10.0`) even when an extra index URL for the CUDA build is configured in `pyproject.toml`. To get GPU support **after every `uv sync`**:

```bash
# Download once (~2.3 GB), cached thereafter
bash setup_cuda_torch.sh

# Verify
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.5.1+cu121 True
```

If you skip this, `embed_files.py` will run on CPU only — ~10× slower on an RTX 3050 class GPU.

- **OpenCC direction:** `s2tw` (Simplified → Traditional, Taiwan standard). Switch to `s2twp` if phrase-level idioms matter.
- **L1 PMI thresholds:** `PMI ≥ 2.0`, `pair_freq ≥ 3`, `doc_freq ≥ 5`.
- **Fusion α/β (L2 file edges):** `0.7 · cosine + 0.3 · community_overlap`.
- **K in K-NN:** 15 (+ 1 for self).
- **Sample size:** 100 (50 數學 + 50 經濟金融); 5 hard seeds.
- **Relation taxonomy (L3):** fixed 6-type set.
- **L3 embedding model:** OpenAI `text-embedding-3-small`. Only affects GraphRAG's internal retrieval; K-NN still uses local BGE-M3.
- **FAISS:** CPU build (Windows has no `faiss-gpu` wheel; 10K × 1024 is trivial for CPU FlatIP).
- **No guardrails** — intentional, per user direction in MEMO §2 / §10.
