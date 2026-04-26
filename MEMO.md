# 關聯分析 — 技術選型備忘錄 (rev. 3)

**Scope.** Choose techniques (not implementations) for computing two kinds of connections over the provided Chinese text corpus:

1. **概念 ↔ 概念** (word / concept edges)
2. **檔案 ↔ 檔案** (file / chunk edges, 1 file = 1 chunk at the embedding level)

Downstream target: **RAG + a typed-relation knowledge graph**, with cross-domain bridges allowed between 數學 and 經濟金融.

**Architectural choice (rev. 3): three independent layers, then fuse.**

| Layer | Scope | Cost | Produces |
|---|---|---|---|
| **L1 — PMI concept graph** | full 10K | free (local CPU) | untyped co-occurrence edges (concept ↔ concept) |
| **L2 — BGE-M3 K-NN file graph + traditional RAG** | full 10K | free (local GPU) | file embeddings + file ↔ file edges |
| **L3 — GraphRAG demo** | 100-file stratified sample | ~US$0.05 + NIM free | typed KG (is-a / part-of / …) + Leiden communities + `graphrag query` |

L3 provides depth (typed relations, communities, LLM-authored summaries) on a representative slice; L1 + L2 give full-corpus coverage cheaply. The three are merged into one NetworkX graph.

**Why three layers instead of Path B (full-corpus GraphRAG).** Earlier plan assumed GraphRAG over all 9,943 files. Two blockers surfaced:

- **Cost/time.** At NIM's 40 RPM free ceiling, full-corpus GraphRAG is ~40–80 hours wall-clock. On paid APIs it is ~US$150–200. Too expensive for homework.
- **GraphRAG ⇄ NIM embedding incompatibility.** Microsoft's `fnllm` serializes `encoding_format: null` into the request body; NIM's strict validator rejects it. All workarounds (LiteLLM proxy, custom shim, patching `fnllm`) add infrastructure we do not want to maintain.

Splitting the pipeline sidesteps both: GraphRAG runs only on 100 files (quick + cheap), and its internal embedding calls route to paid OpenAI (`text-embedding-3-small`, ~US$0.05 one-off) instead of NIM — eliminating the `encoding_format` bug at the source.

Hard constraints:

- Local compute: **RTX 3050, 6 GB VRAM**
- LLM inference for GraphRAG (L3 only): **NVIDIA NIM API key** (free, 40 RPM) hosting **DeepSeek-V3**
- GraphRAG-internal embeddings (L3 only): **paid OpenAI** `text-embedding-3-small`
- Our K-NN embeddings (L2): **local BGE-M3 FP16** via `FlagEmbedding`, in-process
- Evaluation: qualitative only (no gold labels)

---

## 1. Data Profile

| | 數學 | 經濟金融 |
|---|---:|---:|
| Files | 2,523 | 7,420 |
| Total | 9,943 | |

File size distribution (bytes; UTF-8 Chinese ≈ 3 B/char):

| | P10 | P50 | P90 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| 數學 | 16 B | 1.3 KB | 6.1 KB | 25 KB | **3.5 MB** |
| 經濟金融 | 8 B | 1.0 KB | 7.3 KB | 35 KB | 3.7 MB |

Observed article shapes (from samples [100410](data/數學/100410.txt), [593424](data/數學/593424.txt), [1000073](data/經濟金融/1000073.txt), [54718](data/經濟金融/54718.txt)):

- **Language:** Traditional + Simplified Chinese mixed within a single corpus; template markers like `zh cn` / `zh hant` sometimes appear inline.
- **Noise:** HTML / wiki template residue (`border 1 cellpadding 4 cellspacing 0 ...`). LaTeX with spaces stripped (`\varphi\left \int_ \Omega g\ d\mu\right`).
- **Length is bimodal:** long definitional articles (倍立方, 延森不等式) vs short geographic / administrative stubs (新英灣區, 新都區). A handful of mega-files (up to 3.5 MB) live in the long tail.
- **Math vs econ split matters:** math files are definition-heavy with LaTeX density; econ files are entity-heavy (places, regulations, organizations).

---

## 2. Preprocessing Layer (shared by L1 / L2 / L3)

The *only* data transform we own. Output is a single CSV consumed by all three layers.

**Pipeline.**

1. Read each `.txt` under `data/<folder>/<id>.txt`
2. OpenCC normalize **Simplified → Traditional (Taiwan std, `s2tw`)** — corpus is already mostly Traditional; this just unifies the minority Simplified content
3. Regex scrub of HTML / wiki-template residue and collapse whitespace
4. Attach **folder name as `domain` metadata** on each row
5. Write one row per file to `processed/docs.csv`

**Output format (CSV).**

```csv
id,domain,title,text,char_count
100410,數學,延森不等式,"琴生不等式 延森不等式 以丹麥數學家約翰 延森 命名...",1854
1000073,經濟金融,新英灣區,"洋浦經濟開發區 新英灣區 位於...",421
```

- `id` = original file stem (keeps lineage back to `data/`)
- `domain` = folder name, flows as document metadata into L3 (GraphRAG) and is attached verbatim as a node attribute in L2
- `title` = first substantive line or first short header
- `text` = cleaned body
- `char_count` for triage / analytics

**Directory layout — originals are never mutated.**

```
hw/
├── data/                     ← original, immutable
│   ├── 數學/
│   └── 經濟金融/
├── processed/
│   ├── docs.csv              ← full corpus for L1 + L2
│   └── sample_100.csv        ← concept-seeded sample for L3
├── analysis/                 ← L1 + L2 + fusion artifacts
└── graphrag_project/         ← L3 artifacts
```

**Guardrails: intentionally none.** Stubs and mega-files pass through unchanged. Trade-offs accepted:

- Empty-ish stubs (P10 ≈ 8–16 B) will generate low-signal embeddings in L2 and low-signal tokens in L1. Revisit only if eval surfaces noise.
- The 3.5 MB outlier expanded into ~1000 GraphRAG chunks under the old full-corpus plan. Under the new three-layer plan it almost certainly stays out of the 100-file sample, so the token-spend concern is moot; it still gets a single BGE-M3 embedding in L2 (batched on GPU).

**Alternatives considered.**

| Option | Trade-off |
|---|---|
| OpenCC ✅ | Standard, accurate. |
| hanziconv | Simpler, less accurate on edge cases. |
| jieba (user-dict) ✅ | Required by L1 for tokenization. |
| CKIP Transformers | Stronger traditional Chinese, heavier install — wasted here. |
| Per-file `.txt` output | Works, but CSV is what GraphRAG consumes natively. |
| Parquet output | Typed + compact; CSV is fine at 10K rows and easier to diff. |

---

## 3. L1 — PMI Concept Graph (full 10K, free, untyped)

Purpose: give full-corpus concept coverage that L3 cannot afford to provide. Untyped edges, but deterministic and ~free.

**Pipeline.**

1. Load `processed/docs.csv`.
2. Tokenize each doc with `jieba.cut` on a math/econ user-dict; keep tokens with `len ≥ 2`; drop stopwords.
3. Treat each document as a **bag of tokens** — file-level co-occurrence, no sliding window. This is coarser but matches the "1 file = 1 chunk" framing.
4. Compute `doc_freq[t]` and `pair_freq[(t₁, t₂)]` for pairs where both terms have `doc_freq ≥ 5`.
5. `PMI(t₁, t₂) = log( p(t₁, t₂) / ( p(t₁) · p(t₂) ) )` with probabilities over documents.
6. Keep edges with `PMI ≥ 2.0` and `pair_freq ≥ 3` (tunable after eval).
7. Attach a computed `domain_distribution = {數學: p, 經濟金融: 1−p}` to each concept node from its source docs.

**Why PMI, not LLM extraction at scale.** PMI is the standard cheap baseline for concept co-occurrence; it is deterministic, audit-able, and has no API dependency. It gives **untyped** edges — acceptable since L3 contributes the typed relations we care about.

**What L1 does *not* do.** No disambiguation (琴生 vs 延森 stay separate unless a jieba user-dict entry collapses them), no hierarchy, no relation types. These are all L3's job on its 100-file sample.

**How L1 complements RAG (L2 retrieval).** L1 is a *graph over concepts*, not over passages. At retrieval time the system can use L1 to expand a concept query (e.g., 期望值 → 隨機變數, 變異數, 機率分布) and feed the expanded term set into L2's traditional RAG. This is the "RAG + concept graph" combination we promised downstream.

**Directional PMI (parallel variant, not fused).** A forward-window directional PMI is computed alongside symmetric PMI: `PMI(t1→t2)` uses only co-occurrences where t2 appears **after** t1 within a fixed token window (window_size=50 tokens). This produces asymmetric scores — `PMI(inflation→recession) ≠ PMI(recession→inflation)` — useful for inferring "A tends to precede / cause B" ordering. Output is `analysis/directional_pmi.parquet` with `src`, `dst`, `pmi_forward`, `pmi_reverse`, `co_doc_count`, `asymmetry` columns. This is a **research artifact only**; it is not integrated into the unified graph or query APIs. See `src/build_directional_pmi.py`.

**Alternatives considered.**

| Option | Verdict |
|---|---|
| **PMI (doc-level)** ✅ | Fast, free, deterministic; matches 1-file-1-chunk framing. |
| PMI (window-level) | Finer-grained but introduces a window-size hyperparameter. Not worth it for our coarse concept-graph use. |
| NPMI | Normalized; good if we later want thresholds that are scale-free. Revisit at eval time. |
| TextRank / LexRank | Ranks sentences / tokens by importance, not pairwise edges. Wrong shape. |
| LLM over full 10K | Best quality but ruled out by cost / time (see opening section). |

---

## 4. L2 — BGE-M3 K-NN File Graph (full 10K, free, local GPU)

**One embedding model, one job.** BGE-M3 (~2.3 GB, FP16 on 6 GB VRAM) runs in-process via `FlagEmbedding`. We embed each `text` from `processed/docs.csv` once and build a K-NN graph with K = 15 on cosine similarity via FAISS CPU (`IndexFlatIP` after L2-normalizing; Windows has no `faiss-gpu` wheel).

**Fusion weight for L2 file-file edges (set in `fuse_graph.py`, not in L2 itself).**

```
edge_weight(fᵢ, fⱼ) = α · cos(E(fᵢ), E(fⱼ)) + β · community_overlap(fᵢ, fⱼ)
```

- `E(·)` = BGE-M3 document embedding
- `community_overlap` = Jaccard over the set of L3 communities touched by each file
- `α ≈ 0.7, β ≈ 0.3` as a sensible default; tune qualitatively
- Files outside the 100-sample have empty community sets, so their edges fall back to pure cosine — acceptable

**Note on "shared BGE-M3" (rev. 2 invariant, removed in rev. 3).** Earlier plans routed GraphRAG's internal embeddings through the same BGE-M3 model as our K-NN. We drop that coupling: GraphRAG (L3) now uses OpenAI's `text-embedding-3-small`. BGE-M3 is used **only** for L2's file-level K-NN. The two embedding spaces are disjoint and that is fine — they serve different graphs and are not compared directly.

**Alternatives considered.**

| Option | Trade-off |
|---|---|
| TF-IDF cosine | Interpretable; weak on Trad↔Simp variants and paraphrase. |
| **BGE-M3 K-NN** ✅ | Best quality fitting 6 GB. |
| `bge-large-zh-v1.5` | ~1.3 GB fallback if VRAM tight. |
| `text2vec-large-chinese` | Strong on simplified only. |

---

## 5. L3 — GraphRAG Demo on a 100-File Sample

Microsoft [GraphRAG](https://github.com/microsoft/graphrag) runs end-to-end over a **data-driven cross-domain seeded sample** of 100 files: 50 數學 + 50 經濟金融, biased toward files containing cross-domain anchor terms derived from L1 PMI (`concept_domains.parquet`). Seeds are computed via an **anchor-neighbor** approach: 10 math-specific anchor terms (函數, 定理, 證明, 方程式, 代數, 幾何, 導數, 積分, 機率, 統計) and 10 econ-specific anchors (貨幣, 利率, 投資, 風險, 價格, 市場, 交易, 銀行, 貿易, 通貨膨脹) are used to find their cross-domain PMI neighbors; the top-8 neighbors ranked by anchor-PMI sum form the hard seeds for the stratified sample. GraphRAG owns:

- **Chunking** (~1200-token chunks internal to GraphRAG; long files split, short files = 1 chunk)
- **Entity extraction** via LLM (DeepSeek-V3 on NIM free)
- **Typed-relation extraction** via LLM — same pass, same prompt
- **Entity disambiguation / canonicalization**
- **Leiden community detection** on the entity graph
- **Community summaries** via LLM

**Relation taxonomy (fixed schema).** The default GraphRAG prompt is open-vocabulary; we constrain it to: `is-a`, `part-of`, `defines`, `special-case-of`, `used-in`, `related-to`. Fixed schemas keep the typed-edge vocabulary stable across runs and make the KG usable for retrieval.

**Why sample rather than run on all 10K.** See cost/time reasoning in the opening. 100 files × ~3 chunks/file × 2 LLM calls/chunk ≈ 600 calls ≈ 17 minutes at 35 RPM; plus community summaries, total wall-clock ~45 minutes.

**Why the sample is enough.** L3 contributes *depth* (typed relations + communities), not *coverage*. Cross-domain seed concepts are derived from L1 data, ranked by PMI connectivity, guaranteeing each selected concept has source files in the sample and mixed-domain communities emerge naturally. Full-corpus coverage is delegated to L1 (PMI, untyped) and L2 (embeddings).

**GraphRAG internal embeddings: paid OpenAI.** `settings.yaml` points `embeddings.llm.api_base` at OpenAI's default endpoint with `model: text-embedding-3-small`. Cost: ~US$0.05 for the 100 files. This choice eliminates the `encoding_format: null` bug because the bug only reproduces against NIM's strict validator — OpenAI accepts `null`. No proxy, no shim, no patched `fnllm`.

**Alternatives considered (rejected in rev. 3).**

| Option | Verdict |
|---|---|
| Run a local `infinity` BGE-M3 server for GraphRAG | Works, but re-introduces a local server dependency we are explicitly dropping. Also keeps the "shared BGE-M3" invariant that rev. 3 abandons. |
| LiteLLM proxy in front of NIM | Adds infra; doesn't solve `encoding_format: null` without a custom filter. |
| Patch `fnllm` | Unmaintainable across graphrag upgrades. |
| HuggingFace Inference API (BGE-M3) | Free but ~1 req/s; would stretch the 45-min run to 2–3 hrs. |
| Custom NIM proxy (our `src/embed_proxy.py`) | Worked in early testing but adds a 500 LOC dependency for ~US$0.05 of savings. Delete. |

---

## 6. Folder Labels → Document Metadata (not concept nodes)

Folder names attach at the **document** level in `processed/docs.csv` (`domain` column). They flow as metadata into L3 and as a node attribute in L2; they **are not hand-coded as concept nodes** in any layer.

Three derived uses fall out naturally:

1. **File-level filter.** "Show me entities mostly from 數學 files."
2. **Concept-level computed attribute.** Each concept (L3 entity or L1 PMI term) accumulates a domain distribution from its source documents (e.g., 期望值 might be `{數學: 0.7, 經濟金融: 0.3}`) — a *computed* attribute, not an extraction rule.
3. **Community-level domain mix (L3 only).** Communities with mixed `{數學, 經濟金融}` distributions are the **cross-domain bridges** we care about.

---

## 7. Integration Architecture

```
                    data/*.txt (9,943 files, immutable)
                              │
                              │  OpenCC s2tw + scrub + attach domain
                              ▼
                      processed/docs.csv
                  (id, domain, title, text, char_count)
                              │
       ┌──────────────────────┼───────────────────────────┐
       ▼                      ▼                           ▼
 L1: jieba + PMI       L2: BGE-M3 embed           (sample_100 selector)
    + Directional PMI   + FAISS K-NN (full 10K)           │
  (full 10K, parallel)                                     │
       │                      │                           ▼
       ▼                      ▼                 processed/sample_100.csv
  pmi_graph.parquet    embeddings.parquet                 │
+ concept_domains.parquet   knn_graph.parquet             ▼
+ directional_pmi.parquet                              L3: graphrag index
                                                  (DeepSeek-V3 / NIM
                                                   + text-embed-3-small / OpenAI)
                                                          │
                                                          ▼
                                          graphrag_project/output/*.parquet
                                          (entities, typed relations,
                                           communities, community reports)

                       ┌──────────────┼──────────────┐
                       ▼              ▼              ▼
                          analysis/unified_graph.pkl
                     (NetworkX: file nodes + concept nodes;
                      L1 untyped + L3 typed concept edges;
                      L2 file-file edges weighted by
                      0.7·cosine + 0.3·community_overlap)
                                      │
                                      ▼
                       concept_sim.py / file_sim.py
                        (query APIs, qualitative eval)
```

**Why this shape works.**

- Preprocessing is a pure transform; originals stay immutable.
- Cost is bounded: full-corpus work runs locally and free; only ~US$0.05 of cloud spend on the sample.
- Cross-domain bridges emerge from the data via L3 community domain-mix, not hand-tuned edges.
- No local servers: BGE-M3 runs in-process; GraphRAG calls cloud endpoints directly.

---

## 8. Chosen Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Normalization | **OpenCC (`s2tw`)** | Normalize mixed corpus to Traditional (Taiwan std). |
| Noise scrub | Regex (custom) | Strips HTML/wiki/LaTeX residue. |
| Tokenizer | **jieba** + domain user-dict | Required by L1 PMI. |
| Preprocessing output | **CSV** (`processed/docs.csv`) | Native GraphRAG input; pandas-friendly. |
| L1 co-occurrence | **PMI** (doc-level) | Free, deterministic. |
| L2 embeddings | **BGE-M3** (fallback: `bge-large-zh-v1.5`) | Local FP16 via `FlagEmbedding`; in-process, no server. |
| L2 K-NN index | **FAISS CPU `IndexFlatIP`** | 10K × 1024 is trivial on CPU; Windows has no faiss-gpu wheel. |
| L3 LLM | **DeepSeek-V3 via NVIDIA NIM** (free, 40 RPM) | OpenAI-compatible; `requests_per_minute: 35` under the 40 ceiling. |
| L3 LLM fallback | **Qwen2.5-72B-Instruct** (also on NIM) | Swap in if DeepSeek-V3 throttled. |
| L3 embeddings | **OpenAI `text-embedding-3-small`** (paid, ~US$0.05 for 100 files) | Eliminates the `encoding_format: null` bug vs NIM; isolated from L2's BGE-M3. |
| L3 extraction pipeline | **Microsoft GraphRAG** (`graphrag index`) | Owns chunking, NER, RE, disambiguation, communities, summaries. |
| L3 community detection | **Leiden** | Internal to GraphRAG. |
| Graph store | **parquet** (GraphRAG native) → **NetworkX** for analytics | Neo4j only if scaling out. |
| Package management | **`uv`** | `pyproject.toml`; no pip / conda / poetry. |

---

## 9. Staged Roadmap

Each stage yields a usable artifact; stop when quality suffices.

> ⚠️ **Claude Code will NOT auto-execute long-running or paid steps.** For Stage 1 (`embed_files.py`) and Stage 3 (`graphrag index`), announce the command and instruct the user to run it in a new terminal.

1. **Stage 0 — Preprocess ($0, local, ~1 min).** OpenCC + scrub + build `processed/docs.csv` with `domain` metadata. Artifact: clean CSV.
2. **Stage 1 — L2 embeddings + K-NN ($0, local GPU, ~80 min).** BGE-M3 FP16 on all 10K; FAISS K-NN. Artifacts: `embeddings.parquet`, `knn_graph.parquet`. ⚠️ Run in new terminal.
3. **Stage 2 — L1 PMI ($0, local CPU, ~3 min).** jieba tokenize + doc-level PMI. Artifact: `pmi_graph.parquet`.
4. **Stage 3 — L3 GraphRAG on sample_100 (~US$0.05 OpenAI + NIM free, ~45 min).** Seeds are PMI anchor-neighbor cross-domain concepts (see §5). Run `graphrag index --root .\graphrag_project`. Artifacts under `graphrag_project/output/`. `graphrag query` works on the sample. ⚠️ Run in new terminal.
5. **Stage 4 — Fusion in NetworkX (~5 min).** Merge L1 + L2 + L3 into `unified_graph.pkl`; compute domain-mix per concept; surface cross-domain bridges.
6. **Stage 5 — Qualitative eval (<1 min).** Hand-inspect 30+ sampled connections across the three layers; iterate thresholds.

**Budget estimate.** Effectively **~US$0.05** for the full pipeline (OpenAI embeddings on the 100-file sample). All other LLM work runs on NIM's free tier; all embeddings except L3 are local GPU.

---

## 10. Open Questions & Limitations

- **No guardrails (user direction).** 3.5 MB outlier and empty stubs pass through. Under the three-layer plan the outlier no longer dominates token spend because L3 only touches the 100-file sample; the outlier is almost certainly excluded. L2 still embeds it (single BGE-M3 call, non-critical).
- **LaTeX residue** in math files will still pollute L2 embeddings if not scrubbed. Consider a targeted strip pass that replaces LaTeX with a `__MATH__` sentinel before embedding, while keeping raw form for the LLM (L3).
- **GraphRAG prompts are English-tuned by default.** The one-off `graphrag prompt-tune --language "Traditional Chinese"` step + manual edit to lock the 6-relation taxonomy is the fix.
- **NIM rate limit (40 RPM)** still gates L3 wall-clock. GraphRAG's built-in token-bucket + `tenacity` backoff (`requests_per_minute: 35`, `max_retries: 10`) absorbs 429 bursts. No custom rate limiter.
- **Disjoint embedding spaces (L2 BGE-M3 vs L3 OpenAI).** Accepted consequence of decoupling in rev. 3. They serve different graphs; we never compare a BGE-M3 vector to an OpenAI vector directly.
- **L1 PMI is untyped and non-disambiguated.** 琴生 / 延森 remain separate unless handled by the jieba user-dict. L3 provides disambiguation on the 100-file slice; outside that slice, concept-concept similarity from L1 alone is an approximation.
- **Evaluation without gold labels.** Build a small (~30-pair) hand-curated eval set during Stage 5 to anchor qualitative discussion.

---

## Appendix: Installing PyTorch CUDA Build Under uv

**Root cause.** When `uv sync` resolves `torch`, it picks the first version on PyPI satisfying `>=2.5.0,<2.11` — which is `2.10.0`, the **CPU-only** wheel. Adding the PyTorch CUDA index URL (`https://download.pytorch.org/whl/cu121`) via `tool.uv.index` in `pyproject.toml` does not help: uv's lock mechanism still records only the PyPI CPU wheel, and `override-dependencies` only affects resolution, not which wheel gets installed.

**Solution.** After `uv sync`, manually extract the CUDA wheel into the venv's `site-packages`:

```bash
# 1. sync to get CPU torch first
uv sync

# 2. extract CUDA wheel directly (bypasses uv pip's overwrite behavior)
python -c "
import zipfile, os, shutil

wheel = os.path.join(os.environ['LOCALAPPDATA'], 'torch-cu121',
                     'torch-2.5.1+cu121-cp311-cp311-win_amd64.whl')
site = '.venv/Lib/site-packages'

shutil.rmtree(os.path.join(site, 'torch'), ignore_errors=True)
for d in os.listdir(site):
    if d.startswith('torch-'):
        shutil.rmtree(os.path.join(site, d), ignore_errors=True)

with zipfile.ZipFile(wheel) as z:
    z.extractall(site)
print('Done')
"
```

Or use the provided helper script (wheel is cached locally after first download):

```bash
bash setup_cuda_torch.sh
```

**Verify:**
```python
import torch
print(torch.__version__, torch.cuda.is_available())  # 2.5.1+cu121 True
```

**Caveat.** Re-run `setup_cuda_torch.sh` after every `uv sync` — the next sync will overwrite torch back to CPU. Alternatively, bake it into the pipeline task chain (see `src/task_chain.py` or Makefile target) so it runs automatically after setup.

---

*Memo rev. 3. Source data: [company/UC/hw/data](data). Rev-2 (Path B full-corpus GraphRAG) retired; rationale in the opening section.*
