# 關聯分析 — Algorithm Design Document

This file records all algorithm design details, methods, plans, tools, and technique trade-offs. It supplements `MEMO.md` and `PLAN.md`.

---

## 1. System Architecture: Three Independent Layers, Then Fuse

```
data/*.txt (9,943 files, immutable)
          │
          │  OpenCC s2tw + scrub + attach domain
          ▼
  processed/docs.csv
      (id, domain, title, text, char_count)
          │
  ┌───────┼───────┐
  ▼       ▼       ▼
L1: jieba    L2: BGE-M3    (sample_100 selector)
+ PMI        embed             │
+ Directional PMI              │
 (full 10K, parallel)          ▼
    │           │      processed/sample_100.csv
    ▼           ▼              │
pmi_graph.parquet  embeddings.parquet    ▼
+ concept_domains.parquet  + knn_graph.parquet   L3: graphrag index
+ directional_pmi.parquet                         (qwen/qwen3.5-397b-a17b via NIM +
                                     text-embedding-3-small via OpenAI)
                                           │
                                           ▼
                            graphrag_project/output/*.parquet
                            (entities, typed relations,
                             communities, community reports)
                                           │
              ┌────────────────────────────┴─────────────┐
              ▼                                       ▼
                         analysis/unified_graph.pkl
          (NetworkX: file nodes + concept nodes;
           L1 untyped + L3 typed concept edges;
           L2 file-file edges weighted by
           0.7·cosine + 0.3·community_overlap)
```

**Why three layers instead of one:**

Each layer produces a graph that the others cannot replicate, and the combination enables retrieval that no single method achieves alone:

- **L1 — PMI concept graph (10K corpus, free)**: captures lexical co-occurrence across all 9,943 documents — every shared vocabulary between any pair of math or economics articles, regardless of topic similarity. Coverage is maximal; structure is flat and untyped.
- **L2 — BGE-M3 K-NN file graph (10K corpus, free)**: captures semantic similarity that lexical methods miss — two articles may be topically near without sharing any significant vocabulary. Dense embeddings filter out template noise and surface content-level affinity.
- **L3 — GraphRAG typed knowledge graph (100-file sample, paid)**: captures typed relations (is-a, part-of, defines, used-in, special-case-of, related-to) that no co-occurrence or embedding method can infer. Entities and relations are grounded in specific text spans and LLM-generated community summaries.

**Together**, L1 delivers breadth (concept recall), L2 delivers precision (semantic similarity), and L3 delivers structure (typed relational reasoning). A single-layer approach — whether pure PMI, pure embedding similarity, or pure GraphRAG on 10K — would have to sacrifice at least one of these three properties.

---

## 2. Data Profile

|               | 數學    | 經濟金融 |
|---------------|------:|--------:|
| File count    | 2,523 | 7,420  |
| **Total**     | **9,943**   |         |

### 2.1 Language and Format

- **Language**: Predominantly Traditional Chinese (繁體中文); a minority of Simplified Chinese (簡體中文) content is normalized via OpenCC `s2tw` (Taiwan standard).
- **Noise**: HTML / Wiki template residue (e.g., `border 1 cellpadding 4 cellspacing 0`), CSS tokens, broken LaTeX (e.g., `\varphi\left \int_`).
- **Length distribution**: Bimodal — long definitional articles (e.g., 倍立方, 延森不等式) vs. short geographic / administrative stubs (e.g., 新英灣區, 都區). A handful of mega-files (up to 3.5 MB) live in the long tail.
- **Math files**: High LaTeX density (`\varphi`, `\int`, `\Omega`).
- **Econ files**: High entity density (locations, regulations, organizations).

### 2.2 Preprocessing Pipeline

**Input**: `data/**/*.txt`
**Output**: `processed/docs.csv` (columns: `id, domain, title, text, char_count`)

**Steps per file**:

1. Read UTF-8.
2. OpenCC `s2tw` — Simplified → Traditional, Taiwan standard.
3. Regex scrub: strip HTML / Wiki template lines, CSS tokens, broken LaTeX, collapse whitespace.
4. Title selection: first non-empty non-template line; fallback to `id`.
5. Attach `domain` = parent folder name (`數學` | `經濟金融`).

**Output format**:

```csv
id,domain,title,text,char_count
100410,數學,延森不等式,"琴生不等式 延森不等式 以丹麥數學家約翰 延森 命名...",1854
1000073,經濟金融,新英灣區,"洋浦經濟開發區 新英灣區 位於...",421
```

**Design rationale**: No guardrails (length filtering, empty-stub removal) — intentional per project direction. The mega-file outlier is still a single BGE-M3 call in L2 (non-blocking); it is nearly certain to be excluded from the 100-file L3 sample (data-driven seed sampling is biased toward cross-domain concepts, not outliers).

---

## 3. L1 — PMI Concept Graph (full corpus, free, local CPU, untyped edges)

### 3.1 Goal

Provide full-corpus concept coverage that L3 cannot afford. Untyped edges, but deterministic and zero-cost.

### 3.2 Method: Document-Level Co-occurrence PMI

**Algorithm**:

1. Tokenize each doc with `jieba.cut`; use a domain-specific user dictionary (math + economics terms); keep tokens with `len(token) ≥ 2`; drop stopwords.
2. Treat each document as a **bag of unique tokens** — file-level co-occurrence, **not** a sliding window.
3. Track `doc_freq[t]` (number of documents containing token t) and `pair_freq[(t₁, t₂)]` (number of documents containing both t₁ and t₂). Only consider pairs where both tokens have `doc_freq ≥ 5`.
4. Compute PMI:

$$
\text{PMI}(t_1, t_2) = \log \frac{p(t_1, t_2)}{p(t_1) \cdot p(t_2)}
$$

where `p(t) = doc_freq[t] / N_docs` and `p(t₁, t₂) = pair_freq[(t₁, t₂)] / N_docs`.

5. Keep edges with `PMI ≥ 2.0` and `pair_freq ≥ 3`.
6. Attach a computed `domain_distribution = {數學: p, 經濟金融: 1−p}` to each concept node — a **derived attribute**, not hand-coded.

**Why PMI over LLM extraction at scale**: PMI is the standard cheap baseline for concept co-occurrence; deterministic, auditable, no API dependency. L3 provides the typed relations we care about.

**Threshold trade-offs**:

| Threshold | Effect |
|-----------|--------|
| PMI ≥ 2.0, pair_freq ≥ 3 | Standard; captures statistically significant co-occurrence |
| NPMI (Normalized PMI) | Scale-free; good if cross-scale thresholds are needed later. Revisit at eval. |

**Outputs**:

- `pmi_graph.parquet`: src, dst, pmi, co_doc_count
- `concept_domains.parquet`: concept, doc_freq, docs_數學, docs_經濟金融, cross_domain_count

### 3b. Directional PMI — Forward-Window Asymmetric Variant (parallel L1)

**Goal**: Compute asymmetric PMI scores that capture "A tends to precede B" ordering — parallel to symmetric L1, not fused into the unified graph.

**Method**: Sliding forward window (window_size=50 tokens) over ordered token sequences within each document.

**Algorithm**:

1. Tokenize each doc with `jieba.cut` (same filter as L1); track **ordered positions**, not bag-of-tokens.
2. For each token t1 at position i, count occurrences of t2 at positions j where i < j ≤ i + window_size.
3. Accumulate `forward_count[t1→t2]` across all documents.
4. Compute directional PMI:

$$
\text{PMI}(t_1 \rightarrow t_2) = \log \frac{p(t_2 \mid t_1)}{p(t_2)} = \log \frac{\text{forward\_count}(t_1 \rightarrow t_2) / \text{doc\_freq}(t_1)}{\text{doc\_freq}(t_2) / N}
$$

5. Keep directed edges with `forward_count ≥ 3` and `PMI(t1→t2) ≥ 1.5` (lower than symmetric L1's 2.0 because directional signal is sparser).
6. For each pair (A, B) where at least one direction passes the threshold, record: `pmi_forward` (A→B), `pmi_reverse` (B→A, if available), `co_doc_count`, `asymmetry = PMI(A→B) − PMI(B→A)`.

**Output**: `analysis/directional_pmi.parquet` with columns: `src`, `dst`, `pmi_forward`, `pmi_reverse`, `co_doc_count`, `asymmetry`.

**Why parallel, not fused**: Directional PMI captures lexical-order patterns, not semantic entailment. It cannot distinguish phrase collocation from causal relation. It is a **research artifact only** — not integrated into the unified NetworkX graph or query APIs.

**Empirical results** (from actual run):
- Edge count: 2,429,061 (0.20× symmetric PMI's 12,427,187)
- Bidirectional edges: 993,760 (40.9%)
- Unidirectional edges: 1,435,301 (59.1%)
- Asymmetry range: −4.46 to +10.39

---

## 4. L2 — BGE-M3 K-NN File Graph (full corpus, free, local GPU)

### 4.1 Goal

Build a **file-to-file** similarity graph over the full 10K corpus based on embedding-level semantic similarity — complementary to the purely lexical PMI of L1.

### 4.2 Method: Local BGE-M3 Embedding + FAISS K-NN

**Embedding model**: BAAI/bge-m3 (FP16, ~2.3 GB)
**Runtime**: Local in-process, no server, called directly via `FlagEmbedding`
**VRAM**: FP16 + batch_size=4 + max_length=8192 ≈ 4.5 GB peak (fits in RTX 3050's 6 GB)
**Fallback**: Reduce max_length to 4096 if OOM; or switch to `bge-large-zh-v1.5` (~1.3 GB)

**Embedding parameters**:

```python
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
vecs = model.encode(docs['text'].tolist(), batch_size=4, max_length=8192)['dense_vecs']
```

**K-NN graph construction**:

1. L2-normalize all 9,943 document vectors (cosine similarity via IP search).
2. FAISS `IndexFlatIP` (CPU build; no faiss-gpu wheel on Windows).
3. K=15 (+ 1 self, excluded to remove self-loops).
4. Cosine similarity range: 0.4312 – 1.0000.

```python
faiss.normalize_L2(X)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
D, I = index.search(X, k=16)  # +1 for self
```

**Fusion weight (L2 → unified_graph)**:

$$
\text{edge\_weight}(f_i, f_j) = \alpha \cdot \cos(E(f_i), E(f_j)) + \beta \cdot \text{community\_overlap}(f_i, f_j)
$$

- `α ≈ 0.7, β ≈ 0.3` (sensible default; tune qualitatively after eval)
- `community_overlap` = Jaccard over the set of L3 communities touched by each file
- Files outside the 100-sample have empty community sets, so their edges fall back to pure cosine — acceptable

---

## 5. L3 — GraphRAG Demo (100-file sample, paid OpenAI embed + NIM free chat)

### 5.1 Goal

Provide **depth** on a concept-seeded 100-file stratified sample: typed relations + Leiden communities + LLM-authored community summaries. L1 + L2 provide cheap breadth; L3 provides the structured semantic layer.

### 5.2 Sampling Strategy: Seed-of-Seed

**Why data-driven (not random) sampling**:
- Random sampling cannot guarantee cross-domain bridge concepts appear at sufficient frequency
- Seed-of-seed ensures concepts co-occurring with domain name words (數學, 經濟, 金融) in the PMI graph are preferentially included

**Algorithm**:

1. Domain name words (數學, 經濟, 金融) as root nodes.
2. Take their PMI-graph neighbors as hard seeds.
3. Each seed selects at least one 數學 and one 經濟金融 document.
4. Stratified fill to 100 (50 數學 + 50 經濟金融), weighted by `char_count^0.5`.
5. Hard cap at 100 rows.

**Sampling result** (actual run):
- Total seeds: 9 (3 seeds × 3 domain names)
- 數學 documents: 51 | 經濟金融 documents: 49

### 5.3 GraphRAG Pipeline

**Internal embedding (paid, bypasses NIM incompatibility)**: `text-embedding-3-small` via OpenAI
**Cost**: ~US$0.05 for 100 files
**Why not NIM for embedding**: NIM's strict validator rejects `encoding_format: null` which `fnllm` sends. OpenAI accepts `null`. HuggingFace Inference API is too slow (~1 req/s, extending the run from 45 min to 2–3 hours).

**Chunk parameters** (GraphRAG internal):
- size: 1200 tokens
- overlap: 100 tokens
- encoding_model: o200k_base

**Entity extraction**:
- Model: qwen/qwen3.5-397b-a17b via NVIDIA NIM
- Entity types: `[概念, 定理, 公式, 人物, 地點, 組織]`
- max_gleanings: 1

**Relation extraction** (same pass, same prompt):
- Relation types (fixed schema): `is-a, part-of, defines, special-case-of, used-in, related-to`
- Why fixed schema: open-vocabulary leads to relation-type drift across runs; fixed schema keeps the KG stable and usable for retrieval

**Community detection**: Leiden algorithm, max_cluster_size: 10

### 5.4 RPM Configuration (Safe Margin)

NIM free tier: **40 RPM**
`settings.yaml` setting: `requests_per_minute: 10`
GraphRAG built-in token-bucket + `tenacity` exponential backoff: `max_retries: 10`

---

## 6. Fusion Architecture

### 6.1 Data Sources

| Source | Files |
|--------|-------|
| L3 GraphRAG | `create_final_entities.parquet`, `create_final_relationships.parquet`, `create_final_communities.parquet`, `create_final_text_units.parquet` |
| L2 K-NN | `knn_graph.parquet` |
| L1 PMI | `pmi_graph.parquet` |

### 6.2 Fusion Algorithm (NetworkX)

1. **File nodes** (all 9,943): `type='file'`, `domain`
2. **Concept nodes**:
   - L3 entities: `type='concept'`, `source='graphrag'`, `entity_type`, `description`
   - L1 PMI nodes not in L3: `type='concept'`, `source='pmi'`, `domain_distribution`
3. **Concept ↔ concept edges**:
   - L3 typed: `relation ∈ {is-a, part-of, defines, special-case-of, used-in, related-to}`, `source='graphrag'`
   - L1 untyped: `pmi`, `co_doc_count`, `source='pmi'`
4. **File ↔ file edges (L2)**:
   `community_overlap = Jaccard(doc_to_communities[src], doc_to_communities[dst])`
   `weight = 0.7 · cos_sim + 0.3 · community_overlap`
5. **Concept ↔ file edges (L3)**: each L3 concept attached to the files it appears in

---

## 7. Tech Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| **Normalization** | OpenCC `s2tw` | Simplified → Traditional, Taiwan std |
| **Noise scrub** | Regex (custom) | Strips HTML/Wiki/LaTeX residue |
| **Tokenizer** | jieba + domain user-dict | Required by L1 PMI |
| **Preprocessing output** | CSV | Native GraphRAG input; pandas-friendly |
| **L1 co-occurrence** | PMI (doc-level) | Free, deterministic |
| **L2 embeddings** | BGE-M3 (FP16 via FlagEmbedding) | Local in-process, no server |
| **L2 K-NN index** | FAISS CPU `IndexFlatIP` | Windows has no faiss-gpu wheel; 10K×1024 is trivial on CPU |
| **L3 chat LLM** | qwen/qwen3.5-397b-a17b via NIM (free, 40 RPM) | OpenAI-compatible endpoint |
| **L3 internal embedding** | OpenAI `text-embedding-3-small` (paid, ~US$0.05) | Bypasses `encoding_format: null` issue; isolated from L2's BGE-M3 |
| **L3 extraction pipeline** | Microsoft GraphRAG | Owns chunking, NER, RE, disambiguation, communities, summaries |
| **L3 community detection** | Leiden | Built into GraphRAG |
| **Graph store** | Parquet (GraphRAG native) → NetworkX (analytics) | Neo4j only if scaling out |
| **Package management** | `uv` | `pyproject.toml`; no pip / conda / poetry |

---

## 8. Key Design Decision Trade-offs

| Decision | Chosen | Rejected alternatives | Rejection reason |
|----------|--------|----------------------|-----------------|
| **Embedding spaces** | Two disjoint spaces (L2 BGE-M3 + L3 OpenAI) | Shared BGE-M3 via local `infinity` server | Adds a local server dependency; L3 embedding via NIM hits the `encoding_format: null` bug |
| **L3 embedding** | OpenAI `text-embedding-3-small` | NIM or HuggingFace Inference API | NIM strict validator rejects `encoding_format: null`; HuggingFace too slow (~1 req/s → 2–3 hrs) |
| **PMI scope** | Full 10K file-level co-occurrence | Sliding window or LLM over full 10K | Sliding window introduces a window-size hyperparameter; LLM full-corpus too expensive |
| **Directional PMI** | Forward-window directional PMI (parallel L1) | Not computed | Directional PMI captures lexical-order proxy for temporal precedence; not fused into unified graph (research artifact only) |
| **K-NN K value** | 15 (+ 1 self) | — | Balances local vs global neighborhood |
| **Relation schema** | Fixed 6-type set | Open-vocabulary | Stable across runs; avoids type drift |
| **Sampling strategy** | Seed-of-seed (PMI domain-name → neighbors → docs) | Random sampling | Ensures cross-domain bridge concepts are represented |
| **Preprocessing guardrails** | None (intentional) | Length filters or empty-stub removal | User direction; mega-file outlier is excluded from L3 sample anyway |
| **Torch CUDA setup** | Manual wheel置换 (`setup_cuda_torch.sh`) | `uv pip install torch --index-url ...` | uv sync overwrites CUDA wheel back to CPU each time |
| **NIM model** | qwen/qwen3.5-397b-a17b | deepseek-ai/deepseek-v3 | qwen available on NIM free endpoint; no separate deployment needed |

---

## 9. Output Artifact Map

| Artifact | Source | Description |
|----------|--------|-------------|
| `processed/docs.csv` | L0 preprocessor | 9,943 rows, full corpus |
| `processed/sample_100.csv` | L3a sampler | 100 rows, concept-seeded |
| `analysis/embeddings.parquet` | L2 embedder | 9,943 × 1024-dim dense vectors |
| `analysis/knn_graph.parquet` | L2 build_knn | 105,747 edges, K=15 |
| `analysis/pmi_graph.parquet` | L1 build_pmi | 12,427,187 edges, PMI≥2.0 |
| `analysis/concept_domains.parquet` | L1 build_pmi | domain distribution per concept |
| `analysis/directional_pmi.parquet` | L1b build_directional_pmi | 2,429,061 directed edges, forward window PMI |
| `graphrag_project/output/entities.parquet` | L3 graphrag | 250 entities (after dedup) |
| `graphrag_project/output/relationships.parquet` | L3 graphrag | 223 typed relations |
| `graphrag_project/output/communities.parquet` | L3 graphrag | 6 Leiden communities |
| `graphrag_project/output/community_reports.parquet` | L3 graphrag | LLM-authored community summaries |
| `graphrag_project/output/text_units.parquet` | L3 graphrag | chunk-level with document_ids |
| `analysis/unified_graph.pkl` | Fusion | NetworkX graph, all layers merged |

---

## 10. Execution Order

```
0. uv sync
   ⚠️ bash setup_cuda_torch.sh          # run after every uv sync
1. uv run python src/preprocess.py      # ~1 min → docs.csv
2. uv run python src/embed_files.py    # ⚠️ ~80 min on GPU (new terminal)
3. uv run python src/build_knn.py       # <1 min → knn_graph.parquet
4. uv run python src/build_pmi.py       # ~3 min → pmi_graph.parquet
4b. uv run python src/build_directional_pmi.py  # ~3 min → directional_pmi.parquet
5. uv run python src/sample_100.py     # <1 min → sample_100.csv
6. uv run graphrag init --root .\graphrag_project  # one-off
7. uv run graphrag prompt-tune ...       # one-off, ~5 min
8. Manually edit prompts/entity_extraction.txt to lock 6-relation schema
9. uv run graphrag index --root .\graphrag_project  # ⚠️ ~45 min, ~US$0.05 (new terminal)
10. uv run python src/fuse_graph.py     # ~5 min → unified_graph.pkl
11. uv run python src/eval.py           # <1 min → reports/eval_sample.md
```

**⚠️ Long-running / paid steps must NOT be auto-executed from Claude Code.** Steps 2 (`embed_files.py`) and 9 (`graphrag index`) must be run by the user in a new terminal.

## 11. Financial Generalization

### 11.1 Why the architecture extends to live financial data

The current pipeline operates on a static Wikipedia-style corpus of 9,943 articles (數學 + 經濟金融). The same three-layer architecture generalizes to live financial data with minor adaptations.

### 11.2 Data source integration

| Data Source | Format | Integration approach |
|-------------|--------|----------------------|
| **Bloomberg / Reuters** | Structured tickers + 中文新闻 | Append as new domain; entity extraction targets instruments (AAPL, TSMC) and macro events; relations ground in price/action semantics |
| **Earnings reports (EDGAR / 公開資訊觀測站)** | XBRL / PDF text | Chunk at section level (Management Discussion, Financial Statements); attach company/ticker as metadata; L3 entity types expand to include 公司, 財務指標, 營收 |
| **ETF / Futures / Options chains** | Structured tables (ISIN, expiry, strike, greeks) | Represent as typed entities (ETF name, contract month, strike price) with relations like `追蹤-index-of`, `履約價-underlies`, `delta-hedge-of`; populate L3 schema |
| **News headlines** | Timestamped short text | Temporal PMI variant: rolling-window PMI (daily/weekly) to detect concept-level topic shifts; add time dimension to L1 edges |

### 11.3 Key architectural changes for live data

1. **Temporal awareness**: L1 currently uses static file-level co-occurrence. For live data, add a timestamp column to `docs.csv` and compute time-windowed PMI (e.g., weekly concept co-occurrence matrices). Directional PMI on financial news can approximate Granger-causal concept flow.

2. **Entity typing expansion**: L3's fixed 6-relation schema is sufficient for the Wikipedia corpus. For financial data, extend entity types to `[金融商品, 公司, 指標, 法規, 人物, 事件]` and relation types to `[公告, 公布, 受影響, 追蹤, 履約, 對沖]` to capture the specific semantics of financial documents.

3. **Cross-source file edges**: L2 K-NN edges currently span only the Wikipedia corpus. With multiple sources, cross-source similarity becomes meaningful — e.g., an earnings call transcript of TSMC vs. a Bloomberg macro article on semiconductor outlook — enabling a cross-source document graph bridging earnings, news, and equity research.

4. **Sampling strategy adjustment**: Seed-of-seed sampling (currently PMI anchor concepts) would be replaced by stratified sampling over sectors (半導體, 金融, 製藥) or event types (財報發布, 央行會議, 併購) to ensure coverage of economically significant segments.

5. **LLM cost management**: Live financial data at scale (e.g., 10,000 Bloomberg articles/day) would make L3 GraphRAG prohibitively expensive. Consider a hybrid: L1 PMI on daily batch, L2 BGE-M3 K-NN on daily batch, L3 GraphRAG only on high-signal events (earnings surprises, Fed decisions) identified via L1/L2 anomaly scoring.

### 11.4 What stays the same

- **L1 (PMI)**: No architecture change; retrain on new corpus with domain-specific stopwords (e.g., "公布", "本報告", "根據" for financial text).
- **L2 (BGE-M3)**: Model unchanged; re-embed new documents on the same BGE-M3 infrastructure.
- **L3 (GraphRAG)**: Model unchanged; re-run on new sample with updated entity/prompt configuration for financial domain.
- **Fusion**: Same NetworkX fusion strategy; only node/edge schema updates.

---

## 12. Execution Order

```
0. uv sync
   ⚠️ bash setup_cuda_torch.sh          # run after every uv sync
1. uv run python src/preprocess.py      # ~1 min → docs.csv
2. uv run python src/embed_files.py    # ⚠️ ~80 min on GPU (new terminal)
3. uv run python src/build_knn.py       # <1 min → knn_graph.parquet
4. uv run python src/build_pmi.py       # ~3 min → pmi_graph.parquet
4b. uv run python src/build_directional_pmi.py  # ~3 min → directional_pmi.parquet
5. uv run python src/sample_100.py       # <1 min → sample_100.csv
6. uv run graphrag init --root .\graphrag_project  # one-off
7. uv run graphrag prompt-tune ...      # one-off, ~5 min
8. Manually edit prompts/entity_extraction.txt to lock 6-relation schema
9. uv run graphrag index --root .\graphrag_project  # ⚠️ ~45 min, ~US$0.05 (new terminal)
10. uv run python src/fuse_graph.py     # ~5 min → unified_graph.pkl
11. uv run python src/eval.py           # <1 min → reports/eval_sample.md
```

**⚠️ Long-running / paid steps must NOT be auto-executed from Claude Code.** Steps 2 (`embed_files.py`) and 9 (`graphrag index`) must be run by the user in a new terminal.

### Verification Checkpoints

| Step | Verification |
|------|-------------|
| L0 preprocessed | `processed/docs.csv` has 9,943 rows; `domain.value_counts() == {數學: 2523, 經濟金融: 7420}` |
| L2 embeddings | `analysis/embeddings.parquet` has 9,943 rows; each `embedding` is length 1024 |
| K-NN sanity | 期望值 (expectation) article's nearest neighbors include other math/statistics articles |
| L1 PMI sanity | Top PMI neighbors of 期望值 include: 隨機變數, 變異數, 機率分布, 條件期望 |
| L3 sampler | `processed/sample_100.csv` has 100 rows; contains all 9 hard seeds |
| L3 graphrag | `entities.parquet` non-empty; entity names are Traditional Chinese; relations ⊂ {is-a, part-of, defines, special-case-of, used-in, related-to} |
| L3 query | `uv run graphrag query --root .\graphrag_project --method local "什麼是期望值?"` returns a grounded Chinese answer citing files from `sample_100.csv` |
| Fusion | `unified_graph.pkl` loads; `type='file'` node count == 9,943; concept nodes include both `source='graphrag'` and `source='pmi'` |
| Cross-domain bridge | 期望值's L3 source docs span both folders; one of its L3 communities has mixed `domain_distribution` |
| Eval | `reports/eval_sample.md` renders; 35 rows populated (10 L3 KG + 10 L2 K-NN + 10 cross-domain + 5 L1 PMI) |
