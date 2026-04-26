# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Chinese text **association-analysis pipeline** over a corpus of ~9,943 Wikipedia-style articles in two domains (`數學`, `經濟金融`). The goal is to produce two kinds of edges:

- **concept ↔ concept** — a typed knowledge graph (`is-a`, `part-of`, `defines`, `special-case-of`, `used-in`, `related-to`), plus an untyped PMI co-occurrence graph for coverage outside the sample
- **file ↔ file** — a K-NN graph over whole-document embeddings

These are fused into a single NetworkX graph with community-overlap weights, and queried via small similarity APIs. There is no user-facing app — the deliverables are data artifacts plus eval notes.

The pipeline is organized into **three independent layers**:

- **L1 — PMI concept graph** (full 10K, free, local CPU, untyped)
- **L2 — BGE-M3 K-NN file graph** (full 10K, free, local GPU)
- **L3 — GraphRAG demo on a concept-seeded 100-file sample** (~US$0.05 OpenAI embeddings + NIM free chat)

## Source of Truth

**Read these first; they drive every decision:**

- [MEMO.md](MEMO.md) — technique selection memo. Explains **why** each technique was chosen; records rejected alternatives.
- [PLAN.md](PLAN.md) — coding plan. Explains **what** to build, **in what order**, with module-by-module specs and verification steps.
- [DESIGN.md](DESIGN.md) — algorithm design document. Detailed technique specs, method trade-offs, and output artifact map. Supplements MEMO and PLAN.
- [CUDA_TORCH_UV_NOTE.md](CUDA_TORCH_UV_NOTE.md) — PyTorch CUDA installation note. Documents why `setup_cuda_torch.sh` is needed and how to run it after `uv sync`.

If CLAUDE.md disagrees with MEMO/PLAN, MEMO/PLAN wins. Update them first, then reflect the change here.

## Architecture at a Glance

Three independent computations feed one fused graph:

```
data/ (immutable)
    │
    ▼
processed/docs.csv          ← OpenCC s2tw + regex scrub + folder→domain metadata
    │
    ├────────────────┬──────────────────────┬────────────────────────┐
    ▼                ▼                      ▼                        ▼
L1: jieba + PMI   L2: BGE-M3 embed   sample_100.py (data-driven seeds) │
 (full 10K)       (FP16, in-proc)    ▼                               │
    │                │               processed/sample_100.csv        │
    ▼                ▼                      │                        │
pmi_graph.parquet  embeddings.parquet       ▼                        │
+ concept_domains.parquet  + knn_graph.parquet  L3: graphrag index    │
                                        (DeepSeek-V3 via NIM for chat;│
                                         text-embedding-3-small via  │
                                         OpenAI for embed)           │
                                              │                      │
                                              ▼                      │
                                     graphrag_project/output/*.parquet
                                              │
        ┌─────────────────────────────────────┴──────────────┐
        ▼                                                    ▼
                  analysis/unified_graph.pkl
       (NetworkX; L2 file edges weighted 0.7·cosine + 0.3·community_overlap;
        L1 untyped + L3 typed concept edges; ~9,943 file nodes + concept nodes)
```

**Non-obvious architectural invariants:**

- **Two disjoint embedding spaces, on purpose.** L2 uses local BGE-M3 (in-process via `FlagEmbedding`). L3 uses paid OpenAI `text-embedding-3-small` for GraphRAG's internal retrieval only. The earlier "shared BGE-M3 via a local `infinity` server" invariant was dropped in MEMO rev. 3 — it added a local-server dependency and didn't solve the `encoding_format: null` bug. Do not re-introduce a local embedding server. Do not compare a BGE-M3 vector to an OpenAI vector directly.
- **GraphRAG only runs on the 100-file sample.** Not the full 10K — that was too expensive on paid APIs and too slow on NIM free. `sample_100.csv` is data-driven seeded: seeds are PMI anchor-neighbor cross-domain concepts (top-8 by anchor-PMI sum from 10 math + 10 econ anchor terms), 50/50 stratified by domain. Full-corpus concept coverage is the job of L1 (PMI).
- **`1 file = 1 chunk` only applies to L2's file-level embedding.** GraphRAG auto-chunks the 100 sample files internally (~1200 tokens per chunk). Do not force GraphRAG to treat whole files as chunks.
- **DeepSeek-V3 runs via NVIDIA NIM's OpenAI-compatible endpoint.** GraphRAG's `llm.api_base` points at NIM. Budget cap comes from NIM's free 40 RPM; GraphRAG handles throttling + retries itself (`requests_per_minute: 35`, `max_retries: 10`, `tenacity` backoff). Do not add a custom rate limiter.
- **`encoding_format: null` bug is sidestepped, not patched.** `fnllm` sends `encoding_format: null` on embedding calls; NIM rejects it. Rev. 3 routes L3 embeddings to OpenAI instead — OpenAI accepts `null`. Do not add a proxy/shim to "fix" NIM embeddings.
- **Preprocessing guardrails (revised after L2 sanity).** Initially none (intentional per user direction); revised after `embed_files.py` sanity check exposed embedding collapse on short stubs (cos_sim=1.0000 across math/econ cross-domain pairs). Now: `char_count < 50` or `unique_token_count < 5` → skip. Skipped docs tracked in `filtered_reason` column of `processed/docs.csv`. Regenerate with `uv run python src/preprocess.py`.
- **Folder name is document metadata, not a concept node.** `domain: 數學` or `domain: 經濟金融` is attached per document; entity / community / PMI-concept domain attributes are *computed* by aggregation, never hand-coded.
- **All text is normalized to Traditional Chinese** (Taiwan standard via OpenCC `s2tw`) during preprocessing. Downstream code can assume Traditional.

## Data Invariants

- `data/` is **immutable**. Never write to it. If you need to transform, write to `processed/`.
- `processed/` is regenerable — running `src/preprocess.py` from scratch must reproduce it.
- `graphrag_project/output/` and `analysis/` are build artifacts — safe to delete and rebuild.
- The source corpus is also archived at `關聯分析文本資料.zip` — never unpack it into `data/` again.

## Package Management

This project uses **`uv`** (not pip / poetry / conda). Dependencies live in `pyproject.toml`. Always run scripts via `uv run …` so the right venv is picked.

```powershell
uv sync                                         # install everything from pyproject.toml
uv add <package>                                # add a dep + lock
uv run python src/<script>.py                   # run a script
uv run graphrag index --root .\graphrag_project # run graphrag CLI
```

## ⚠️ Long-Running & Paid Operations — Do NOT Auto-Execute

**Never auto-run the following from Claude Code.** These are slow, costly, or both — running them here risks losing progress on disconnect and consuming resources without explicit user oversight. **Always announce the command and instruct the user to run it in a new terminal.**

| Operation | Script / Command | Duration | Cost | What to Say |
|---|---|---|---|---|
| L2: Full-corpus BGE-M3 embeddings | `uv run python src/embed_files.py` | ~80 min | free (local GPU) | Announce: "Run in a new terminal: `uv run python src/embed_files.py`" |
| L3: GraphRAG index on 100-file sample | `uv run graphrag index --root .\graphrag_project` | ~45 min | ~US$0.05 OpenAI | Announce: "Run in a new terminal: `uv run graphrag index --root .\graphrag_project`" |

**Why this matters:**
- A long-running process started in Claude Code will be terminated if the session disconnects.
- Paid operations (even ~US$0.05) should only run when the user explicitly decides to run them.
- Breaking these into terminal-managed background jobs also keeps Claude Code responsive for follow-up tasks.

All other steps (`preprocess.py`, `build_knn.py`, `build_pmi.py`, `sample_100.py`, `fuse_graph.py`, `eval.py`) are fast enough to run via `uv run` directly.

## Common Commands (in execution order)

Full pipeline from a clean checkout:

```powershell
# 0. One-time setup
uv sync
copy .env.example .env         # then fill in NIM_API_KEY + OPENAI_API_KEY

# 1. Preprocess → processed/docs.csv
uv run python src/preprocess.py

# 2. L2 — full-corpus BGE-M3 embeddings + K-NN  (local GPU, no server)
#    ⚠️ ~80 min — run in a NEW TERMINAL:
uv run python src/embed_files.py        # ~80 min on GPU
uv run python src src/build_knn.py          # <1 min

# 3. L1 — PMI concept graph over full corpus  (local CPU)
uv run python src/build_pmi.py          # ~3 min

# 4. L3a — concept-seeded 100-file sample for GraphRAG
uv run python src/sample_100.py         # <1 min → processed/sample_100.csv

# 5. One-off GraphRAG setup
uv run graphrag init --root .\graphrag_project
uv run graphrag prompt-tune --root .\graphrag_project `
  --domain "Chinese wiki articles in math and economics" `
  --language "Traditional Chinese" `
  --output .\graphrag_project\prompts
# then manually edit prompts/entity_extraction.txt to lock the 6-relation taxonomy

# 6. L3b — run GraphRAG on the 100-file sample
#    ⚠️ ~45 min, ~US$0.05 OpenAI — run in a NEW TERMINAL:
uv run graphrag index --root .\graphrag_project

# 7. Fuse L1 + L2 + L3 graphs + evaluate
uv run python src/fuse_graph.py
uv run python src/eval.py               # writes reports/eval_sample.md
```

> **⚠️ Long-running / paid steps (steps 2 & 6):** Do NOT run these from Claude Code. Announce the command and instruct the user to run it in a new terminal. Claude Code should not auto-execute `embed_files.py` or `graphrag index`. See table above for details.

No separate GraphRAG smoke-test step — the 100-file sample run *is* the smoke test. If outputs look wrong (English entities, wrong relation types), adjust `prompts/entity_extraction.txt` and re-run. No local embedding server to start / stop — BGE-M3 (L2) runs in-process; GraphRAG (L3) calls cloud endpoints directly.

## Hard Constraints

- **GPU:** RTX 3050, 6 GB VRAM. BGE-M3 at FP16 + `batch_size=4` + `max_length=8192` peaks ~4.5 GB; drop `max_length` if OOM.
- **LLM rate limit:** NIM free tier is **40 RPM**. `settings.yaml` must keep `requests_per_minute` ≤ 35.
- **OpenAI spend:** L3 embeddings are ~US$0.05 for 100 files on `text-embedding-3-small`. Cap alerts on the OpenAI account if running repeatedly.
- **Evaluation is qualitative.** No gold labels. `src/eval.py` samples 35 edges (10 L3 KG, 10 L2 K-NN, 10 cross-domain, 5 L1 PMI) into a markdown report for manual inspection — that is the eval.

## Outer Context

This project lives inside a personal job-search workspace; see [../../../CLAUDE.md](../../../CLAUDE.md) at the JOBS root for the candidate profile and the resume template used elsewhere in the workspace. Nothing in the outer CLAUDE.md applies to this pipeline — the two scopes are independent.

# Coding Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.