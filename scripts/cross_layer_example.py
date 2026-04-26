"""
Cross-Layer RAG Example: Tracing a Query through L1 → L2 → L3
Demonstrates how to retrieve and synthesize information using the three-layer pipeline.
Uses NIM free API (qwen/qwen3.5-397b-a17b) to answer a question about the domain-set concept.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from openai import OpenAI
import re

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports"

# ── Load data ──────────────────────────────────────────────────────────────
pmi = pd.read_parquet(ROOT / "analysis" / "pmi_graph.parquet")
dom = pd.read_parquet(ROOT / "analysis" / "concept_domains.parquet")
dom.columns = ["concept", "doc_freq", "docs_數學", "docs_經濟金融", "cross_domain_count"]

knn = pd.read_parquet(ROOT / "analysis" / "knn_graph.parquet")
knn["src_int"] = knn["src"].astype(int)
knn["dst_int"] = knn["dst"].astype(int)
docs = pd.read_csv(ROOT / "processed" / "docs.csv", keep_default_na=False, dtype={"id": str})
docs["id_int"] = docs["id"].astype(int)

emb = pd.read_parquet(ROOT / "analysis" / "embeddings.parquet")

ent = pd.read_parquet(ROOT / "graphrag_project" / "output" / "entities.parquet")
rel = pd.read_parquet(ROOT / "graphrag_project" / "output" / "relationships.parquet")

# ── Config ─────────────────────────────────────────────────────────────────
QUERY_CONCEPT = "通貨膨脹"
MODEL = "qwen/qwen3.5-397b-a17b"

def load_nim_client():
    config = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("NIM_"):
                key, _, value = line.partition("=")
                config[key] = value.strip()
    api_key = config.get("NIM_API_KEY", "")
    base_url = config.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)

# ── L1: PMI — find documents containing the concept ─────────────────────────
print("=" * 60)
print(f"CROSS-LAYER RAG EXAMPLE — Query: {QUERY_CONCEPT}")
print("=" * 60)

# Find all documents containing the concept (via co-doc count with concept itself)
concept_docs = docs[docs["text"].str.contains(QUERY_CONCEPT, regex=False)]
print(f"\n[L1] Documents containing '{QUERY_CONCEPT}': {len(concept_docs)}")

# Get PMI-co-occurring concepts in same documents
concept_texts = concept_docs["text"].tolist()
vocab_in_docs = set()
for text in concept_texts:
    import jieba
    for word in jieba.lcut(text):
        vocab_in_docs.add(word)

# PMI neighbors of the concept
pmi_neighbors = pmi[
    (pmi["src"] == QUERY_CONCEPT) | (pmi["dst"] == QUERY_CONCEPT)
].copy()
pmi_neighbors = pmi_neighbors[pmi_neighbors["pmi"] >= 2.0]
pmi_neighbors["concept_other"] = pmi_neighbors.apply(
    lambda r: r["dst"] if r["src"] == QUERY_CONCEPT else r["src"], axis=1
)
top_neighbors = pmi_neighbors.nlargest(8, "pmi")[["concept_other", "pmi", "co_doc_count"]]
print(f"[L1] Top PMI neighbors of '{QUERY_CONCEPT}':")
for _, row in top_neighbors.iterrows():
    print(f"  - {row['concept_other']} (PMI={row['pmi']:.2f}, co-doc={row['co_doc_count']})")

l1_info = {
    "query": QUERY_CONCEPT,
    "doc_count": len(concept_docs),
    "top_neighbors": top_neighbors.to_dict("records")
}

# ── L2: K-NN — find similar documents ───────────────────────────────────────
# Get doc IDs of concept-containing docs
concept_ids = set(concept_docs["id_int"].tolist())

# Find K-NN edges where src is in concept_ids
knn_from_concept = knn[knn["src_int"].isin(concept_ids)].copy()
knn_from_concept = knn_from_concept[knn_from_concept["src"] != knn_from_concept["dst"]]
knn_from_concept = knn_from_concept.merge(
    docs[["id_int", "title", "domain"]], left_on="dst_int", right_on="id_int"
).drop("id_int", axis=1).rename(columns={"title": "neighbor_title", "domain": "neighbor_domain"})

top_knn = knn_from_concept.nlargest(5, "cos_sim")[["neighbor_title", "neighbor_domain", "cos_sim"]]
print(f"\n[L2] Top K-NN neighbors of concept-containing documents:")
for _, row in top_knn.iterrows():
    print(f"  - {row['neighbor_title'][:40]} ({row['neighbor_domain']}) cos={row['cos_sim']:.4f}")

l2_info = {"top_neighbors": top_knn.to_dict("records")}

# ── L3: GraphRAG entities around the concept ───────────────────────────────
# Search entities whose title or description contains the concept
related_entities = ent[
    ent["title"].str.contains(QUERY_CONCEPT, regex=False) |
    ent["description"].str.contains(QUERY_CONCEPT, regex=False)
]
if len(related_entities) == 0:
    # Fall back to entities of type "金融機構" or "經濟概念" sorted by degree
    priority_types = ["金融機構", "經濟概念", "金融工具", "金融指標", "中央銀行", "基礎設施"]
    for ptype in priority_types:
        subset = ent[ent["type"] == ptype].nlargest(5, "degree")
        if len(subset) > 0:
            related_entities = subset
            break
    if len(related_entities) == 0:
        related_entities = ent.nlargest(5, "degree")

related_entity_ids = set(related_entities["id"].tolist())
related_relations = rel[
    rel["source"].isin(related_entity_ids) | rel["target"].isin(related_entity_ids)
].nlargest(5, "weight")

print(f"\n[L3] GraphRAG entities related to '{QUERY_CONCEPT}': {len(related_entities)}")
for _, e in related_entities.head(5).iterrows():
    print(f"  - {e['title']} ({e['type']}) degree={e['degree']}")

print(f"[L3] GraphRAG relations:")
for _, r in related_relations.iterrows():
    s = ent[ent["id"] == r["source"]]["title"].values
    t = ent[ent["id"] == r["target"]]["title"].values
    src_t = s[0] if len(s) else r["source"][:20]
    tgt_t = t[0] if len(t) else r["target"][:20]
    print(f"  - {src_t} --[{r['description']}]--> {tgt_t} (weight={r['weight']:.3f})")

l3_info = {
    "entities": related_entities[["title", "type", "degree"]].head(5).to_dict("records"),
    "relations": related_relations[["source", "target", "description", "weight"]].head(5).to_dict("records")
}

# ── Synthesize: Build context for RAG ───────────────────────────────────────
context_parts = []

# L1 context
if len(concept_docs) > 0:
    sample_doc = concept_docs.sample(1, random_state=42).iloc[0]
    context_parts.append(
        f"[L1 Concept Co-occurrence] '{QUERY_CONCEPT}' appears in {len(concept_docs)} documents. "
        f"Example document: {sample_doc['title']} (domain: {sample_doc['domain']}). "
        f"Top PMI-co-occurring concepts: {', '.join(top_neighbors['concept_other'].tolist())}."
    )

# L2 context
if len(top_knn) > 0:
    neighbor_titles = top_knn["neighbor_title"].tolist()
    context_parts.append(
        f"[L2 Document Similarity] Documents similar to '{QUERY_CONCEPT}'-containing docs include: "
        f"{', '.join([t[:30] for t in neighbor_titles])}."
    )

# L3 context
if len(related_entities) > 0:
    entity_names = related_entities["title"].tolist()
    context_parts.append(
        f"[L3 Knowledge Graph] Related entities: {', '.join(entity_names[:5])}."
    )
    if len(related_relations) > 0:
        rel_triples = []
        for _, r in related_relations.head(3).iterrows():
            src_t = ent[ent["id"] == r["source"]]["title"].values
            tgt_t = ent[ent["id"] == r["target"]]["title"].values
            src_s = src_t[0] if len(src_t) else "?"
            tgt_s = tgt_t[0] if len(tgt_t) else "?"
            rel_triples.append(f"{src_s} --[{r['description']}]--> {tgt_s}")
        context_parts.append(f"Key relations: {'; '.join(rel_triples)}")

context = "\n\n".join(context_parts)
print(f"\n[SYNTHESIZED CONTEXT]\n{context[:500]}...")

# ── Call NIM API to answer a question ───────────────────────────────────────
rag_question = f"請根據以下上下文，說明「{QUERY_CONCEPT}」的概念，並舉例它與哪些領域相關。"

rag_prompt = f"""你是一個專業的學術助理。根據以下三層檢索上下文回答問題。

Context:
{context}

Question: {rag_question}

Answer:"""

print(f"\n[Calling NIM API: {MODEL}]")
print(f"Prompt: {rag_prompt[:200]}...")

try:
    client = load_nim_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": rag_prompt}],
        max_tokens=300,
        temperature=0.3,
        timeout=60,
    )
    answer = response.choices[0].message.content
    print(f"\n[LLM Answer]\n{answer}")

    # Save output
    result = {
        "query": QUERY_CONCEPT,
        "l1_info": l1_info,
        "l2_info": l2_info,
        "l3_info": l3_info,
        "context": context,
        "answer": answer,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    }

    out_path = OUT / "rag_example_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")

except Exception as e:
    print(f"[ERROR] NIM API call failed: {e}")
    result = {
        "query": QUERY_CONCEPT,
        "l1_info": l1_info,
        "l2_info": l2_info,
        "l3_info": l3_info,
        "context": context,
        "error": str(e)
    }
    out_path = OUT / "rag_example_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved partial output: {out_path}")