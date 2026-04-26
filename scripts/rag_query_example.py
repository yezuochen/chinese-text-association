"""
RAG Query Example: 甚麼是通貨膨脹
Demonstrates retrieval → LLM answering using NIM free API.
L1 finds concept-bearing docs, L2 finds similar docs, context is packed into a
system+user prompt and sent to qwen/qwen3.5-397b-a17b via NVIDIA NIM.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports"

# ── Load data ──────────────────────────────────────────────────────────────
pmi = pd.read_parquet(ROOT / "analysis" / "pmi_graph.parquet")
import pyarrow.parquet as pq
dom = pq.read_table(ROOT / "analysis" / "concept_domains.parquet").to_pandas()
dom.columns = ["concept", "doc_freq", "docs_數學", "docs_經濟金融", "cross_domain_count"]

knn = pd.read_parquet(ROOT / "analysis" / "knn_graph.parquet")
knn["src_int"] = knn["src"].astype(int)
knn["dst_int"] = knn["dst"].astype(int)
docs = pd.read_csv(ROOT / "processed" / "docs.csv", keep_default_na=False, dtype={"id": str})
docs["id_int"] = docs["id"].astype(int)

ent = pd.read_parquet(ROOT / "graphrag_project" / "output" / "entities.parquet")
rel = pd.read_parquet(ROOT / "graphrag_project" / "output" / "relationships.parquet")

# ── Config ─────────────────────────────────────────────────────────────────
USER_QUERY = "甚麼是通貨膨脹"
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

# ── L1: PMI — documents containing the concept ─────────────────────────────
QUERY_CONCEPT = "通貨膨脹"
concept_docs = docs[docs["text"].str.contains(QUERY_CONCEPT, regex=False)]
print(f"[L1] Docs containing '{QUERY_CONCEPT}': {len(concept_docs)}")

# PMI neighbors
pmi_neighbors = pmi[
    (pmi["src"] == QUERY_CONCEPT) | (pmi["dst"] == QUERY_CONCEPT)
].copy()
pmi_neighbors["concept_other"] = pmi_neighbors.apply(
    lambda r: r["dst"] if r["src"] == QUERY_CONCEPT else r["src"], axis=1
)
top_neighbors = pmi_neighbors.nlargest(6, "pmi")[["concept_other", "pmi", "co_doc_count"]]

# Get a sample document as representative text
sample_doc = concept_docs.sample(1, random_state=7).iloc[0]
sample_text = sample_doc["text"].replace("\n", " ").strip()[:300]

# Build L1 context string
l1_context = (
    f"L1 (PMI Concept Graph): '{QUERY_CONCEPT}' appears in {len(concept_docs)} documents. "
    f"Top PMI-co-occurring concepts: {', '.join(top_neighbors['concept_other'].tolist())}. "
    f"Example document excerpt: {sample_text}... "
    f"(Document title: {sample_doc['title']}, domain: {sample_doc['domain']})"
)
print(f"[L1] Context built")

# ── L2: K-NN — similar documents ─────────────────────────────────────────
concept_ids = set(concept_docs["id_int"].tolist())
knn_from_c = knn[knn["src_int"].isin(concept_ids)].copy()
knn_from_c = knn_from_c[knn_from_c["src"] != knn_from_c["dst"]]
knn_from_c = knn_from_c.merge(
    docs[["id_int", "title", "domain", "text"]],
    left_on="dst_int", right_on="id_int"
).drop("id_int", axis=1).rename(columns={"title": "n_title", "domain": "n_domain", "text": "n_text"})

top_knn = knn_from_c.nlargest(5, "cos_sim")
l2_context_parts = []
for _, row in top_knn.iterrows():
    excerpt = row["n_text"].replace("\n", " ").strip()[:150]
    l2_context_parts.append(f"- \"{row['n_title']}\" (domain: {row['n_domain']}, cos={row['cos_sim']:.4f}): {excerpt}...")

l2_context = (
    f"L2 (K-NN Document Graph): {len(top_knn)} most similar documents to '{QUERY_CONCEPT}'-bearing docs:\n"
    + "\n".join(l2_context_parts)
)
print(f"[L2] Context built ({len(top_knn)} neighbors)")

# ── L3: GraphRAG — related entities ───────────────────────────────────────
priority_types = ["金融機構", "經濟概念", "金融工具", "金融指標", "中央銀行", "基礎設施"]
related_entities = pd.DataFrame()
for ptype in priority_types:
    subset = ent[ent["type"] == ptype].nlargest(4, "degree")
    if len(subset) > 0:
        related_entities = pd.concat([related_entities, subset])
        if len(related_entities) >= 6:
            break
if len(related_entities) == 0:
    related_entities = ent.nlargest(6, "degree")

related_entity_ids = set(related_entities["id"].tolist())
related_rels = rel[rel["source"].isin(related_entity_ids) | rel["target"].isin(related_entity_ids)].nlargest(5, "weight")

id_to_title_ent = dict(zip(ent["id"], ent["title"]))
entity_names = related_entities["title"].tolist()
rel_triples = []
for _, r in related_rels.iterrows():
    src_t = id_to_title_ent.get(r["source"], "?")
    tgt_t = id_to_title_ent.get(r["target"], "?")
    rel_triples.append(f"{src_t} --[{r['description']}]--> {tgt_t}")

l3_context = (
    f"L3 (GraphRAG Knowledge Graph): Related entities: {', '.join(entity_names)}. "
    f"Key relations: {'; '.join(rel_triples)}"
)
print(f"[L3] Context built")

# ── Build system prompt and user prompt ────────────────────────────────────
SYSTEM_PROMPT = """你是一位專業的學術助理，專精於經濟學與數學領域。請根據以下檢索到的上下文（來自三層關聯分析系統：L1=PMI概念圖, L2=K-NN文件圖, L3=GraphRAG知識圖譜）回答用戶的問題。

規則：
1. 只使用上下文中提供的資訊回答，不要捏造內容。
2. 如果上下文不足以回答，請明確說明「根據檢索結果，無法完整回答此問題」。
3. 回答請使用繁體中文，技術術語可以保留英文。
4. 請以結構化方式回答（可使用列表或段落）。

上下文：
{l1}
{l2}
{l3}
""".format(l1=l1_context, l2=l2_context, l3=l3_context)

USER_PROMPT = f"用戶問題：{USER_QUERY}\n\n請根據以上上下文回答。"

print("\n" + "="*60)
print("SYSTEM PROMPT:")
print("="*60)
print(SYSTEM_PROMPT[:500] + "...")
print("\n" + "="*60)
print("USER PROMPT:")
print("="*60)
print(USER_PROMPT)

# ── Call NIM API ────────────────────────────────────────────────────────────
print("\n[Calling NIM API]")
try:
    client = load_nim_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        max_tokens=400,
        temperature=0.3,
        timeout=60,
    )
    answer = response.choices[0].message.content
    print(f"\n[LLM Answer]\n{answer}")

    result = {
        "user_query": USER_QUERY,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "l1_context": l1_context,
        "l2_context": l2_context,
        "l3_context": l3_context,
        "answer": answer,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    }

except Exception as e:
    print(f"[ERROR] NIM API call failed: {e}")
    result = {
        "user_query": USER_QUERY,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "l1_context": l1_context,
        "l2_context": l2_context,
        "l3_context": l3_context,
        "error": str(e)
    }

out_path = OUT / "rag_query_output.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_path}")