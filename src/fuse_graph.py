"""
Fusion: L1 + L2 + L3 → unified_graph.pkl

Input:
  analysis/pmi_graph.parquet         — L1 untyped concept edges
  analysis/knn_graph.parquet         — L2 file-file edges (cosine)
  analysis/concept_domains.parquet   — L1 concept domain distributions
  graphrag_project/output/documents.parquet   — L3 doc metadata (raw_data[·"id"] = original doc id)
  graphrag_project/output/entities.parquet     — L3 entities (uuid id)
  graphrag_project/output/relationships.parquet — L3 edges
  graphrag_project/output/communities.parquet  — L3 communities
  graphrag_project/output/text_units.parquet   — L3 text units (links entities ↔ docs)
  processed/docs.csv                 — all 9,943 docs (domain, title)
  processed/sample_100.csv           — L3-covered file ids

Output:
  analysis/unified_graph.pkl  — NetworkX graph

Edge weighting (L2 file-file):
  community_overlap = Jaccard(doc_to_communities[src], doc_to_communities[dst])
  weight = 0.7 · cos_sim_normalized + 0.3 · community_overlap
"""

from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import networkx as nx
import pickle

ROOT = Path(__file__).resolve().parent.parent

print("Loading inputs...")

docs = pd.read_csv(ROOT / "processed" / "docs.csv", dtype={"id": str}, keep_default_na=False)
l3_docs = pd.read_parquet(ROOT / "graphrag_project" / "output" / "documents.parquet")
l3_entities = pd.read_parquet(ROOT / "graphrag_project" / "output" / "entities.parquet")
l3_relations = pd.read_parquet(ROOT / "graphrag_project" / "output" / "relationships.parquet")
l3_communities = pd.read_parquet(ROOT / "graphrag_project" / "output" / "communities.parquet")
l3_text_units = pd.read_parquet(ROOT / "graphrag_project" / "output" / "text_units.parquet")
l2_knn = pd.read_parquet(ROOT / "analysis" / "knn_graph.parquet")
l1_pmi = pd.read_parquet(ROOT / "analysis" / "pmi_graph.parquet")
l1_domains = pd.read_parquet(ROOT / "analysis" / "concept_domains.parquet")
l1_domains.columns = ["concept", "doc_freq", "docs_數學", "docs_經濟金融", "cross_domain_count"]

print(f"  docs: {len(docs):,}")
print(f"  L3 entities: {len(l3_entities)}")
print(f"  L3 relationships: {len(l3_relations)}")
print(f"  L3 communities: {len(l3_communities)}")
print(f"  L3 text_units: {len(l3_text_units)}")

# ── Build hash→original-doc-id mapping ─────────────────────────────────────────
# GraphRAG replaces CSV "id" with a hash uuid in documents.parquet.
# We recover the original id from raw_data['﻿"id"'] (BOM-prefixed key).
hash_to_doc_id: dict[str, str] = {}
for _, row in l3_docs.iterrows():
    raw = row["raw_data"]
    if isinstance(raw, dict):
        id_keys = [k for k in raw.keys() if "id" in k.lower()]
        if id_keys:
            orig_id = str(raw[id_keys[0]])
            hash_to_doc_id[str(row["id"])] = orig_id

print(f"  hash→doc_id mappings: {len(hash_to_doc_id)}")

# ── Build entity_id → set of community ids ─────────────────────────────────────
# entity_ids in communities.parquet is numpy.ndarray of uuid strings
entity_communities: dict[str, set[int]] = defaultdict(set)
for _, row in l3_communities.iterrows():
    cid = int(row["community"])
    eids = row.get("entity_ids")
    if eids is not None and len(eids) > 0:
        for e in eids:
            entity_communities[str(e)].add(cid)

print(f"  entities with community membership: {len(entity_communities)}")

# ── Build doc_id (original) → set of community ids ─────────────────────────────
# text_units.document_id is the graphrag hash; map to original doc id
# text_units.entity_ids is numpy.ndarray of entity uuids
doc_to_communities: dict[str, set[int]] = defaultdict(set)
for _, row in l3_text_units.iterrows():
    hash_doc_id = str(row["document_id"])
    orig_doc_id = hash_to_doc_id.get(hash_doc_id, hash_doc_id)
    eids = row.get("entity_ids")
    if eids is not None and len(eids) > 0:
        for e in eids:
            for cid in entity_communities.get(str(e), []):
                doc_to_communities[orig_doc_id].add(cid)

print(f"  docs with community membership: {len(doc_to_communities)}")

# ── Build text_unit_id (hash) → original doc id ─────────────────────────────────
tu_doc: dict[str, str] = {}
for _, row in l3_text_units.iterrows():
    hash_doc = str(row["document_id"])
    orig_doc = hash_to_doc_id.get(hash_doc, hash_doc)
    tu_doc[str(row["id"])] = orig_doc

# ── Build graph ────────────────────────────────────────────────────────────────

print("Building unified graph...")

G = nx.Graph()

# File nodes (all 9,943 from docs.csv)
for _, row in docs.iterrows():
    G.add_node(row["id"], type="file", domain=row["domain"], title=row["title"])

# L3 concept nodes
l3_concept_ids = set()
for _, row in l3_entities.iterrows():
    nid = str(row["id"])
    l3_concept_ids.add(nid)
    G.add_node(
        nid,
        type="concept",
        source="graphrag",
        entity_type=str(row.get("type", "unknown")),
        title=str(row.get("title", "")),
        description=str(row.get("description", "")),
    )

# L3 typed edges — relationships.parquet source/target are entity titles (not UUIDs).
# We need to resolve them to entity UUIDs via l3_entities['title'] → 'id'.
title_to_entity_id: dict[str, str] = dict(zip(l3_entities["title"].astype(str), l3_entities["id"].astype(str)))

for _, row in l3_relations.iterrows():
    src_title = str(row["source"])
    dst_title = str(row["target"])
    src_id = title_to_entity_id.get(src_title)
    dst_id = title_to_entity_id.get(dst_title)
    if src_id is None or dst_id is None:
        continue
    if src_id not in G or dst_id not in G:
        continue
    G.add_edge(
        src_id, dst_id,
        weight=float(row.get("weight", 1.0)),
        relation="related-to",
        source="graphrag",
        combined_degree=int(row.get("combined_degree", 0)),
    )

# L1-only concept nodes
l1_concept_nodes = set(l1_pmi["src"].unique()) | set(l1_pmi["dst"].unique())
l1_only_nodes = l1_concept_nodes - l3_concept_ids
concept_domain_dict = l1_domains.set_index("concept")

for concept in l1_only_nodes:
    row = concept_domain_dict.get(concept)
    m = int(row["docs_數學"]) if row is not None else 0
    e = int(row["docs_經濟金融"]) if row is not None else 0
    total = m + e
    G.add_node(
        concept,
        type="concept",
        source="pmi",
        domain_數學=m,
        domain_經濟金融=e,
        domain_ratio_math=m / total if total > 0 else 0.5,
    )

def _safe_id(val) -> str:
    """Convert a parquet cell to a clean string node id."""
    if isinstance(val, (int, float)):
        return str(int(val))
    return str(val)


# L1 untyped concept-concept edges
for _, row in l1_pmi.iterrows():
    src, dst = _safe_id(row["src"]), _safe_id(row["dst"])
    if src not in G or dst not in G:
        continue
    if G.has_edge(src, dst):
        continue
    G.add_edge(src, dst, pmi=float(row["pmi"]), co_doc_count=int(row["co_doc_count"]), source="pmi")

# L2 file-file edges
COS_MIN, COS_MAX = 0.43, 1.0

for _, row in l2_knn.iterrows():
    src, dst = _safe_id(row["src"]), _safe_id(row["dst"])
    if src not in G or dst not in G:
        continue

    cos_sim = float(row["cos_sim"])
    src_comm = doc_to_communities.get(src, set())
    dst_comm = doc_to_communities.get(dst, set())

    if src_comm or dst_comm:
        inter = len(src_comm & dst_comm)
        union = len(src_comm | dst_comm)
        community_overlap = inter / union if union > 0 else 0.0
    else:
        community_overlap = 0.0

    cos_norm = (cos_sim - COS_MIN) / (COS_MAX - COS_MIN)
    weight = 0.7 * cos_norm + 0.3 * community_overlap

    G.add_edge(src, dst, cos_sim=cos_sim, community_overlap=community_overlap,
               weight=weight, source="knn")

# L3 concept ↔ file edges
for _, row in l3_entities.iterrows():
    eid = str(row["id"])
    if eid not in G:
        continue
    tu_ids = row.get("text_unit_ids")
    if tu_ids is None or len(tu_ids) == 0:
        continue
    for tu_id in tu_ids:
        doc = tu_doc.get(str(tu_id))
        if doc and doc in G and not G.has_edge(eid, doc):
            G.add_edge(eid, doc, source="graphrag", type="concept_file")

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = ROOT / "analysis" / "unified_graph.pkl"
with open(out_path, "wb") as f:
    pickle.dump(G, f)

print(f"\nUnified graph written to {out_path}")
print(f"  Nodes: {G.number_of_nodes():,}")
print(f"    file nodes:        {sum(1 for n in G.nodes if G.nodes[n].get('type') == 'file'):,}")
print(f"    L3 concept nodes:  {sum(1 for n in G.nodes if G.nodes[n].get('source') == 'graphrag'):,}")
print(f"    L1-only concepts:  {sum(1 for n in G.nodes if G.nodes[n].get('source') == 'pmi'):,}")
print(f"  Edges: {G.number_of_edges():,}")
print(f"    L3 typed edges:    {sum(1 for u, v in G.edges if G.edges[u, v].get('source') == 'graphrag' and G.edges[u, v].get('relation') is not None):,}")
print(f"    L1 untyped edges:  {sum(1 for u, v in G.edges if G.edges[u, v].get('source') == 'pmi'):,}")
print(f"    L2 K-NN file edges:{sum(1 for u, v in G.edges if G.edges[u, v].get('source') == 'knn'):,}")
print(f"    L3 concept-file:   {sum(1 for u, v in G.edges if G.edges[u, v].get('type') == 'concept_file'):,}")