"""
Stage 1: Build PMI concept graph over the full corpus (local CPU, free).

Input: processed/docs.csv
Output:
  - analysis/pmi_graph.parquet     (src, dst, pmi, co_doc_count)
  - analysis/concept_domains.parquet  (concept, doc_freq, domain_counts)

PMI(t1, t2) = log( p(t1,t2) / (p(t1) * p(t2)) )  [probabilities over documents]
Keep: PMI >= 2.0, pair_freq >= 3, doc_freq >= 5.
"""

from pathlib import Path
from collections import defaultdict
from math import log

import pandas as pd
import jieba

ROOT = Path(__file__).resolve().parent.parent
DOCS_IN = ROOT / "processed" / "docs.csv"
PMI_OUT = ROOT / "analysis" / "pmi_graph.parquet"
DOMAIN_OUT = ROOT / "analysis" / "concept_domains.parquet"

STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一個", "上", "也", "很", "到", "說", "要", "去", "你", "會", "著", "一個",
    "對", "這個", "那個", "什麼", "可以", "這樣", "因為", "所以", "但是",
    "或", "而且", "如果", "或者", "則", "而", "於", "為", "之", "以", "及",
    "等", "等等", "其", "此", "彼", "與", "如", "而被", "而被", "被", "把",
    "讓", "使", "由", "所", "又", "再", "才", "還", "已經", "只", "已經",
    "並", "沒有", "不是", "但", "然而", "雖然", "即使", "若是", "假如",
    "除非", "無論", "不論", "不管", "除了", "且", "另", "另外", "此外",
}

# Domain folder names ARE included in PMI vocabulary (not filtered as stopwords).
# However, their extreme document frequency (~15-30% of corpus each) drives
# PMI(t_domain, any_concept) < 2.0 for most pairs (pair_freq required >= 3
# but p(t_domain) is so high that co-occurrence probability doesn't exceed
# the product of marginals by enough). Result: 數學/經濟/金融 appear in
# concept_domains.parquet (doc_freq, domain counts) but have ZERO PMI edges.
# This is fine — they are still USABLE as 'seed of seed' anchors in sample_100.py
# via the anchor-neighbor approach (find cross-domain PMI neighbors of these
# domain-name terms, rank those neighbors by PMI degree, use as seeds).
# STOPWORDS still excludes: wiki, zh, hant, cn (boilerplate leakage).
# 僅適用於 wiki domain，金融語料須移除
STOPWORDS |= {"wiki", "zh", "hant", "cn", "參考", "連結"}

# English tokens that leak into Chinese Wikipedia articles (LaTeX, math notation, titles)
STOPWORDS |= {
    "in", "on", "at", "by", "for", "to", "of", "the", "and", "or", "as", "is",
    "an", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can",
    "with", "from", "that", "this", "these", "those", "it", "its",
    "be", "to", "was", "were", "are", "been",
    "if", "else", "then", "than", "so", "but", "not", "no",
    "all", "each", "every", "any", "some", "such", "only", "more", "most",
    "also", "well", "very", "just", "about", "into", "over", "after",
    "where", "when", "what", "which", "who", "how", "why",
}


def tokenize(text: str) -> set[str]:
    return {
        tok
        for tok in jieba.cut(text)
        if (
            len(tok) >= 2
            and tok.lower() not in STOPWORDS
            and not tok.isdigit()
            # Reject tokens containing any Latin letters (English leakage)
            and not any("a" <= c <= "z" or "A" <= c <= "Z" for c in tok)
        )
    }


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    docs = pd.read_csv(DOCS_IN, keep_default_na=False)
    n = len(docs)
    print(f"Tokenizing {n:,} docs with jieba...")

    # Bag of tokens per doc (with domain)
    doc_tokens: list[tuple[str, set[str]]] = []
    for domain, text in zip(docs["domain"], docs["text"]):
        doc_tokens.append((domain, tokenize(text)))

    # doc_freq per token + domain_counts per token
    doc_freq: dict[str, int] = defaultdict(int)
    domain_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for domain, tokens in doc_tokens:
        for t in tokens:
            doc_freq[t] += 1
            domain_counts[t][domain] += 1

    # Filter to tokens with doc_freq >= 5 AND doc_freq <= 0.30 * n (remove ultra-common stopwords).
    # High-frequency cutoff removes collocation noise (左右/三十/主力) that passes length-based
    # stopword filtering but dominate co-occurrence pairs at PMI >= 2.0.
    # 0.30 threshold: tokens appearing in >30% of docs are generic Chinese function words or
    # named entities (朝代年號, 數量詞, 地名) that corrupt concept graph without semantic value.
    MAX_DOC_FREQ_RATIO = 0.30
    vocab = {t for t, c in doc_freq.items() if c >= 5 and c / n <= MAX_DOC_FREQ_RATIO}
    print(f"Vocab (doc_freq >= 5, <= {MAX_DOC_FREQ_RATIO:.0%} of corpus): {len(vocab):,} tokens")

    # pair_freq: co-doc counts for pairs within same doc (bag-of-tokens model)
    pair_freq: dict[tuple[str, str], int] = defaultdict(int)
    for _, tokens in doc_tokens:
        vt = sorted(tokens & vocab)
        for i in range(len(vt)):
            for j in range(i + 1, len(vt)):
                pair_freq[(vt[i], vt[j])] += 1

    # PMI for pairs with pair_freq >= 3
    # Special case: 數學/經濟/金融 bypass the PMI >= 2.0 threshold so they
    # accumulate PMI neighbors despite their extreme document frequency
    # (p ≈ 0.14–0.30 drags PMI below 2.0 for most pairs with these terms).
    SPECIAL_TERMS = {"數學", "經濟", "金融"}
    total_docs = n
    pmi_edges = []
    for (t1, t2), cf in pair_freq.items():
        if cf < 3 or t1 == t2:
            continue
        pt1 = doc_freq[t1] / total_docs
        pt2 = doc_freq[t2] / total_docs
        pt1t2 = cf / total_docs
        pmi = log(pt1t2 / (pt1 * pt2))
        is_special = t1 in SPECIAL_TERMS or t2 in SPECIAL_TERMS
        if is_special or pmi >= 2.0:
            pmi_edges.append((t1, t2, pmi, cf))

    print(f"PMI edges (PMI>=2.0, pair_freq>=3): {len(pmi_edges):,}")

    edges_df = pd.DataFrame(pmi_edges, columns=["src", "dst", "pmi", "co_doc_count"])

    # Build concept_domains: per-concept domain distribution
    all_domains = {"數學", "經濟金融"}
    domain_rows = []
    for t in vocab:
        dc = domain_counts[t]
        total = sum(dc.values())
        row = {"concept": t, "doc_freq": doc_freq[t]}
        for d in all_domains:
            row[f"docs_{d}"] = dc.get(d, 0)
        row["cross_domain_count"] = sum(1 for d in all_domains if dc.get(d, 0) > 0)
        domain_rows.append(row)

    domains_df = pd.DataFrame(domain_rows)

    (ROOT / "analysis").mkdir(exist_ok=True)
    edges_df.to_parquet(PMI_OUT)
    print(f"Written {len(edges_df):,} edges to {PMI_OUT}")
    domains_df.to_parquet(DOMAIN_OUT)
    print(f"Written {len(domains_df):,} concept domain rows to {DOMAIN_OUT}")

    return edges_df, domains_df


if __name__ == "__main__":
    run()
