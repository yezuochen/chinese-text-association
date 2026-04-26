"""
L1b: Build directional PMI graph over the full corpus (local CPU, free).

Input: processed/docs.csv
Output:
  - analysis/directional_pmi.parquet   (src, dst, pmi_forward, pmi_reverse, co_doc_count, asymmetry)

Algorithm:
  - For each document, track token positions (ordered list, not bag).
  - Sliding forward window (window_size=50): for each occurrence of t1 at position i,
    count t2 occurrences at positions j where i < j ≤ i + window_size.
  - Build forward_count[t1→t2] (directed pair count).
  - PMI(t1→t2) = log( forward_count[t1→t2] / doc_freq[t1] / (doc_freq[t2] / total_docs) )
  - Keep directed edges with forward_count >= 3 and PMI >= 1.5 in at least one direction.

This is a RESEARCH ARTIFACT only — not fused into the unified graph or query APIs.
"""

from pathlib import Path
from collections import defaultdict
from math import log

import pandas as pd
import jieba

ROOT = Path(__file__).resolve().parent.parent
DOCS_IN = ROOT / "processed" / "docs.csv"
DPMI_OUT = ROOT / "analysis" / "directional_pmi.parquet"

WINDOW_SIZE = 50   # tokens forward
MIN_FORWARD_COUNT = 3
MIN_PMI = 1.5      # lower than symmetric L1 (2.0) since directional signal is noisier

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
STOPWORDS |= {"wiki", "zh", "hant", "cn", "參考", "連結", "研究", "外部", "發展", "主要", "大學", "問題", "學家", "其中", "文獻", "政府", "國家", "社會", "公司", "教授", "理論", "方法", "包括"}
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


def tokenize_with_positions(text: str) -> list[str]:
    """Return ordered list of filtered tokens (positions matter for directional PMI)."""
    return [
        tok
        for tok in jieba.cut(text)
        if (
            len(tok) >= 2
            and tok.lower() not in STOPWORDS
            and not tok.isdigit()
            and not any("a" <= c <= "z" or "A" <= c <= "Z" for c in tok)
        )
    ]


def run() -> pd.DataFrame:
    docs = pd.read_csv(DOCS_IN, keep_default_na=False)
    n = len(docs)
    print(f"Tokenizing {n:,} docs with jieba (directional PMI)...")

    # doc_freq per token (for marginal probability denominator)
    doc_freq: dict[str, int] = defaultdict(int)

    # forward_count[t1→t2] — number of times t2 appears within WINDOW_SIZE tokens AFTER t1
    forward_count: dict[tuple[str, str], int] = defaultdict(int)

    for _, text in zip(docs["id"], docs["text"]):
        tokens = tokenize_with_positions(text)

        # Update doc_freq (unique tokens per doc, same as L1)
        seen = set()
        for t in tokens:
            if t not in seen:
                doc_freq[t] += 1
                seen.add(t)

        # Sliding forward window: for each position i, look at tokens[i+1 : i+1+WINDOW_SIZE]
        num_tokens = len(tokens)
        for i, t1 in enumerate(tokens):
            window_end = min(i + 1 + WINDOW_SIZE, num_tokens)
            for j in range(i + 1, window_end):
                t2 = tokens[j]
                if t2 != t1:
                    forward_count[(t1, t2)] += 1

    # Filter to vocab with doc_freq >= 5 (same threshold as L1)
    vocab = {t for t, c in doc_freq.items() if c >= 5}
    print(f"Vocab (doc_freq >= 5): {len(vocab):,} tokens")
    print(f"Directed pairs with forward_count >= {MIN_FORWARD_COUNT}: {sum(1 for c in forward_count.values() if c >= MIN_FORWARD_COUNT):,}")

    total_docs = n
    edges = []
    for (t1, t2), fc in forward_count.items():
        if fc < MIN_FORWARD_COUNT:
            continue
        if t1 not in vocab or t2 not in vocab:
            continue

        # PMI(t1→t2) = log( P(t2|t1) / P(t2) )
        # P(t2|t1) = forward_count[t1→t2] / doc_freq[t1]   (conditional probability)
        # P(t2) = doc_freq[t2] / total_docs                   (marginal)
        p_t2_given_t1 = fc / doc_freq[t1]
        p_t2 = doc_freq[t2] / total_docs

        if p_t2_given_t1 <= 0 or p_t2 <= 0:
            continue

        pmi = log(p_t2_given_t1 / p_t2)

        if pmi >= MIN_PMI:
            edges.append((t1, t2, pmi, fc))

    print(f"Directional PMI edges (PMI >= {MIN_PMI}, forward_count >= {MIN_FORWARD_COUNT}): {len(edges):,}")

    # Build symmetric table: we need both (A→B) and (B→A) for asymmetry
    # Filter to pairs where at least one direction passes the threshold
    edge_dict: dict[tuple[str, str], tuple[float, int]] = {}
    for t1, t2, pmi, fc in edges:
        edge_dict[(t1, t2)] = (pmi, fc)

    rows = []
    seen = set()
    for (t1, t2), (pmi_fwd, fc) in edge_dict.items():
        # Get reverse direction if exists
        rev_pmi = edge_dict.get((t2, t1), (None, None))[0]
        asymmetry = pmi_fwd - rev_pmi if rev_pmi is not None else pmi_fwd

        # Only keep if forward OR reverse passes threshold (keep asymmetric pairs)
        if pmi_fwd >= MIN_PMI or (rev_pmi is not None and rev_pmi >= MIN_PMI):
            rows.append({
                "src": t1,
                "dst": t2,
                "pmi_forward": pmi_fwd,
                "pmi_reverse": rev_pmi if rev_pmi is not None else float("nan"),
                "co_doc_count": fc,
                "asymmetry": asymmetry,
            })

    df = pd.DataFrame(rows)
    print(f"Final directional PMI edges (at least one direction >= {MIN_PMI}): {len(df):,}")

    (ROOT / "analysis").mkdir(exist_ok=True)
    df.to_parquet(DPMI_OUT)
    print(f"Written {len(df):,} edges to {DPMI_OUT}")

    return df


if __name__ == "__main__":
    run()