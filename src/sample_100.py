"""
L3a: Concept-seeded stratified sample of 100 files for GraphRAG.

Seeds are DATA-DRIVEN from L1 PMI: seed-of-seed via domain name words.
  - Seed-of-seed: 數學, 經濟, 金融 (domain folder names, now in PMI vocab)
  - For each: top-3 PMI neighbors by PMI value → 9 seeds total
  - Seeds are cached to processed/seed_terms.csv for reproducibility

Input: processed/docs.csv, analysis/pmi_graph.parquet, analysis/concept_domains.parquet
Output: processed/sample_100.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS_IN = ROOT / "processed" / "docs.csv"
PMI_IN = ROOT / "analysis" / "pmi_graph.parquet"
DOMAINS_IN = ROOT / "analysis" / "concept_domains.parquet"
SAMPLE_OUT = ROOT / "processed" / "sample_100.csv"

SEEDS_CACHE = ROOT / "processed" / "seed_terms.csv"
SEED_OF_SEED = ["數學", "經濟", "金融"]
NEIGHBORS_PER_SEED = 3


def _find_seed_of_seed(domains_df: pd.DataFrame, pmi_df: pd.DataFrame) -> list[str]:
    """
    Seed-of-seed approach using domain name words as roots.

    Strategy:
    1. Load from cache if SEEDS_CACHE exists.
    2. Otherwise: for each of 數學/經濟/金融, find its top-3 PMI neighbors
       by PMI degree (excluding the domain word itself).
    3. This gives 9 seeds total (3 per domain word).
    4. Save computed seeds to SEEDS_CACHE for reproducibility.
    """
    if SEEDS_CACHE.exists():
        cached = pd.read_csv(SEEDS_CACHE, keep_default_na=False)
        print(f"Loaded {len(cached)} seeds from {SEEDS_CACHE}")
        for _, row in cached.iterrows():
            print(
                f"  {row['concept']}: from={row['source']}, "
                f"pmi={row['pmi']:.2f}, co_doc={row['co_doc']}"
            )
        return cached["concept"].tolist()

    rows = []
    for sos in SEED_OF_SEED:
        if sos not in domains_df["concept"].values:
            print(f"Warning: '{sos}' not in PMI vocabulary — skipping")
            continue
        edges = pmi_df[
            (pmi_df["src"] == sos) | (pmi_df["dst"] == sos)
        ].copy()
        edges["other"] = edges.apply(
            lambda r: r["dst"] if r["src"] == sos else r["src"], axis=1
        )
        # Exclude: (a) the seed-of-seed itself, (b) other seed-of-seed words,
        # (c) any term that contains a seed-of-seed word as a substring
        #     (e.g. 數學家 contains 數學)
        edges = edges[~edges["other"].isin(SEED_OF_SEED)]
        edges = edges[~edges["other"].apply(lambda t: any(sos in t for sos in SEED_OF_SEED))]
        top = edges.nlargest(NEIGHBORS_PER_SEED, "co_doc_count")
        for _, row in top.iterrows():
            rows.append({
                "concept": row["other"],
                "source": sos,
                "pmi": row["pmi"],
                "co_doc": row["co_doc_count"],
            })

    if not rows:
        raise RuntimeError(
            f"No PMI neighbors found for any of {SEED_OF_SEED} — "
            "check that these terms are in the PMI graph"
        )

    df = pd.DataFrame(rows)
    SEEDS_CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(SEEDS_CACHE, index=False, encoding="utf-8-sig")
    print(f"\nComputed and cached {len(df)} seeds to {SEEDS_CACHE}")
    for _, row in df.iterrows():
        print(
            f"  {row['concept']}: from={row['source']}, "
            f"pmi={row['pmi']:.2f}, co_doc={row['co_doc']}"
        )
    return df["concept"].tolist()


def run() -> pd.DataFrame:
    domains_df = pd.read_parquet(DOMAINS_IN)
    pmi_df = pd.read_parquet(PMI_IN)

    seed_terms = _find_seed_of_seed(domains_df, pmi_df)

    docs_df = pd.read_csv(DOCS_IN, keep_default_na=False)
    math_docs = docs_df[docs_df["domain"] == "數學"].copy()
    econ_docs = docs_df[docs_df["domain"] == "經濟金融"].copy()

    math_selected = []
    econ_selected = []

    for seed in seed_terms:
        math_matches = math_docs[math_docs["text"].str.contains(seed, regex=False)]
        econ_matches = econ_docs[econ_docs["text"].str.contains(seed, regex=False)]
        if len(math_matches) > 0:
            math_selected.append(math_matches.sample(1).iloc[0]["id"])
        if len(econ_matches) > 0:
            econ_selected.append(econ_matches.sample(1).iloc[0]["id"])

    math_pool = math_docs[~math_docs["id"].isin(math_selected)].copy()
    econ_pool = econ_docs[~econ_docs["id"].isin(econ_selected)].copy()

    math_pool["weight"] = math_pool["char_count"] ** 0.5
    econ_pool["weight"] = econ_pool["char_count"] ** 0.5

    target_per_domain = 50
    math_needed = target_per_domain - len(math_selected)
    econ_needed = target_per_domain - len(econ_selected)

    math_fill = (
        math_pool.sample(n=min(math_needed, len(math_pool)), weights="weight")
        if math_needed > 0
        else math_pool.iloc[:0]
    )
    econ_fill = (
        econ_pool.sample(n=min(econ_needed, len(econ_pool)), weights="weight")
        if econ_needed > 0
        else econ_pool.iloc[:0]
    )

    selected_ids = (
        math_selected
        + econ_selected
        + math_fill["id"].tolist()
        + econ_fill["id"].tolist()
    )
    sample_df = docs_df[docs_df["id"].isin(selected_ids)].head(100).copy()

    (ROOT / "processed").mkdir(exist_ok=True)
    sample_df.to_csv(SAMPLE_OUT, index=False, quoting=2, encoding="utf-8-sig")
    print(f"\nWritten {len(sample_df)} rows to {SAMPLE_OUT}")
    print(f"  數學: {len(sample_df[sample_df['domain'] == '數學'])}")
    print(f"  經濟金融: {len(sample_df[sample_df['domain'] == '經濟金融'])}")
    return sample_df


if __name__ == "__main__":
    run()
