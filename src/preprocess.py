"""
Stage 0: Preprocess corpus → processed/docs.csv

In:  data/<domain>/*.txt  (9,943 files)
Out: processed/docs.csv   (id, domain, title, text, char_count, filtered_reason)

Hard filters applied:
  - char_count < 50  → skip (embedding collapse on stubs)
  - unique_token_count < 5  → skip (same)
  - filtered_reason column added so embed_files.py can detect stub docs post-hoc

Steps per file:
  1. Read UTF-8
  2. OpenCC normalize Simplified → Traditional (Taiwan, s2tw)
  3. Regex scrub: strip HTML/wiki-template residue, collapse whitespace
  4. Pick title (first non-empty non-template line; fall back to id)
  5. Emit row; domain = parent folder name
"""

import re
from pathlib import Path

import opencc
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_FILE = ROOT / "processed" / "docs.csv"

# ── OpenCC (Simplified → Traditional, Taiwan standard) ───────────────────────

cc = opencc.OpenCC("s2tw")

# ── Regex scrubber ────────────────────────────────────────────────────────────

# Strip HTML/wiki template lines and broken LaTeX, collapse whitespace.
# Covers patterns observed in the corpus:
#   border 1 cellpadding 4 cellspacing 0 ...
#   style .*
#   \\\w+  (backslash commands)
#   {.*?}  placeholders inside formulas
#   \w+\s*\(.*?\)  bare function calls
_WIKI_LINE = re.compile(
    r"^\s*(border\s+|cellpadding\s+|cellspacing\s+|style\s+|class\s+|valign\s+|align\s+|width\s+|height\s+"
    r"|<[^>]+>|\[\[|\]\]|\{|\||\}\s*$|\bdiv\b|\bspan\b|\bbr\b|~~|::|\|\|).*",
    re.IGNORECASE,
)
_LATEX_JUNK = re.compile(r"\\\([a-zA-Z]+\s*|left\s*|right\s*\)")
_EXTRA_BRACES = re.compile(r"\{([{}])\}")
_MULTI_WS = re.compile(r"\s{3,}")
_LEADING_WS = re.compile(r"^\s+")
_TRAILING_WS = re.compile(r"\s+$")


def scrub(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if _WIKI_LINE.match(line):
            continue
        line = _LATEX_JUNK.sub("", line)
        line = _EXTRA_BRACES.sub(r"\1", line)
        line = _MULTI_WS.sub(" ", line)
        line = _LEADING_WS.sub("", line)
        line = _TRAILING_WS.sub("", line)
        cleaned.append(line)
    return _MULTI_WS.sub("\n", "\n".join(cleaned)).strip()


# ── Title picker ─────────────────────────────────────────────────────────────

_TEMPLATE_LINE = re.compile(
    r"^\s*(=+|{|}|border\s|cellpadding|cellspacing|style\s|class\s|<|~~|\|\|).*",
    re.IGNORECASE,
)


def pick_title(lines, file_id):
    for line in lines:
        stripped = line.strip()
        if not stripped or _TEMPLATE_LINE.match(stripped):
            continue
        # First line that looks like a real title / header
        if len(stripped) < 200:
            return stripped
    return str(file_id)


def tokenize_for_count(text: str) -> list[str]:
    import jieba
    return [
        tok
        for tok in jieba.cut(text)
        if (
            len(tok) >= 2
            and not tok.isdigit()
            and not any("a" <= c <= "z" or "A" <= c <= "Z" for c in tok)
        )
    ]


# ── Main ─────────────────────────────────────────────────────────────────────


def run() -> pd.DataFrame:
    import jieba
    rows = []
    skipped = 0
    for domain_dir in sorted(DATA_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        for txt_path in sorted(domain_dir.glob("*.txt")):
            file_id = txt_path.stem
            raw = txt_path.read_text(encoding="utf-8")
            text_trad = cc.convert(raw)
            text_clean = scrub(text_trad)
            char_count = len(text_clean)

            # Hard filter: skip stubs that cause embedding collapse
            unique_tokens = set(tokenize_for_count(text_clean))
            if char_count < 50 or len(unique_tokens) < 5:
                skipped += 1
                continue

            title = pick_title(text_clean.splitlines(), file_id)
            first_line = text_clean.split("\n", 1)[0]
            if len(first_line) > len(title) and len(first_line) < 200:
                title = first_line
            rows.append(
                {
                    "id": file_id,
                    "domain": domain,
                    "title": title,
                    "text": text_clean,
                    "char_count": char_count,
                    "filtered_reason": "",
                }
            )

    df = pd.DataFrame(rows, columns=["id", "domain", "title", "text", "char_count", "filtered_reason"])

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False, quoting=1, encoding="utf-8-sig")

    print(f"Written {len(df):,} rows to {OUT_FILE} (skipped {skipped} stub files)")
    print(df["domain"].value_counts().to_string())
    return df


if __name__ == "__main__":
    run()
