"""Ingestion: PDF -> clean text -> one chunk per article -> data/processed/chunks.jsonl

Pipeline per document:
  1. extract    : text per page with PyMuPDF
  2. clean      : Unicode NFKC, Arabic fixes, remove page numbers / banners
  3. split      : detect article headings ("الفصل 12", "ARTICLE 12 :") and cut between them
  4. chunk      : long articles are split into overlapping parts
  5. write      : JSONL, one chunk per line, with metadata for filtering and citations

Run:  python -m src.ingest
"""
import json
import re
import unicodedata
from dataclasses import dataclass

import pymupdf

from src.config import (
    CHUNKS_PATH, CHUNK_OVERLAP_CHARS, DOCUMENTS, MAX_CHUNK_CHARS, PROCESSED_DIR, RAW_DIR,
)

# ---------------------------------------------------------------- 1. extract

def extract_pages(pdf_path) -> list[str]:
    """Return the raw text of each page. Empty pages would mean a scan (needs OCR)."""
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


# ---------------------------------------------------------------- 2. clean

ARABIC_DIACRITICS = re.compile(r"[ً-ْٰ]")   # tashkeel (fatha, damma, shadda...)
TATWEEL = "ـ"                                         # ـ  stretching character
# Some PDFs extract the lam-alef ligature in the wrong order: "األشخاص" instead of "الأشخاص"
LAM_ALEF_SWAP = re.compile(r"ا([أإآ])ل")
BANNERS = [
    '"دعم حقوق المرأة والطفل من خلال تكنولوجيا المعلومات"',  # header repeated on every page (CAWTAR copy)
    "Imprimerie Officielle de la République Tunisienne",
]


def clean_text(text: str) -> str:
    """Normalization for DISPLAY and embedding: readable text, consistent characters."""
    text = unicodedata.normalize("NFKC", text)   # Arabic presentation forms -> standard letters, ligatures -> letters
    text = LAM_ALEF_SWAP.sub(r"ال\1", text)
    # Sub-article numbers sit alone on a line as "ـ2": glue them to the heading as " -2"
    # BEFORE removing tatweel, otherwise "2" looks like a page number and gets deleted.
    text = re.sub(r"\n[ \t]*ـ[ \t]*(\d{1,2})[ \t]*(?=\n)", r" -\1", text)
    text = text.replace(TATWEEL, "")
    for b in BANNERS:
        text = text.replace(b, "")
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)  # lines that are only a page number
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


# ---------------------------------------------------------------- 2b. lam-alef repair
# Some IORT PDFs map the ligature "لا" to the characters "ال" (reversed), so
# "البلاد" is extracted as "البالد" and "الاتفاقيات" as "االتفاقيات".
# A blind replace would break genuine "ال" (أعمال، مال). Instead we try swapping each
# "ال" inside a word and keep the spelling that exists in a vocabulary built from a
# clean Arabic document (the Labour Code). Unknown words are left untouched.

AR_WORD = re.compile(r"[ء-يّ]+")
AR_PREFIXES = ("وبال", "وال", "بال", "فال", "كال", "لل", "ال", "و", "ب", "ف", "ل", "ك")


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for t in texts:
        for w in AR_WORD.findall(t):
            vocab[w] = vocab.get(w, 0) + 1
    return vocab


def _score(word: str, vocab: dict[str, int]) -> int:
    if word in vocab:
        return vocab[word]
    # also accept the word if it's known once a prefix (و، ب، ال...) is removed
    return max((vocab.get(word[len(p):], 0) for p in AR_PREFIXES
                if word.startswith(p) and len(word) - len(p) >= 3), default=0)


def _repair_word(word: str, vocab: dict[str, int]) -> str:
    positions = [m.start() for m in re.finditer("ال", word)]
    if not positions or len(positions) > 4 or _score(word, vocab):
        return word                                   # nothing suspicious, or already a known word
    best, best_score = word, 0
    for mask in range(1, 2 ** len(positions)):        # every combination of swaps
        chars = list(word)
        for bit, pos in enumerate(positions):
            if mask >> bit & 1:
                chars[pos], chars[pos + 1] = "ل", "ا"
        cand = "".join(chars)
        s = _score(cand, vocab)
        if s > best_score:
            best, best_score = cand, s
    return best


def repair_lam_alef(text: str, vocab: dict[str, int]) -> str:
    return AR_WORD.sub(lambda m: _repair_word(m.group(), vocab), text)


def normalize_for_search(text: str) -> str:
    """Aggressive normalization for MATCHING (keyword/sparse search, eval comparisons).
    Not shown to users: it changes spelling (ة->ه, ى->ي) so variants match each other."""
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub("[أإآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------- 3. split into articles

AR_ORDINALS = {"الأول": "1", "الاول": "1"}
FR_ORDINALS = {"PREMIER": "1"}

# Heading = start of line + keyword + number (or "first") + optional suffix (مكرر / bis / ـ2 ...)
AR_HEADING = re.compile(
    r"(?m)^[ \t]*الفص\s?ل[ \t]*(?P<num>\d+|الأول|الاول)"
    # suffix: "مكرر", "ثالثا"... (RTL extraction can put the ":" before it) or a sub-number "-2"
    r"(?P<suffix>(?:\s*:?\s*(?:مكرّ?ر|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا))?(?:\s*ـ?\s*-?\s*\d{1,2}(?=\s*\n))?)"
    # A real heading is followed by ":" / "(" / ")" / end of line — a cross-reference is followed by text...
    r"(?=[ \t]*(?:[:()\-.]?[ \t]*\n|:|\(|\)))"
    # ...including "الفصل 12 \n من هذه المجلة" ("article 12 of this code") wrapped onto the next line.
    r"(?!\s*\.?\s*من\s)"
)
FR_HEADING = re.compile(
    r"(?m)^[ \t]*ARTICLE[ \t]+(?P<num>\d+|PREMIER)"
    r"(?P<suffix>(?:[ \t]*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies))?)[ \t]*[:.\-]"
)


@dataclass
class Article:
    article: str      # "151", "151-2", "52 bis"
    page: int         # 1-based page where the heading appears
    text: str


def fix_glued_suffix(num: str, prev: str | None) -> str:
    """'262' right after article 26 is really '26-2' (the 'ـ2' got glued to the number)."""
    if prev and prev.isdigit() and num.isdigit():
        p, n = int(prev), int(num)
        if n > p + 20 and num.startswith(prev) and len(num) - len(prev) <= 2:
            return f"{prev}-{num[len(prev):]}"
    return num


def split_articles(pages: list[str], lang: str, start_page: int = 1, vocab: dict[str, int] | None = None) -> list[Article]:
    heading = AR_HEADING if lang == "ar" else FR_HEADING
    ordinals = AR_ORDINALS if lang == "ar" else FR_ORDINALS

    # Join pages but remember where each page starts, to map a heading back to its page.
    # Pages before `start_page` (promulgation law, table of contents...) are left out.
    full, starts = "", []
    for i, p in enumerate(pages):
        starts.append(len(full))
        if i + 1 < start_page:
            continue
        text = clean_text(p)
        if lang == "ar" and vocab:
            text = repair_lam_alef(text, vocab)
        full += text + "\n"

    def page_of(pos: int) -> int:
        return max(i for i, s in enumerate(starts) if s <= pos) + 1

    matches = list(heading.finditer(full))
    articles, prev_base = [], None
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        num = ordinals.get(m.group("num"), m.group("num"))
        num = fix_glued_suffix(num, prev_base)
        if num.isdigit():
            prev_base = num
        suffix = re.sub(r"[\sـ:]+", " ", m.group("suffix").replace("ّ", "")).strip()  # drop ":" and shadda
        suffix = re.sub(r"^-?\s*(\d+)$", r"-\1", suffix)          # "ـ2" -> "-2"
        label = f"{num}{suffix}" if suffix.startswith("-") else f"{num} {suffix}".strip()
        body = full[m.start():end].strip()
        if len(body) > 40:                                        # skip stray references / empty headings
            articles.append(Article(label, page_of(m.start()), body))
    return articles


# ---------------------------------------------------------------- 4. chunk

def hard_split(para: str, limit: int) -> list[str]:
    """Cut a paragraph longer than `limit` into pieces, at the last space before the limit."""
    pieces = []
    while len(para) > limit:
        cut = para.rfind(" ", 0, limit)   # last space within the first `limit` chars
        if cut <= 0:                      # no space at all (very rare): cut exactly at the limit
            cut = limit
        pieces.append(para[:cut])
        para = para[cut:].lstrip()
    pieces.append(para)
    return pieces


def overlap_tail(text: str, size: int) -> str:
    """Last ~`size` characters of `text`, starting at a word boundary."""
    if len(text) <= size:
        return text
    tail = text[-size:]
    space = tail.find(" ")                # first space in the tail
    return tail[space + 1:] if space != -1 else tail


def split_long(text: str) -> list[str]:
    """Split an over-long article on paragraph boundaries, with a word-aligned overlap."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    # Leave room for the overlap that gets prepended to each new part.
    budget = MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS
    paras = [piece for para in text.split("\n") for piece in hard_split(para, budget)]

    parts, current = [], ""
    for para in paras:
        if current.strip() and len(current) + len(para) > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            current = overlap_tail(current.strip(), CHUNK_OVERLAP_CHARS) + "\n"
        current += para + "\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def build_chunks(doc: dict, vocab: dict[str, int] | None = None) -> list[dict]:
    pages = extract_pages(RAW_DIR / doc["file"])
    articles = split_articles(pages, doc["lang"], doc.get("start_page", 1),
                              vocab if doc.get("repair_lam_alef") else None)
    chunks, seen = [], {}
    for art in articles:
        # The same article number can appear twice (e.g. the promulgation law before the code).
        n = seen[art.article] = seen.get(art.article, 0) + 1
        base_id = f"{doc['doc_id']}::art{art.article.replace(' ', '_')}" + (f"#{n}" if n > 1 else "")
        parts = split_long(art.text)
        for j, part in enumerate(parts):
            chunks.append({
                "chunk_id": base_id + (f"::p{j + 1}" if len(parts) > 1 else ""),
                "doc_id": doc["doc_id"],
                "code": doc["code"],
                "title": doc["title"],
                "lang": doc["lang"],
                "edition": doc["edition"],
                "article": art.article,
                "part": j + 1,
                "n_parts": len(parts),
                "page": art.page,
                "text": part,
                "text_norm": normalize_for_search(part),
            })
    return chunks


# ---------------------------------------------------------------- 5. write + report

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Vocabulary of correctly-encoded Arabic, used to repair the lam-alef ligature elsewhere.
    vocab = build_vocab([clean_text(p) for d in DOCUMENTS if d.get("vocab_source")
                         for p in extract_pages(RAW_DIR / d["file"])])
    all_chunks = []
    for doc in DOCUMENTS:
        chunks = build_chunks(doc, vocab)
        lengths = sorted(len(c["text"]) for c in chunks)
        print(f"{doc['doc_id']:<24} articles={len({c['chunk_id'].split('::p')[0] for c in chunks}):>4}  "
              f"chunks={len(chunks):>4}  median_chars={lengths[len(lengths)//2]:>5}  max_chars={lengths[-1]:>5}")
        all_chunks.extend(chunks)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_chunks)} chunks -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
