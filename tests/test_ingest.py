"""Unit tests for the tricky parts of ingestion. Run: python -m pytest -q"""
from src.ingest import (
    clean_text, fix_glued_suffix, normalize_for_search, repair_lam_alef, split_articles,split_long,MAX_CHUNK_CHARS
)

VOCAB = {"البلاد": 5, "الاتفاقيات": 3, "أعمال": 4}


def test_nfkc_turns_presentation_forms_into_letters():
    assert clean_text("ﺍﻟﻔﺼل") == "الفصل"


def test_lam_alef_repair_uses_vocabulary():
    assert repair_lam_alef("بالبالد", VOCAB) == "بالبلاد"          # prefix ب + البلاد
    assert repair_lam_alef("واالتفاقيات", VOCAB) == "والاتفاقيات"
    assert repair_lam_alef("أعمال", VOCAB) == "أعمال"              # genuine "ال" is kept
    assert repair_lam_alef("كلمةمجهولة", VOCAB) == "كلمةمجهولة"    # unknown words untouched


def test_glued_sub_article_number():
    assert fix_glued_suffix("262", prev="26") == "26-2"
    assert fix_glued_suffix("27", prev="26") == "27"


def test_headings_and_cross_references():
    page = (
        "الفصل 5\nنص الفصل الخامس وهو نص طويل بما يكفي للاحتفاظ به في القائمة.\n"
        "الفصل 5\n مكرر\n)\nنص الفصل الخامس مكرر وهو كذلك نص طويل بما يكفي.\n"
        "تطبق أحكام\nالفصل12\n من هذه المجلة على الحالات التالية وهذا ليس عنوانا.\n"
    )
    labels = [a.article for a in split_articles([page], "ar")]
    assert labels == ["5", "5 مكرر"]

def test_split_long_handles_one_huge_paragraph():
    text = " ".join(["mot"] * 2000)                # 8,000 chars, no newline
    parts = split_long(text)
    assert len(parts) > 1
    assert all(len(p) <= MAX_CHUNK_CHARS for p in parts)
    assert all(p.startswith("mot") for p in parts)  # overlap starts on a whole word

def test_french_headings():
    page = ("ARTICLE PREMIER : L'impôt sur le revenu est dû par toute personne physique.\n"
            "ARTICLE 11 bis : Texte de l'article onze bis, assez long pour être gardé.\n")
    assert [a.article for a in split_articles([page], "fr")] == ["1", "11 bis"]


def test_search_normalization():
    assert normalize_for_search("الأجرة") == normalize_for_search("الاجره")
