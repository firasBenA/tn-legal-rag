"""Central configuration: paths, corpus registry, chunking and model settings."""
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# One entry per source document. `code` groups the FR and AR versions of the same law,
# so we can later filter by code or compare languages.
DOCUMENTS = [
    {
        "doc_id": "code_travail_ar_2004",
        "file": "code_travail_ar_2004.pdf",
        "code": "code_travail",
        "title": "مجلة الشغل",
        "lang": "ar",
        "edition": "2004",
        "start_page": 6,          # pages 1-5 = cover + 1966 promulgation law (its own articles 1-3)
        "vocab_source": True,     # clean Arabic text -> reference vocabulary for lam-alef repair
    },
    {
        "doc_id": "irpp_is_ar",
        "file": "irpp_is_ar.pdf",
        "code": "irpp_is",
        "title": "مجلة الضريبة على دخل الأشخاص الطبيعيين والضريبة على الشركات",
        "lang": "ar",
        "edition": "IORT",
        "repair_lam_alef": True,  # this PDF extracts "لا" as "ال" (e.g. البالد -> البلاد)
    },
    {
        "doc_id": "irpp_is_fr",
        "file": "irpp_is_fr.pdf",
        "code": "irpp_is",
        "title": "Code de l'IRPP et de l'IS",
        "lang": "fr",
        "edition": "IORT",
    },
]

# Chunking: one chunk per article; articles longer than this are split into parts.
MAX_CHUNK_CHARS = 2500
CHUNK_OVERLAP_CHARS = 200

# Models / services (used from week 1 step 2 onward)
EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_PATH = os.getenv("QDRANT_PATH", str(ROOT / "qdrant_data"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tn_legal")
TOP_K = 5
