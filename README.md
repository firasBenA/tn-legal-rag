# TN Legal RAG — Bilingual (FR/AR) Assistant for Tunisian Labour Law

A retrieval-augmented question-answering system over Tunisian legal texts (starting with the Labour Code),
answering in French or Arabic, citing the exact articles it used, and refusing when the documents don't cover the question.

> Status: Week 1 — building the naive baseline.

## Planned architecture

```
PDFs (FR + AR)
   │  ingest.py   : extract text → normalize Arabic → split by article → chunks.jsonl
   ▼
chunks.jsonl
   │  index.py    : BGE-M3 embeddings (dense + sparse) → Qdrant collection
   ▼
Qdrant
   │  retrieve.py : top-k search (dense → hybrid → hybrid + reranker)
   ▼
generate.py       : prompt with retrieved articles → Llama-3 (Groq) → answer + citations / refusal
   │
api.py (FastAPI) → UI (Streamlit)        eval/ : gold questions → retrieval + RAGAS metrics
```

## Repository layout

| Path | Purpose |
|---|---|
| `data/raw/` | Source PDFs (not committed; downloaded by `scripts/`) |
| `data/processed/` | `chunks.jsonl` produced by ingestion |
| `src/config.py` | Paths, model names, collection name, env vars |
| `src/ingest.py` | Parse → clean → chunk |
| `src/index.py` | Embed chunks and load them into Qdrant |
| `src/retrieve.py` | Retrieval strategies |
| `src/generate.py` | Prompting and LLM call |
| `src/api.py` | FastAPI service (week 3) |
| `eval/` | Gold question set and evaluation scripts (week 2) |
| `notebooks/` | Kaggle notebooks that run the pipeline on GPU |
| `scripts/` | Data download helpers |

## Roadmap

- [ ] **Week 1** — corpus collected, ingestion, indexing, naive RAG answering with citations
- [ ] **Week 2** — gold eval set, baseline metrics, hybrid search, reranker, Langfuse tracing
- [ ] **Week 3** — FastAPI + Streamlit UI, refusal tuning, results table, demo video

## Results

_To be filled after week 2 (Recall@5, faithfulness, answer relevancy, refusal accuracy, latency)._

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
```
