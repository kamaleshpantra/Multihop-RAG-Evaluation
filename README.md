# MultiHop-RAG-Evaluation

A reproducible evaluation framework comparing **dense, sparse, hybrid, and reranked** retrieval strategies for multi-hop Retrieval-Augmented Generation (RAG), benchmarked on HotpotQA.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green)
![Tests](https://img.shields.io/badge/tests-pytest-yellow)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Why This Project

Single-passage retrieval breaks down on multi-hop questions, where the answer depends on evidence spread across multiple documents. This project builds a **retrieve-then-rerank pipeline** and measures, quantitatively, how much each stage — dense retrieval, sparse (BM25), hybrid fusion, and cross-encoder reranking — improves the odds of surfacing the right supporting evidence.

## Results

Benchmarked on the HotpotQA dev distractor set (~108K chunks):

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| Dense (FAISS + MiniLM) | 0.337 | 0.578 | 0.644 | 0.440 |
| BM25 | 0.317 | 0.579 | 0.679 | 0.431 |
| Hybrid (RRF) | 0.351 | 0.659 | 0.747 | 0.481 |
| **Hybrid + Cross-Encoder Reranker** | **0.434** | **0.753** | **0.793** | **0.573** |

**Adding a cross-encoder reranker on top of hybrid retrieval lifts Recall@1 by +8.4 points and MRR by +9.2 points** — the single biggest gain in the pipeline, at the cost of extra inference time on a small (top-20) candidate pool.

## Architecture

```text
Query
  │
  ├──────────────┬──────────────┐
  ▼              ▼
Dense (FAISS +  BM25 (Sparse
 MiniLM-L6-v2)   Lexical)
  │              │
  └──────┬───────┘
         ▼
  Reciprocal Rank Fusion (k=60)
         ▼
     Top-20 Candidates
         ▼
  Cross-Encoder Reranker
   (ms-marco-MiniLM-L-6-v2)
         ▼
        Top-10
```

## Key Features

- End-to-end pipeline: preprocessing → chunking → indexing → retrieval → reranking → evaluation
- Dense retrieval via Sentence Transformers + FAISS
- Sparse retrieval via BM25 (rank-bm25)
- Hybrid fusion via Reciprocal Rank Fusion
- Cross-encoder reranking for final relevance scoring
- Recall@K and MRR evaluation against HotpotQA's gold supporting facts
- Candidate-pool size ablation (K = 10/20/50/100)
- Unit-tested core components (loader, chunker, embeddings, metrics, tokenizer, retrievers)
- Precomputed dataset, chunks, and FAISS index shipped via Git LFS — run experiments without rebuilding

## Tech Stack

Python · LangChain · Sentence Transformers · Hugging Face · FAISS · BM25 (rank-bm25) · NumPy · Pytest · Git LFS

## Project Structure

```text
MultiHop-RAG-Evaluation/
├── data/               # Raw HotpotQA data + processed chunks
├── vectorstore/faiss/  # Prebuilt FAISS index
├── src/
│   ├── data/           # Loading & preprocessing
│   ├── indexing/       # Chunking, embeddings, index build
│   ├── retrieval/      # Dense, BM25, hybrid, reranker
│   └── evaluation/     # Recall@K, MRR, evaluation scripts
├── tests/              # Unit tests
└── experiments/results/
```

## Quick Start

```bash
git lfs install
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MultiHop-RAG-Evaluation

python3 -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Run any stage of the pipeline (dataset + FAISS index are already included):

```bash
python -m src.evaluation.evaluate_retrieval   # Dense
python -m src.evaluation.evaluate_bm25        # BM25
python -m src.evaluation.evaluate_hybrid      # Hybrid RRF
python -m src.evaluation.evaluate_reranker    # Hybrid + Reranker (best)
```

Rebuild the pipeline from scratch:

```bash
python -m src.data.process_dataset     # → data/processed/chunks.json
python -m src.indexing.build_index     # → vectorstore/faiss/
```

Run tests:

```bash
python -m pytest
```

## Methodology

Chunks are created with LangChain's `RecursiveCharacterTextSplitter` (500 chars, 100 overlap), deduplicated on `(title, text)` before chunking, and tagged with sentence IDs. A retrieved chunk counts as a match when its title equals the gold supporting-fact title **and** the gold sentence ID is present in its metadata — evaluation is at the chunk level, not the official end-to-end HotpotQA QA protocol.

## Limitations & Future Work

This project evaluates **retrieval only** — not answer generation. Planned extensions:

- Local LLM answer generation (Ollama / Qwen) for full end-to-end RAG evaluation (Answer EM/F1, faithfulness)
- Explicit multi-hop query decomposition with iterative retrieval
- Ablations over embedding models, chunk size/overlap, RRF constant, and reranker choice
- Latency and memory profiling per pipeline stage

## License

Research and educational use. Add an appropriate open-source license before public release.