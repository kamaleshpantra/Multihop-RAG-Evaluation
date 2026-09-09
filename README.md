# MultiHop-RAG-Evaluation

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green)
![Tests](https://img.shields.io/badge/tests-pytest-yellow)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A Retrieval-Augmented Generation (RAG) system built to measure not just demo how different retrieval strategies affect multi-hop question answering. It combines dense retrieval, BM25, Reciprocal Rank Fusion, cross-encoder reranking, and a recall-preserving query-expansion step, evaluated end-to-end on the HotpotQA distractor benchmark.

**[Live Demo](YOUR_STREAMLIT_APP_URL)** · **[Repository](https://github.com/kamaleshpantra/Multihop-RAG-Evaluation)**

---

## What this project answers

Most RAG demos stop at "it works." This one asks: *where* does it work, and *why does it fail* when it doesn't. Every stage dense retrieval, BM25, fusion, reranking, multi-hop expansion, and final generation is evaluated in isolation and against each other, with retrieval and generation errors analyzed separately.

## Results

**Retrieval** (50 HotpotQA dev examples):

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| Dense (MiniLM) | 33.72% | 57.75% | 64.36% | 0.4404 |
| BM25 | 31.65% | 57.94% | 67.85% | 0.4308 |
| Hybrid (RRF) | 35.14% | 64.96% | 74.06% | 0.4790 |
| Hybrid + Cross-Encoder | **43.40%** | **75.30%** | **79.32%** | **0.5731** |

**Multi-hop retrieval:**

| System | Complete Retrieval | Supporting-Fact Coverage | Supporting-Fact Recall |
|---|---:|---:|---:|
| Baseline (hybrid + reranked) | 66.00% | 83.13% | 82.79% |
| Multi-hop expansion | 68.00% | 84.93% | 85.25% |
| Recall-preserving multi-hop | **70.00%** | **85.93%** | **86.07%** |

**End-to-end RAG (50-question subset):**

| Metric | Baseline RAG | Final Multi-Hop RAG | Δ |
|---|---:|---:|---:|
| Exact Match | 42.00% | 44.00% | +2.0 pts |
| F1 | 0.4593 | 0.5065 | +0.0472 (≈10.3% relative) |
| Complete Retrieval | 66.00% | 70.00% | +4.0 pts |
| Supporting-Fact Coverage | 83.13% | 85.93% | +2.8 pts |

The cross-encoder is the single largest lever on retrieval quality; recall-preserving multi-hop expansion gives a smaller but consistent lift on top of it.

## Error analysis

Splitting the 50-example generation run by retrieval outcome vs. generation outcome:

| Outcome | Count |
|---|---:|
| Retrieval success + generation success | 18 |
| Retrieval success + generation failure | 17 |
| Retrieval failure + generation success | 4 |
| Retrieval failure + generation failure | 11 |

17 questions had all supporting evidence retrieved but were still answered incorrectly — proof that better retrieval alone doesn't guarantee better answers. Observed generation failures include numerical extraction errors (e.g. a stated venue capacity of 3,677 generated as 4,000), entity confusion, and answer-normalization mismatches. Observed retrieval failures cluster around multi-entity questions where the retriever finds one entity but misses the second (e.g. queries involving *Shirley Temple*, *Adriana Trigiani*, *Androscoggin Bank Colisée*), and around rare entities that generic semantic similarity struggles to surface.

## Architecture

```
Question
   ├── Dense Retrieval (MiniLM + FAISS, top-20)
   └── BM25 Retrieval (top-20)
          │
   Reciprocal Rank Fusion (k=60)
          │
   Initial Top-15 → extract top-5 titles
          │
   Query + Title Expansion → merge candidates
          │
   Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
          │
   Protect baseline Top-5 → fill remaining from reranked pool
          │
   Final Top-15
          │
   Qwen 2.5 Coder 7B (via Ollama) → Answer
```

**Retrieval K ablation** — depth beyond K=15 gave no further benefit on this evaluation subset, so K=15 was used as the operating point:

| K | Complete Retrieval | Supporting-Fact Coverage |
|---|---:|---:|
| 5 | 58.00% | 79.47% |
| 10 | 64.00% | 82.13% |
| 15 | 66.00% | 83.13% |
| 20 | 66.00% | 83.13% |

## Why these choices

- **Dense + BM25**: dense retrieval catches semantic matches, BM25 catches exact entity/term matches. Combining beats either alone.
- **RRF over score blending**: rank-based fusion avoids having to calibrate dense and BM25 scores to a common scale.
- **Cross-encoder reranking**: jointly encodes (query, document) rather than comparing independent embeddings, which matters most on a small candidate set.
- **Title-based query expansion**: a lightweight way to chase a second entity discovered during initial retrieval, without building a full iterative reasoning agent.
- **Protecting baseline Top-5**: naive expansion can push good baseline documents out of the final ranking during rerank; protecting them caps that regression.

## Tech stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector search | FAISS |
| Sparse search | BM25 (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Qwen 2.5 Coder 7B via Ollama (Groq supported for cloud deployment) |
| App | Streamlit |
| Dataset | HotpotQA, distractor setting (~73,700 documents / 108,270 chunks after chunking) |

## Setup

```bash
git clone https://github.com/kamaleshpantra/Multihop-RAG-Evaluation.git
cd Multihop-RAG-Evaluation

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell

pip install -r requirements.txt

ollama pull qwen2.5-coder:7b
```

**Build the index and run the pipeline:**

```bash
python -m src.data.process_dataset
python -m src.indexing.build_index
```

**Run evaluations:**

```bash
python -m src.evaluation.evaluate_retrieval                        # dense
python -m src.evaluation.evaluate_bm25
python -m src.evaluation.evaluate_hybrid
python -m src.evaluation.evaluate_reranker
python -m src.evaluation.evaluate_retrieval_k --limit 50            # K ablation
python -m src.evaluation.evaluate_multihop_retrieval --limit 50
python -m src.evaluation.evaluate_recall_preserving_multihop --limit 50
python -m src.evaluation.evaluate_rag --limit 50                    # end-to-end
```

**Run the app:**

```bash
streamlit run app/app.py
```

Deployed via Streamlit Community Cloud, pointed at `app/app.py`; secrets (e.g. `GROQ_API_KEY`) are configured through Streamlit's secrets manager, never committed to the repo.

## Limitations

- End-to-end generation was evaluated on a 50-question subset (not the full 7,405-example dev set) — local Qwen 7B inference took ~83 minutes for 50 questions (~99.5s/question), making a full run impractical on the current setup.
- Retrieval is scored against the HotpotQA distractor corpus, not open-domain Wikipedia — results shouldn't be read as open-domain retrieval performance.
- Supporting-fact matching is done at the chunk level (title + sentence ID), which is a project-specific evaluation, not an exact reproduction of the official HotpotQA protocol.
- Recall-preserving multi-hop is a heuristic (protects baseline Top-5), not a proven recall guarantee.
- Faithfulness, answer relevance, and other LLM-as-a-judge metrics aren't implemented yet.

## Future work

- Evaluate on the full HotpotQA dev set
- Add faithfulness, context relevance, and citation-level correctness metrics
- Query decomposition and adaptive stopping for multi-hop retrieval
- Batched cross-encoder inference and retrieval caching to cut latency
- Automated regression tests for retrieval quality

## Project structure

```
src/
├── data/            # loading and preprocessing
├── indexing/        # chunking, embeddings, FAISS index
├── retrieval/       # dense, BM25, hybrid, reranking, multi-hop
├── generation/      # LLM prompting and RAG pipeline
└── evaluation/       # metrics, ablations, error analysis

app/                 # Streamlit application
experiments/         # stored evaluation outputs, for reproducibility without rerunning the pipeline
```

## Author

**Kamalesh Pantra** — [GitHub](https://github.com/kamaleshpantra)

No license has been added yet — add one before distributing this as open source.
