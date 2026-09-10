# MultiHop-RAG-Evaluation

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green)
![Tests](https://img.shields.io/badge/tests-pytest-yellow)
![License](https://img.shields.io/badge/license-unlicensed-lightgrey)

A research-oriented, multi-hop Retrieval-Augmented Generation (RAG) system that experimentally evaluates dense retrieval, BM25, hybrid retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, multi-hop query expansion, recall-preserving retrieval, and end-to-end answer generation — benchmarked on HotpotQA.

**[Live Demo](https://multihop-rag-evaluation.streamlit.app/)** · **[Repository](https://github.com/kamaleshpantra/Multihop-RAG-Evaluation)**


---

## Why this project?

Most public RAG demos stop at "it works." This project treats RAG as an experimental system rather than a black box: every stage — dense retrieval, BM25, fusion, reranking, multi-hop expansion, and generation — is evaluated in isolation and in combination, with retrieval and generation errors analyzed separately. It demonstrates both **research rigor** (ablations, error decomposition, honest reporting of what does and doesn't help) and **engineering ability** (a working retrieval pipeline deployed as an interactive application).

## Research Questions

1. Does combining dense and lexical retrieval improve recall over either alone?
2. How much does cross-encoder reranking improve hybrid retrieval quality?
3. What retrieval depth (K) provides the best supporting-fact coverage without unnecessary cost?
4. Can lightweight multi-hop query expansion recover evidence missed on the first retrieval pass?
5. Can protecting strong baseline results reduce the regressions multi-hop expansion can introduce?
6. How does improved retrieval translate into end-to-end answer quality (EM/F1)?

---

## Local vs Cloud LLM

This project uses two different LLM configurations depending on context, and the distinction matters for interpreting the results below.

| | **Local / Offline (Research)** | **Cloud / Online (Demo)** |
|---|---|---|
| LLM | Qwen 2.5 Coder 7B | GPT-OSS 20B |
| Backend | Ollama | Groq API |
| Used for | The reported end-to-end benchmark results below | The deployed Streamlit application |

The retrieval pipeline is identical in both environments:

```
Question → Dense + BM25 → RRF → Multi-hop expansion → Cross-encoder reranking → Final context → LLM
```

Only the generation model/backend differs.

> **The reported benchmark results were generated using Qwen 2.5 Coder 7B through Ollama. The deployed Streamlit application uses GPT-OSS 20B through the Groq API, so the live demo is not intended to reproduce the exact benchmark numbers.** Groq is the inference platform, not the model; GPT-OSS 20B is the model it serves.

---

## System Evolution

The system was built and evaluated incrementally, with each stage benchmarked independently before being incorporated into the next:

```
Dense Retrieval
      ↓
BM25 Retrieval
      ↓
Hybrid (RRF)
      ↓
Cross-Encoder Reranking
      ↓
Multi-Hop Query Expansion
      ↓
Recall-Preserving Multi-Hop
      ↓
End-to-End RAG
```

Key results at each stage (Recall@1 unless noted):

| Stage | Result |
|---|---|
| Dense | Recall@1 = 33.72% |
| BM25 | Recall@1 = 31.65% |
| Hybrid (RRF) | Recall@1 ≈ 35.14% |
| Hybrid + Cross-Encoder | Recall@1 = 43.40% |
| Recall-Preserving Multi-Hop | Complete Retrieval = 70.00% |

The cross-encoder is the single largest lever on retrieval quality; recall-preserving multi-hop expansion gives a smaller but consistent lift on top of it.

---

## Architecture

```
QUESTION
   │
   ├── Dense Retrieval (MiniLM + FAISS, Top-20)
   └── BM25 Retrieval (Top-20)
              │
   Reciprocal Rank Fusion (k=60)
              │
      Initial Top-15
              │
      Extract Top-5 Titles
              │
   Query + Title Expansion
              │
   Expanded Candidate Retrieval
              │
      Merge Candidates
              │
   Cross-Encoder Reranking
              │
   Protect Baseline Top-5
              │
      Final Top-15
              │
      LLM Generation (Qwen 2.5 Coder 7B / GPT-OSS 20B)
```

**Component roles:**

- **Dense retrieval** finds semantically similar passages even without exact word overlap.
- **BM25** catches exact entity/term matches that dense embeddings can miss.
- **RRF** fuses the two ranked lists using rank position rather than raw scores, avoiding the need to calibrate dissimilar scoring scales.
- **Title/query expansion** chases a second entity discovered in the initial retrieval pass — a lightweight substitute for a full iterative reasoning agent.
- **Cross-encoder reranking** jointly encodes (query, document) pairs, which is more discriminative than comparing independent embeddings on a small candidate set.
- **Baseline protection** guards the original Top-5 hybrid+reranked documents from being pushed out during expansion, capping any regression multi-hop expansion could otherwise introduce.

---

## Retrieval Details

**Dense retrieval**
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)
- Index: FAISS

**Sparse retrieval**
- `rank-bm25`, lowercase/token-based lexical matching

**Fusion**
- Reciprocal Rank Fusion, `k = 60`

**Reranking**
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Final retrieval configuration**

| Parameter | Value |
|---|---|
| `candidate_k` | 20 |
| `retrieval_k` | 15 |
| `initial_k` | 15 |
| `expansion_k` | 5 |
| `expansion_candidate_k` | 20 |
| `protected_k` | 5 |

---

## Multi-Hop Retrieval

**Initial retrieval**
- Retrieve Top-15 via hybrid + reranking.
- Identify the Top-5 source document titles.

**Expansion**
- Construct expansion queries from the original question plus retrieved titles.
- Retrieve additional candidates using these expanded queries.
- Merge initial and expanded candidate sets.
- Rerank the merged pool with the cross-encoder.

**Recall-preserving heuristic (baseline-protected multi-hop retrieval)**
- The original Top-5 baseline documents are protected from being dropped.
- Remaining final positions are filled from the reranked expanded pool.
- The final Top-15 is returned.

This is a **heuristic**, not a formal recall guarantee — it protects known-good documents empirically, without a mathematical bound on recall.

---

## Results

### Retrieval (50 HotpotQA dev examples)

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| Dense (MiniLM) | 33.72% | 57.75% | 64.36% | 0.4404 |
| BM25 | 31.65% | 57.94% | 67.85% | 0.4308 |
| Hybrid (RRF) | 35.14% | 64.96% | 74.06% | 0.4790 |
| Hybrid + Cross-Encoder | **43.40%** | **75.30%** | **79.32%** | **0.5731** |

### Candidate-K Ablation (fusion candidate pool size)

| Candidate K | R@1 | R@5 | R@10 | MRR |
|---:|---:|---:|---:|---:|
| 10 | 0.3485 | 0.6612 | 0.7411 | 0.4792 |
| 20 | 0.3505 | 0.6590 | 0.7470 | 0.4807 |
| 50 | 0.3506 | 0.6526 | 0.7451 | 0.4794 |
| 100 | 0.3514 | 0.6496 | 0.7406 | 0.4790 |

`candidate_k = 20` was selected as the operating point — it gives a strong balance of R@1/R@10/MRR while keeping the reranker's candidate set manageable for latency.

### Retrieval-K Ablation (final retrieval depth)

| K | Complete Retrieval | Supporting-Fact Coverage | Supporting-Fact Recall |
|---:|---:|---:|---:|
| 5 | 58.00% | 79.47% | 78.69% |
| 10 | 64.00% | 82.13% | 81.97% |
| 15 | 66.00% | 83.13% | 82.79% |
| 20 | 66.00% | 83.13% | 82.79% |

`K = 15` was selected because increasing from 15 to 20 produced no additional retrieval improvement on the evaluated subset.

### Multi-Hop Retrieval

| System | Complete Retrieval | Supporting-Fact Coverage | Supporting-Fact Recall |
|---|---:|---:|---:|
| Baseline (Reranked Hybrid) | 66.00% | 83.13% | 82.79% |
| Multi-Hop Expansion | 68.00% | 84.93% | 85.25% |
| Recall-Preserving Multi-Hop | **70.00%** | **85.93%** | **86.07%** |

- Complete retrieval: 66% → 70% (**+4.0 pts**)
- Supporting-fact coverage: 83.13% → 85.93% (**+2.80 pts**)
- Supporting-fact recall: 82.79% → 86.07% (**+3.28 pts**)

### End-to-End RAG (50-question subset)

| Metric | Baseline RAG | Final Multi-Hop RAG |
|---|---:|---:|
| Exact Match | 42.00% | 44.00% |
| F1 | 0.4593 | 0.5065 |
| Complete Retrieval | 66.00% | 70.00% |
| Supporting-Fact Coverage | 83.13% | 85.93% |

- EM improvement: **+2.0 pts**
- F1 improvement: **+0.0472** (≈**10.3% relative**)
- Complete retrieval: **+4.0 pts**
- Supporting-fact coverage: **+2.80 pts**

These end-to-end numbers are from the 50-question evaluation subset described in [Limitations](#limitations), generated with Qwen 2.5 Coder 7B via Ollama.

---

## Error Analysis

The project decomposes errors along two axes — retrieval outcome and generation outcome — to separate retriever failures from generator failures, plus dedicated multi-hop and baseline→multi-hop transition analysis.

**50-question end-to-end generation breakdown:**

| Outcome | Count |
|---|---:|
| Retrieval success + generation success | 18 |
| Retrieval success + generation failure | 17 |
| Retrieval failure + generation success | 4 |
| Retrieval failure + generation failure | 11 |

17 questions had all supporting evidence retrieved but were still answered incorrectly — clear evidence that better retrieval alone doesn't guarantee better answers.

**Observed failure types:**
- Numerical extraction errors (e.g., a stated venue capacity of 3,677 generated as 4,000)
- Entity confusion and distractor competition
- Answer-normalization mismatches and yes/no errors
- Rare-entity retrieval failures, where generic semantic similarity struggles to surface the correct document
- Multi-entity questions where the retriever finds one entity but misses the second (e.g., queries involving *Shirley Temple*, *Adriana Trigiani*, *Androscoggin Bank Colisée*)
- Generation errors despite complete evidence retrieval

---

## Dataset & Evaluation Methodology

**Dataset:** HotpotQA, distractor setting (`hotpot_dev_distractor_v1.json`)

**Processed corpus:** ~73,700 unique documents → 108,270 chunks

**Chunking:** chunk size = 500 characters, chunk overlap = 100 characters. Sentence-level metadata is preserved to support supporting-fact evaluation.

This is a **controlled HotpotQA distractor retrieval experiment**, not open-domain Wikipedia retrieval — results should not be read as open-domain retrieval performance.

**Supporting-fact matching:** a retrieved chunk is counted as matching a gold supporting fact when (1) its title matches and (2) the corresponding gold sentence ID is present in the chunk's metadata. This is this project's own chunk-level evaluation methodology, and is **not** an exact reproduction of the official HotpotQA retrieval evaluation protocol.

## Evaluation Metrics

**Retrieval**
- Recall@1 / Recall@5 / Recall@10
- MRR
- Complete supporting-fact retrieval
- Supporting-fact coverage
- Supporting-fact recall

**Generation**
- Exact Match (EM)
- Token-level F1

**Not yet implemented** (see [Future Work](#future-work)): Answer Relevance, Context Relevance, Faithfulness/Groundedness, and citation-level correctness. These are documented as planned extensions, not current results.

---

## Runtime / Computational Cost

The full 50-question end-to-end RAG evaluation took **≈82.95 minutes** (**≈99.5 seconds/question**) on local Qwen 2.5 Coder 7B inference via Ollama.

Full-dataset generation (7,405 dev examples) was not run — this was an explicit experimental scope decision given the cost of local 7B inference, not a limitation of the pipeline itself.

---

## Deployment

Deployed via **Streamlit Community Cloud**:

- Repository: `github.com/kamaleshpantra/Multihop-RAG-Evaluation`
- Branch: `main`
- Entrypoint: `app/app.py`
- Secrets: configured through Streamlit Cloud's Secrets manager (`GROQ_API_KEY`), never committed to the repo
- Cloud LLM: GPT-OSS 20B via the Groq API (see [Local vs Cloud LLM](#local-vs-cloud-llm))

**Live Demo:** `https://YOUR-STREAMLIT-APP-URL.streamlit.app` *(placeholder — update after deployment)*

## Security

- `GROQ_API_KEY`, `.env`, and `.streamlit/secrets.toml` are **never committed** to the repository.
- `.gitignore` already excludes secrets and environment files.
- API keys are configured exclusively through Streamlit Cloud Secrets, not hardcoded or checked into source control.

---

## Setup

```bash
git clone https://github.com/kamaleshpantra/Multihop-RAG-Evaluation.git
cd Multihop-RAG-Evaluation

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell

pip install -r requirements.txt

ollama pull qwen2.5-coder:7b
```

**Build the index and process the dataset:**

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

**Run the app locally:**

```bash
streamlit run app/app.py
```

## Testing

Component-level tests (`pytest`) cover:

- Dense retrieval
- BM25 retrieval
- Hybrid (RRF) retrieval
- Cross-encoder reranking
- Multi-hop query expansion
- Recall-preserving multi-hop retrieval
- Generation / RAG pipeline components

*(No numerical test coverage percentage is claimed here — none has been formally measured.)*

---

## Project Structure

```
src/
├── data/            # HotpotQA loading and preprocessing
├── indexing/        # chunking, embeddings, FAISS index construction
├── retrieval/       # dense, BM25, hybrid (RRF), reranking, multi-hop expansion
├── generation/       # LLM prompting and RAG pipeline (Ollama / Groq)
└── evaluation/       # metrics, ablations, error analysis

app/                 # Streamlit application (deployed demo)
experiments/         # stored evaluation outputs, for reproducibility without rerunning the pipeline
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Dataset | HotpotQA (distractor setting) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS |
| Sparse search | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Local generation | Qwen 2.5 Coder 7B via Ollama |
| Cloud generation | GPT-OSS 20B via Groq API |
| App | Streamlit (Streamlit Community Cloud) |
| Testing | Pytest |
| Version control | Git, Git LFS |

## Engineering Highlights

- Hybrid retrieval combining semantic (dense) and lexical (BM25) search
- Rank-based fusion (RRF) avoiding cross-scale score calibration
- Cross-encoder reranking on a constrained candidate set
- Multi-hop query expansion to recover missing evidence for multi-entity questions
- Recall-preserving (baseline-protected) heuristic to cap multi-hop regressions
- Systematic retrieval-depth and candidate-K ablations
- Quantitative, stage-by-stage evaluation with separated retrieval/generation error analysis
- Dual-mode LLM setup: local offline inference (Ollama) for benchmarking, cloud inference (Groq) for the live demo
- Deployed, interactive Streamlit application
- Git LFS for large artifacts and reproducible stored experiment outputs

---

## Limitations

- End-to-end generation was evaluated on a 50-question subset, not the full 7,405-example HotpotQA dev set — local Qwen 7B inference took ~83 minutes for 50 questions (~99.5s/question), making a full run impractical on the current setup.
- Retrieval is scored against the HotpotQA distractor corpus, not open-domain Wikipedia — results shouldn't be read as open-domain retrieval performance.
- Supporting-fact matching is done at the chunk level (title + sentence ID), which is a project-specific evaluation, not an exact reproduction of the official HotpotQA protocol.
- Recall-preserving multi-hop is a heuristic (protects baseline Top-5), not a proven recall guarantee.
- Faithfulness, answer relevance, context relevance, and citation-level correctness metrics aren't implemented yet.
- The deployed demo (GPT-OSS 20B via Groq) is not expected to reproduce the exact Qwen/Ollama benchmark numbers reported above.

## Future Work

- Evaluate on the full HotpotQA dev set
- Add faithfulness, context relevance, and citation-level correctness metrics
- Query decomposition and adaptive stopping for multi-hop retrieval
- Batched cross-encoder inference and retrieval caching to reduce latency
- Automated regression tests for retrieval quality
- Experiment tracking and production-style monitoring for the deployed app

---

## Author

**Kamalesh Pantra** — [GitHub](https://github.com/kamaleshpantra)

No license has been added yet — add one before distributing this as open source.
