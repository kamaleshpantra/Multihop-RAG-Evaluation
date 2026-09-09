import json

from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.retrieval_metrics import (
    calculate_recall_at_k,
    calculate_mrr,
)


DATASET_PATH = "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"

CANDIDATE_K_VALUES = [10, 20, 50, 100]

FINAL_K_VALUES = (1, 5, 10)


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_candidate_k(
    hybrid_retriever,
    dataset,
    candidate_k_values,
    final_k_values=(1, 5, 10),
):
    """
    Evaluate multiple candidate_k values using a single retrieval
    up to the maximum candidate_k.

    Example:

        candidate_k_values = [10, 20, 50, 100]

    We retrieve the top 100 Dense and BM25 candidates once,
    then perform RRF using the first 10, 20, 50, and 100
    candidates.
    """

    max_candidate_k = max(candidate_k_values)

    # Store metrics separately for each candidate_k.
    all_results = {}

    for candidate_k in candidate_k_values:

        all_results[candidate_k] = {
            "recall@1": [],
            "recall@5": [],
            "recall@10": [],
            "mrr": [],
        }

    for i, example in enumerate(dataset):

        question = example["question"]
        supporting_facts = example["supporting_facts"]

        # Retrieve once using the maximum candidate_k.
        dense_documents = (
            hybrid_retriever.dense_retriever.similarity_search(
                question,
                k=max_candidate_k,
            )
        )

        bm25_documents = (
            hybrid_retriever.bm25_retriever.retrieve(
                question,
                k=max_candidate_k,
            )
        )

        # Evaluate each candidate_k using the same retrieved
        # Dense/BM25 candidate lists.
        for candidate_k in candidate_k_values:

            fused_documents = fuse_results(
                dense_documents[:candidate_k],
                bm25_documents[:candidate_k],
                rrf_k=hybrid_retriever.rrf_k,
            )

            # We only need the final top 10.
            fused_documents = fused_documents[:max(final_k_values)]

            for final_k in final_k_values:

                top_k_documents = fused_documents[:final_k]

                recall = calculate_recall_at_k(
                    supporting_facts,
                    top_k_documents,
                )

                all_results[candidate_k][
                    f"recall@{final_k}"
                ].append(recall)

            mrr = calculate_mrr(
                supporting_facts,
                fused_documents,
            )

            all_results[candidate_k]["mrr"].append(mrr)

        if (i + 1) % 100 == 0:

            print(
                f"Evaluated {i + 1:,} / "
                f"{len(dataset):,} examples..."
            )

    # Convert lists of individual scores into averages.
    final_results = {}

    for candidate_k in candidate_k_values:

        final_results[candidate_k] = {}

        for metric, scores in all_results[candidate_k].items():

            final_results[candidate_k][metric] = (
                sum(scores) / len(scores)
            )

    return final_results


def fuse_results(
    dense_documents,
    bm25_documents,
    rrf_k=60,
):
    """
    Perform Reciprocal Rank Fusion on two ranked lists.
    """

    scores = {}
    documents = {}

    # Dense results
    for rank, document in enumerate(
        dense_documents,
        start=1,
    ):

        key = document_key(document)

        scores.setdefault(key, 0.0)

        scores[key] += 1 / (rrf_k + rank)

        documents[key] = document

    # BM25 results
    for rank, document in enumerate(
        bm25_documents,
        start=1,
    ):

        key = document_key(document)

        scores.setdefault(key, 0.0)

        scores[key] += 1 / (rrf_k + rank)

        documents[key] = document

    # Sort by RRF score.
    ranked_documents = sorted(
        documents.values(),
        key=lambda document: scores[
            document_key(document)
        ],
        reverse=True,
    )

    return ranked_documents


def document_key(document):

    return (
        document.metadata["title"],
        document.metadata["source_id"],
        document.page_content,
    )


def print_results(results):

    print()
    print("=" * 80)
    print("CANDIDATE K EXPERIMENT")
    print("=" * 80)

    print(
        f"{'Candidate K':<15}"
        f"{'R@1':<12}"
        f"{'R@5':<12}"
        f"{'R@10':<12}"
        f"{'MRR':<12}"
    )

    print("-" * 63)

    for candidate_k, metrics in results.items():

        print(
            f"{candidate_k:<15}"
            f"{metrics['recall@1']:<12.4f}"
            f"{metrics['recall@5']:<12.4f}"
            f"{metrics['recall@10']:<12.4f}"
            f"{metrics['mrr']:<12.4f}"
        )


def main():

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    dataset = load_dataset(DATASET_PATH)

    print(
        f"Loaded {len(dataset):,} examples."
    )

    print()
    print("=" * 60)
    print("BUILDING HYBRID RETRIEVER")
    print("=" * 60)

    hybrid_retriever = HybridRetriever(
        rrf_k=60
    )

    print()
    print("=" * 60)
    print("EVALUATING CANDIDATE K VALUES")
    print("=" * 60)

    print(
        f"Candidate K values: "
        f"{CANDIDATE_K_VALUES}"
    )

    print(
        f"Maximum candidate K: "
        f"{max(CANDIDATE_K_VALUES)}"
    )

    print()
    print(
        "Dense and BM25 will be retrieved once "
        "up to the maximum K."
    )

    print(
        "The different K values will then be "
        "evaluated from those candidates."
    )

    results = evaluate_candidate_k(
        hybrid_retriever,
        dataset,
        candidate_k_values=CANDIDATE_K_VALUES,
        final_k_values=FINAL_K_VALUES,
    )

    print_results(results)


if __name__ == "__main__":
    main()