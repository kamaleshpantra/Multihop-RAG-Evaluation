import json

from src.retrieval.bm25_retriever import BM25Retriever

from src.evaluation.retrieval_metrics import (
    calculate_recall_at_k,
    calculate_mrr,
)


DATASET_PATH = (
    "data/raw/hotpotqa/"
    "hotpot_dev_distractor_v1.json"
)


def evaluate_bm25(k_values=(1, 5, 10)):

    print("Loading HotpotQA...")

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    print(
        f"Loaded {len(data):,} examples."
    )

    retriever = BM25Retriever()

    totals = {
        k: 0.0
        for k in k_values
    }

    total_mrr = 0.0

    max_k = max(k_values)

    for i, example in enumerate(data):

        question = example["question"]

        supporting_facts = (
            example["supporting_facts"]
        )

        documents = retriever.retrieve(
            question,
            k=max_k
        )

        for k in k_values:

            recall = calculate_recall_at_k(
                supporting_facts,
                documents[:k]
            )

            totals[k] += recall

        total_mrr += calculate_mrr(
            supporting_facts,
            documents
        )

        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(data)}"
            )

    results = {
        f"Recall@{k}":
            totals[k] / len(data)

        for k in k_values
    }

    results["MRR"] = (
        total_mrr / len(data)
    )

    return results


if __name__ == "__main__":

    results = evaluate_bm25()

    print(
        "\n===== BM25 Retrieval Results ====="
    )

    for metric, value in results.items():

        print(
            f"{metric}: {value:.4f}"
        )