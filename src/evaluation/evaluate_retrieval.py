import json
from pathlib import Path

from src.retrieval.retriever import load_vectorstore
from src.evaluation.retrieval_metrics import (
    calculate_recall_at_k,
    calculate_mrr,
)


DATASET_PATH = "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"


def load_dataset(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(k_values=(1, 5, 10)):

    data = load_dataset(DATASET_PATH)

    vectorstore = load_vectorstore()

    totals = {
        k: 0.0
        for k in k_values
    }

    total_mrr = 0.0

    num_examples = len(data)

    for i, example in enumerate(data):

        question = example["question"]

        supporting_facts = example["supporting_facts"]

        retrieved_documents = vectorstore.similarity_search(
            question,
            k=max(k_values)
        )

        # Recall@K
        for k in k_values:

            top_k_documents = retrieved_documents[:k]

            recall = calculate_recall_at_k(
                supporting_facts,
                top_k_documents
            )

            totals[k] += recall

        # MRR
        mrr = calculate_mrr(
            supporting_facts,
            retrieved_documents
        )

        total_mrr += mrr

        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{num_examples}"
            )

    results = {
        f"Recall@{k}": totals[k] / num_examples
        for k in k_values
    }

    results["MRR"] = total_mrr / num_examples

    return results


if __name__ == "__main__":

    results = evaluate_retrieval()

    print("\n===== Dense Retrieval Results =====")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")