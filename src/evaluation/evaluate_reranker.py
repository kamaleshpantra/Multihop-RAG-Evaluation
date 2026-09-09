import json
from pathlib import Path

from src.retrieval.reranked_hybrid_retriever import (
    RerankedHybridRetriever,
)
from src.evaluation.retrieval_metrics import (
    calculate_recall_at_k,
    calculate_mrr,
)


DATASET_PATH = (
    "data/raw/hotpotqa/"
    "hotpot_dev_distractor_v1.json"
)

CHECKPOINT_PATH = (
    "data/evaluation/"
    "reranker_checkpoint.json"
)

CANDIDATE_K = 20
FINAL_K_VALUES = (1, 5, 10)
CHECKPOINT_INTERVAL = 100


def load_dataset(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    path,
    processed,
    recall_scores,
    mrr_scores,
):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "processed": processed,
        "recall_scores": recall_scores,
        "mrr_scores": mrr_scores,
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            checkpoint,
            f,
        )


def load_checkpoint(path):

    if not Path(path).exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def evaluate(
    retriever,
    dataset,
    start_index=0,
    recall_scores=None,
    mrr_scores=None,
):

    if recall_scores is None:

        recall_scores = {
            str(k): []
            for k in FINAL_K_VALUES
        }

    if mrr_scores is None:
        mrr_scores = []

    total = len(dataset)

    for i in range(
        start_index,
        total,
    ):

        example = dataset[i]

        question = example["question"]

        supporting_facts = (
            example["supporting_facts"]
        )

        retrieved_documents = (
            retriever.retrieve(
                question,
                k=max(FINAL_K_VALUES),
            )
        )

        for k in FINAL_K_VALUES:

            top_k_documents = (
                retrieved_documents[:k]
            )

            recall = calculate_recall_at_k(
                supporting_facts,
                top_k_documents,
            )

            recall_scores[str(k)].append(
                recall
            )

        mrr = calculate_mrr(
            supporting_facts,
            retrieved_documents,
        )

        mrr_scores.append(mrr)

        processed = i + 1

        if processed % CHECKPOINT_INTERVAL == 0:

            save_checkpoint(
                CHECKPOINT_PATH,
                processed,
                recall_scores,
                mrr_scores,
            )

            print(
                f"Checkpoint saved: "
                f"{processed:,} / {total:,}"
            )

    return (
        recall_scores,
        mrr_scores,
    )


def calculate_final_results(
    recall_scores,
    mrr_scores,
):

    results = {}

    for k in FINAL_K_VALUES:

        scores = recall_scores[str(k)]

        results[f"recall@{k}"] = (
            sum(scores) / len(scores)
        )

    results["mrr"] = (
        sum(mrr_scores)
        / len(mrr_scores)
    )

    return results


def main():

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_PATH
    )

    print(
        f"Loaded {len(dataset):,} examples."
    )

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH
    )

    if checkpoint is not None:

        processed = checkpoint["processed"]

        recall_scores = (
            checkpoint["recall_scores"]
        )

        mrr_scores = (
            checkpoint["mrr_scores"]
        )

        print()
        print("=" * 60)
        print("CHECKPOINT FOUND")
        print("=" * 60)

        print(
            f"Previously processed: "
            f"{processed:,}"
        )

        print(
            f"Remaining: "
            f"{len(dataset) - processed:,}"
        )

        start_index = processed

    else:

        print()
        print("No checkpoint found.")

        recall_scores = {
            str(k): []
            for k in FINAL_K_VALUES
        }

        mrr_scores = []

        start_index = 0

    if start_index >= len(dataset):

        print(
            "Checkpoint already contains "
            "a complete evaluation."
        )

        results = calculate_final_results(
            recall_scores,
            mrr_scores,
        )

        print_results(results)

        return

    print()
    print("=" * 60)
    print("BUILDING RERANKED HYBRID RETRIEVER")
    print("=" * 60)

    retriever = RerankedHybridRetriever(
        candidate_k=CANDIDATE_K
    )

    print()
    print("=" * 60)
    print("EVALUATING")
    print("=" * 60)

    print(
        f"Candidate K: {CANDIDATE_K}"
    )

    print(
        f"Starting from example: "
        f"{start_index + 1:,}"
    )

    print(
        f"Remaining examples: "
        f"{len(dataset) - start_index:,}"
    )

    recall_scores, mrr_scores = evaluate(
        retriever,
        dataset,
        start_index=start_index,
        recall_scores=recall_scores,
        mrr_scores=mrr_scores,
    )

    results = calculate_final_results(
        recall_scores,
        mrr_scores,
    )

    # Evaluation completed successfully.
    Path(CHECKPOINT_PATH).unlink(
        missing_ok=True
    )

    print()
    print("=" * 60)
    print("RERANKED HYBRID RESULTS")
    print("=" * 60)

    print_results(results)


def print_results(results):

    print(
        f"Recall@1:  "
        f"{results['recall@1']:.4f}"
    )

    print(
        f"Recall@5:  "
        f"{results['recall@5']:.4f}"
    )

    print(
        f"Recall@10: "
        f"{results['recall@10']:.4f}"
    )

    print(
        f"MRR:       "
        f"{results['mrr']:.4f}"
    )


if __name__ == "__main__":
    main()