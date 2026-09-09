import argparse
import json
import re
from pathlib import Path

from src.retrieval.recall_preserving_multihop import (
    RecallPreservingMultiHopRetriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hotpotqa"
    / "hotpot_dev_distractor_v1.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "retrieval"
    / "recall_preserving_multihop_evaluation.json"
)


def load_dataset(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def normalize_title(title):

    title = title.lower().strip()

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title


def get_supporting_facts(example):

    facts = []

    for fact in example.get(
        "supporting_facts",
        [],
    ):

        if (
            isinstance(fact, (list, tuple))
            and len(fact) >= 2
        ):

            title = normalize_title(
                fact[0]
            )

            try:

                sentence_id = int(
                    fact[1]
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            facts.append(
                (
                    title,
                    sentence_id,
                )
            )

    return facts


def get_retrieved_facts(documents):

    facts = set()

    for document in documents:

        title = normalize_title(
            document.metadata.get(
                "title",
                "",
            )
        )

        sentence_ids = (
            document.metadata.get(
                "sentence_ids",
                [],
            )
        )

        if sentence_ids is None:
            sentence_ids = []

        for sentence_id in sentence_ids:

            try:

                sentence_id = int(
                    sentence_id
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            facts.add(
                (
                    title,
                    sentence_id,
                )
            )

    return facts


def evaluate_example(
    example,
    documents,
):

    gold_facts = get_supporting_facts(
        example
    )

    retrieved_facts = get_retrieved_facts(
        documents
    )

    matched = [
        fact
        for fact in gold_facts
        if fact in retrieved_facts
    ]

    missed = [
        fact
        for fact in gold_facts
        if fact not in retrieved_facts
    ]

    if gold_facts:

        coverage = (
            len(matched)
            / len(gold_facts)
        )

    else:

        coverage = 0.0

    complete = (
        len(gold_facts) > 0
        and len(matched) == len(gold_facts)
    )

    return {
        "complete": complete,
        "coverage": coverage,
        "matched": matched,
        "missed": missed,
    }


def format_facts(facts):

    return [
        {
            "title": title,
            "sentence_id": sentence_id,
        }
        for title, sentence_id in facts
    ]


def get_titles(documents):

    return [
        document.metadata.get(
            "title",
            "",
        )
        for document in documents
    ]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--initial-k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--expansion-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--protected-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RECALL-PRESERVING MULTI-HOP EVALUATION")
    print("=" * 70)

    dataset = load_dataset(
        DATASET_PATH
    )

    examples = dataset[:args.limit]

    print(
        f"Evaluating {len(examples)} examples"
    )

    # ---------------------------------------------------------
    # Initialize
    # ---------------------------------------------------------

    print("\nInitializing retriever...")

    retriever = (
        RecallPreservingMultiHopRetriever(
            candidate_k=args.candidate_k,
            initial_k=args.initial_k,
            expansion_k=args.expansion_k,
            expansion_candidate_k=args.candidate_k,
            protected_k=args.protected_k,
            final_k=args.retrieval_k,
        )
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    complete_count = 0

    total_coverage = 0.0

    matched_facts = 0
    total_gold_facts = 0

    results = []

    for index, example in enumerate(
        examples,
        start=1,
    ):

        question = example["question"]

        print(
            f"[{index}/{len(examples)}] "
            f"{question}"
        )

        documents = retriever.retrieve(
            question,
            k=args.retrieval_k,
        )

        result = evaluate_example(
            example,
            documents,
        )

        if result["complete"]:

            complete_count += 1

        total_coverage += (
            result["coverage"]
        )

        matched_facts += len(
            result["matched"]
        )

        total_gold_facts += len(
            get_supporting_facts(
                example
            )
        )

        results.append(
            {
                "id": example["_id"],
                "question": question,
                "answer": example["answer"],
                "complete": result["complete"],
                "coverage": result["coverage"],
                "matched_facts": format_facts(
                    result["matched"]
                ),
                "missed_facts": format_facts(
                    result["missed"]
                ),
                "retrieved_titles": get_titles(
                    documents
                ),
            }
        )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    total = len(examples)

    complete_rate = (
        complete_count / total
        if total
        else 0.0
    )

    average_coverage = (
        total_coverage / total
        if total
        else 0.0
    )

    fact_recall = (
        matched_facts / total_gold_facts
        if total_gold_facts
        else 0.0
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Examples: {total}"
    )

    print(
        f"Complete retrieval: "
        f"{complete_count}/{total} "
        f"({complete_rate * 100:.2f}%)"
    )

    print(
        f"Average supporting-fact coverage: "
        f"{average_coverage * 100:.2f}%"
    )

    print(
        f"Supporting-fact recall: "
        f"{fact_recall * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Compare with previous systems
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        "Baseline:"
        "       Complete = 66.00%"
        "  Coverage = 83.13%"
        "  Fact recall = 82.79%"
    )

    print(
        "Multi-hop:"
        "      Complete = 68.00%"
        "  Coverage = 84.93%"
        "  Fact recall = 85.25%"
    )

    print(
        f"Recall-preserving:"
        f" Complete = {complete_rate * 100:.2f}%"
        f"  Coverage = {average_coverage * 100:.2f}%"
        f"  Fact recall = {fact_recall * 100:.2f}%"
    )

    print("\n" + "=" * 70)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = {
        "num_examples": total,
        "configuration": {
            "candidate_k": args.candidate_k,
            "initial_k": args.initial_k,
            "expansion_k": args.expansion_k,
            "protected_k": args.protected_k,
            "retrieval_k": args.retrieval_k,
        },
        "metrics": {
            "complete_count": complete_count,
            "complete_rate": complete_rate,
            "average_coverage": average_coverage,
            "fact_recall": fact_recall,
        },
        "examples": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved results to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()