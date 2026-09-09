import argparse
import json
import re
from pathlib import Path

from src.retrieval.multihop_retriever import MultiHopRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hotpotqa"
    / "hotpot_dev_distractor_v1.json"
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

            title = fact[0]
            sentence_id = fact[1]

            try:
                sentence_id = int(
                    sentence_id
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            facts.append(
                (
                    normalize_title(title),
                    sentence_id,
                )
            )

    return facts


def get_retrieved_facts(documents):

    retrieved_facts = set()

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

            retrieved_facts.add(
                (
                    title,
                    sentence_id,
                )
            )

    return retrieved_facts


def evaluate(
    retriever,
    examples,
    retrieval_k,
):

    total = len(examples)

    complete_count = 0

    total_coverage = 0.0

    matched_facts = 0
    total_gold_facts = 0

    failures = []

    for index, example in enumerate(
        examples,
        start=1,
    ):

        question = example["question"]

        print(
            f"[{index}/{total}] "
            f"{question}"
        )

        documents = retriever.retrieve(
            question,
            k=retrieval_k,
        )

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

        gold_count = len(gold_facts)

        matched_count = len(matched)

        if gold_count > 0:

            coverage = (
                matched_count
                / gold_count
            )

        else:

            coverage = 0.0

        complete = (
            gold_count > 0
            and matched_count == gold_count
        )

        if complete:
            complete_count += 1

        total_coverage += coverage

        matched_facts += matched_count

        total_gold_facts += gold_count

        if not complete:

            failures.append(
                {
                    "id": example["_id"],
                    "question": question,
                    "gold_answer": example["answer"],
                    "matched_facts": [
                        {
                            "title": title,
                            "sentence_id": sid,
                        }
                        for title, sid in matched
                    ],
                    "missed_facts": [
                        {
                            "title": title,
                            "sentence_id": sid,
                        }
                        for title, sid in missed
                    ],
                    "retrieved_titles": [
                        document.metadata.get(
                            "title",
                            "",
                        )
                        for document in documents
                    ],
                }
            )

    if total > 0:

        complete_rate = (
            complete_count
            / total
        )

        average_coverage = (
            total_coverage
            / total
        )

    else:

        complete_rate = 0.0
        average_coverage = 0.0

    if total_gold_facts > 0:

        fact_recall = (
            matched_facts
            / total_gold_facts
        )

    else:

        fact_recall = 0.0

    return {
        "num_examples": total,
        "retrieval_k": retrieval_k,
        "complete_count": complete_count,
        "complete_rate": complete_rate,
        "average_coverage": average_coverage,
        "matched_facts": matched_facts,
        "total_gold_facts": total_gold_facts,
        "fact_recall": fact_recall,
        "failures": failures,
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multi-hop retrieval "
            "on HotpotQA."
        )
    )

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
        default=10,
    )

    parser.add_argument(
        "--expansion-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "experiments/retrieval/"
            "multihop_retrieval_evaluation.json"
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    print("=" * 70)
    print("MULTI-HOP RETRIEVAL EVALUATION")
    print("=" * 70)

    dataset = load_dataset(
        DATASET_PATH
    )

    if args.limit > 0:

        examples = dataset[:args.limit]

    else:

        examples = dataset

    print(
        f"Dataset size: {len(dataset):,}"
    )

    print(
        f"Evaluating: {len(examples):,}"
    )

    # ---------------------------------------------------------
    # Initialize retriever
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("INITIALIZING MULTI-HOP RETRIEVER")
    print("=" * 70)

    print(
        f"Candidate K: {args.candidate_k}"
    )

    print(
        f"Initial K:   {args.initial_k}"
    )

    print(
        f"Expansion K: {args.expansion_k}"
    )

    print(
        f"Final K:     {args.retrieval_k}"
    )

    retriever = MultiHopRetriever(
        candidate_k=args.candidate_k,
        initial_k=args.initial_k,
        expansion_k=args.expansion_k,
        final_k=args.retrieval_k,
    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    result = evaluate(
        retriever=retriever,
        examples=examples,
        retrieval_k=args.retrieval_k,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MULTI-HOP RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"Examples: "
        f"{result['num_examples']}"
    )

    print(
        f"Complete retrieval: "
        f"{result['complete_count']}/"
        f"{result['num_examples']} "
        f"("
        f"{result['complete_rate'] * 100:.2f}%"
        f")"
    )

    print(
        f"Average supporting-fact coverage: "
        f"{result['average_coverage'] * 100:.2f}%"
    )

    print(
        f"Supporting-fact recall: "
        f"{result['fact_recall'] * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Compare with baseline
    # ---------------------------------------------------------

    baseline_complete = 0.66
    baseline_coverage = 0.8313
    baseline_fact_recall = 0.8279

    print("\n" + "=" * 70)
    print("COMPARISON WITH CURRENT BASELINE")
    print("=" * 70)

    complete_delta = (
        result["complete_rate"]
        - baseline_complete
    )

    coverage_delta = (
        result["average_coverage"]
        - baseline_coverage
    )

    fact_recall_delta = (
        result["fact_recall"]
        - baseline_fact_recall
    )

    print(
        f"Complete retrieval:"
        f"  baseline = 66.00%"
        f"  multi-hop = "
        f"{result['complete_rate'] * 100:.2f}%"
        f"  delta = "
        f"{complete_delta * 100:+.2f} pp"
    )

    print(
        f"Coverage:"
        f"  baseline = 83.13%"
        f"  multi-hop = "
        f"{result['average_coverage'] * 100:.2f}%"
        f"  delta = "
        f"{coverage_delta * 100:+.2f} pp"
    )

    print(
        f"Fact recall:"
        f"  baseline = 82.79%"
        f"  multi-hop = "
        f"{result['fact_recall'] * 100:.2f}%"
        f"  delta = "
        f"{fact_recall_delta * 100:+.2f} pp"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Results saved to:\n"
        f"{output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()