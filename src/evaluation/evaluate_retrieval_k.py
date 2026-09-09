import argparse
import json
import re
from pathlib import Path

from src.retrieval.reranked_hybrid_retriever import RerankedHybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hotpotqa"
    / "hotpot_dev_distractor_v1.json"
)


def load_dataset(path: Path):
    """Load the HotpotQA dataset."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_title(title: str) -> str:
    """Normalize article titles for reliable matching."""
    title = title.lower().strip()
    title = re.sub(r"\s+", " ", title)
    return title


def get_supporting_facts(example):
    """
    Extract gold supporting facts.

    Standard HotpotQA format:
        [
            ["Scott Derrickson", 0],
            ["Ed Wood", 0]
        ]

    Returns:
        [
            ("scott derrickson", 0),
            ("ed wood", 0)
        ]
    """

    facts = []

    for fact in example.get("supporting_facts", []):

        # Standard HotpotQA representation
        if isinstance(fact, (list, tuple)) and len(fact) >= 2:

            title = fact[0]
            sentence_id = fact[1]

            try:
                sentence_id = int(sentence_id)
            except (TypeError, ValueError):
                continue

            facts.append(
                (
                    normalize_title(title),
                    sentence_id,
                )
            )

        # Support dictionary representation as well
        elif isinstance(fact, dict):

            title = fact.get("title")

            sentence_id = fact.get(
                "sent_id",
                fact.get("sentence_id"),
            )

            if title is None or sentence_id is None:
                continue

            try:
                sentence_id = int(sentence_id)
            except (TypeError, ValueError):
                continue

            facts.append(
                (
                    normalize_title(title),
                    sentence_id,
                )
            )

    return facts


def get_retrieved_facts(documents):
    """
    Extract all sentence-level facts contained in
    the retrieved chunks.

    Returns a set of:

        (normalized_title, sentence_id)
    """

    retrieved_facts = set()

    for document in documents:

        title = normalize_title(
            document.metadata.get(
                "title",
                "",
            )
        )

        sentence_ids = document.metadata.get(
            "sentence_ids",
            [],
        )

        if sentence_ids is None:
            sentence_ids = []

        for sentence_id in sentence_ids:

            try:
                sentence_id = int(sentence_id)
            except (TypeError, ValueError):
                continue

            retrieved_facts.add(
                (
                    title,
                    sentence_id,
                )
            )

    return retrieved_facts


def evaluate_single_k(
    retriever,
    examples,
    retrieval_k,
):
    """
    Evaluate the retriever at one particular K.

    We check:

    1. Whether all supporting facts were retrieved.
    2. What fraction of supporting facts were retrieved.
    3. Overall supporting-fact recall.
    """

    total_examples = len(examples)

    complete_retrieval_count = 0

    total_coverage = 0.0

    matched_fact_count = 0
    total_gold_fact_count = 0

    results = []

    for index, example in enumerate(
        examples,
        start=1,
    ):

        question = example["question"]

        print(
            f"[{index}/{total_examples}] "
            f"{question}"
        )

        # IMPORTANT:
        # candidate_k is already configured when
        # RerankedHybridRetriever is initialized.
        #
        # We only change the final retrieval K here.
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

        matched_facts = [
            fact
            for fact in gold_facts
            if fact in retrieved_facts
        ]

        gold_count = len(gold_facts)

        matched_count = len(matched_facts)

        if gold_count > 0:

            coverage = (
                matched_count
                / gold_count
            )

        else:

            coverage = 0.0

        all_facts_retrieved = (
            gold_count > 0
            and matched_count == gold_count
        )

        if all_facts_retrieved:
            complete_retrieval_count += 1

        total_coverage += coverage

        matched_fact_count += matched_count

        total_gold_fact_count += gold_count

        results.append(
            {
                "id": example["_id"],
                "question": question,
                "retrieval_k": retrieval_k,
                "gold_facts": [
                    {
                        "title": title,
                        "sentence_id": sentence_id,
                    }
                    for title, sentence_id
                    in gold_facts
                ],
                "matched_facts": [
                    {
                        "title": title,
                        "sentence_id": sentence_id,
                    }
                    for title, sentence_id
                    in matched_facts
                ],
                "matched_count": matched_count,
                "gold_count": gold_count,
                "coverage": coverage,
                "all_facts_retrieved":
                    all_facts_retrieved,
            }
        )

    # ------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------

    if total_examples > 0:

        complete_retrieval_rate = (
            complete_retrieval_count
            / total_examples
        )

        average_coverage = (
            total_coverage
            / total_examples
        )

    else:

        complete_retrieval_rate = 0.0
        average_coverage = 0.0

    if total_gold_fact_count > 0:

        supporting_fact_recall = (
            matched_fact_count
            / total_gold_fact_count
        )

    else:

        supporting_fact_recall = 0.0

    return {
        "retrieval_k": retrieval_k,
        "num_examples": total_examples,
        "complete_retrieval_count":
            complete_retrieval_count,
        "complete_retrieval_rate":
            complete_retrieval_rate,
        "average_supporting_fact_coverage":
            average_coverage,
        "matched_fact_count":
            matched_fact_count,
        "total_gold_fact_count":
            total_gold_fact_count,
        "supporting_fact_recall":
            supporting_fact_recall,
        "results": results,
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval performance "
            "at different top-K values."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help=(
            "Number of HotpotQA examples "
            "to evaluate. "
            "Use 0 for the full dataset."
        ),
    )

    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help=(
            "Final retrieval K values "
            "to evaluate."
        ),
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help=(
            "Number of hybrid candidates "
            "before reranking."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "experiments/retrieval/"
            "retrieval_k_ablation.json"
        ),
        help="Path for saving results.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    print("=" * 70)
    print("RETRIEVAL K ABLATION")
    print("=" * 70)

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        DATASET_PATH
    )

    print(
        f"Loaded {len(dataset):,} examples."
    )

    if args.limit > 0:

        examples = dataset[:args.limit]

    else:

        examples = dataset

    print(
        f"Evaluating {len(examples):,} examples."
    )

    # ------------------------------------------------------------
    # Validate K values
    # ------------------------------------------------------------

    if not args.ks:

        raise ValueError(
            "At least one retrieval K "
            "value must be provided."
        )

    for k in args.ks:

        if k <= 0:

            raise ValueError(
                "Retrieval K must be greater than 0."
            )

        if k > args.candidate_k:

            raise ValueError(
                f"retrieval K={k} cannot be greater "
                f"than candidate K={args.candidate_k}."
            )

    # ------------------------------------------------------------
    # Initialize retriever ONCE
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("INITIALIZING RETRIEVER")
    print("=" * 70)

    print(
        f"Candidate K: {args.candidate_k}"
    )

    print(
        "Retriever: Hybrid + Cross-Encoder Reranker"
    )

    retriever = RerankedHybridRetriever(
        candidate_k=args.candidate_k
    )

    # ------------------------------------------------------------
    # Run each K
    # ------------------------------------------------------------

    all_results = {}

    for retrieval_k in args.ks:

        print("\n" + "=" * 70)
        print(
            f"EVALUATING RETRIEVAL K = "
            f"{retrieval_k}"
        )
        print("=" * 70)

        result = evaluate_single_k(
            retriever=retriever,
            examples=examples,
            retrieval_k=retrieval_k,
        )

        all_results[
            str(retrieval_k)
        ] = result

        # --------------------------------------------------------
        # Print individual K result
        # --------------------------------------------------------

        print("\n" + "-" * 70)
        print(
            f"RESULTS FOR K = {retrieval_k}"
        )
        print("-" * 70)

        print(
            f"All supporting facts retrieved: "
            f"{result['complete_retrieval_count']}/"
            f"{result['num_examples']} "
            f"("
            f"{result['complete_retrieval_rate'] * 100:.2f}%"
            f")"
        )

        print(
            f"Average supporting-fact coverage: "
            f"{result['average_supporting_fact_coverage']:.4f} "
            f"("
            f"{result['average_supporting_fact_coverage'] * 100:.2f}%"
            f")"
        )

        print(
            f"Supporting-fact recall: "
            f"{result['supporting_fact_recall']:.4f} "
            f"("
            f"{result['supporting_fact_recall'] * 100:.2f}%"
            f")"
        )

    # ------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVAL K SUMMARY")
    print("=" * 70)

    print(
        f"{'K':<8}"
        f"{'All Facts':<18}"
        f"{'Coverage':<18}"
        f"{'Fact Recall':<18}"
    )

    print("-" * 62)

    for k in args.ks:

        result = all_results[
            str(k)
        ]

        complete_rate = (
            result["complete_retrieval_rate"]
            * 100
        )

        coverage = (
            result[
                "average_supporting_fact_coverage"
            ]
            * 100
        )

        fact_recall = (
            result[
                "supporting_fact_recall"
            ]
            * 100
        )

        print(
            f"{k:<8}"
            f"{complete_rate:>7.2f}%"
            f"{'':<10}"
            f"{coverage:>7.2f}%"
            f"{'':<10}"
            f"{fact_recall:>7.2f}%"
        )

    # ------------------------------------------------------------
    # Determine best K
    # ------------------------------------------------------------

    best_k_by_complete_retrieval = max(
        args.ks,
        key=lambda k:
            all_results[str(k)][
                "complete_retrieval_rate"
            ],
    )

    best_k_by_fact_recall = max(
        args.ks,
        key=lambda k:
            all_results[str(k)][
                "supporting_fact_recall"
            ],
    )

    best_k_by_coverage = max(
        args.ks,
        key=lambda k:
            all_results[str(k)][
                "average_supporting_fact_coverage"
            ],
    )

    print("\n" + "=" * 70)
    print("BEST K VALUES")
    print("=" * 70)

    print(
        f"Best K by complete retrieval: "
        f"{best_k_by_complete_retrieval}"
    )

    print(
        f"Best K by supporting-fact recall: "
        f"{best_k_by_fact_recall}"
    )

    print(
        f"Best K by average coverage: "
        f"{best_k_by_coverage}"
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "num_examples": len(examples),
        "candidate_k": args.candidate_k,
        "retrieval_k_values": args.ks,
        "results": all_results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("ABLATION COMPLETE")
    print("=" * 70)

    print(
        f"Results saved to:\n"
        f"{output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()