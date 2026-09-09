import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hotpotqa"
    / "hotpot_dev_distractor_v1.json"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "generation"
    / "rag_predictions.json"
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_title(title: str) -> str:
    """Normalize article titles for reliable matching."""
    title = title.lower().strip()
    title = re.sub(r"\s+", " ", title)
    return title


def get_supporting_facts(example):
    """
    Return supporting facts as:
        [(title, sentence_id), ...]
    """

    supporting_facts = example.get("supporting_facts", [])

    facts = []

    # Standard HotpotQA format:
    # [
    #     ["Scott Derrickson", 0],
    #     ["Ed Wood", 0]
    # ]
    for fact in supporting_facts:
        if isinstance(fact, (list, tuple)) and len(fact) >= 2:
            title = fact[0]
            sentence_id = fact[1]

            facts.append(
                (
                    normalize_title(title),
                    int(sentence_id),
                )
            )

        # Also support dictionary-style representations if encountered.
        elif isinstance(fact, dict):
            title = fact.get("title")
            sentence_id = fact.get("sent_id", fact.get("sentence_id"))

            if title is not None and sentence_id is not None:
                facts.append(
                    (
                        normalize_title(title),
                        int(sentence_id),
                    )
                )

    return facts


def retrieved_fact_matches(retrieved_documents, supporting_facts):
    """
    Determine which gold supporting facts were retrieved.

    A gold fact is considered retrieved if:
        retrieved title == gold title
        AND
        gold sentence ID appears in the chunk's sentence_ids.
    """

    retrieved_facts = set()

    for document in retrieved_documents:
        title = normalize_title(document.get("title", ""))

        sentence_ids = document.get("sentence_ids", [])

        if sentence_ids is None:
            sentence_ids = []

        for sentence_id in sentence_ids:
            try:
                sentence_id = int(sentence_id)
            except (TypeError, ValueError):
                continue

            retrieved_facts.add((title, sentence_id))

    matched_facts = []

    for gold_fact in supporting_facts:
        if gold_fact in retrieved_facts:
            matched_facts.append(gold_fact)

    return matched_facts


def analyze_example(example, prediction):
    """
    Analyze one example for retrieval and generation success.
    """

    supporting_facts = get_supporting_facts(example)

    retrieved_documents = prediction.get("retrieved_documents", [])

    matched_facts = retrieved_fact_matches(
        retrieved_documents,
        supporting_facts,
    )

    total_gold_facts = len(supporting_facts)
    matched_count = len(matched_facts)

    if total_gold_facts == 0:
        retrieval_fraction = 0.0
        retrieval_success = False
    else:
        retrieval_fraction = matched_count / total_gold_facts
        retrieval_success = matched_count == total_gold_facts

    generation_success = prediction.get("exact_match", 0.0) == 1.0

    return {
        "id": example["_id"],
        "question": example["question"],
        "gold_answer": example["answer"],
        "prediction": prediction.get("prediction", ""),
        "exact_match": prediction.get("exact_match", 0.0),
        "f1": prediction.get("f1", 0.0),
        "supporting_facts": supporting_facts,
        "matched_facts": matched_facts,
        "total_gold_facts": total_gold_facts,
        "matched_count": matched_count,
        "retrieval_fraction": retrieval_fraction,
        "retrieval_success": retrieval_success,
        "generation_success": generation_success,
        "retrieved_documents": retrieved_documents,
    }


def print_percentage(value, total):
    if total == 0:
        return "0.00%"
    return f"{100.0 * value / total:.2f}%"


def main():

    print("=" * 70)
    print("RETRIEVAL vs GENERATION ERROR ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    print("\nLoading HotpotQA dataset...")

    dataset = load_json(DATASET_PATH)

    print(f"Loaded {len(dataset):,} dataset examples.")

    # ------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------

    print("\nLoading RAG predictions...")

    predictions_data = load_json(PREDICTIONS_PATH)

    predictions = predictions_data.get("results", [])

    print(f"Loaded {len(predictions):,} predictions.")

    # ------------------------------------------------------------
    # Match predictions to dataset examples
    # ------------------------------------------------------------

    dataset_by_id = {
        example["_id"]: example
        for example in dataset
    }

    analyzed_results = []

    missing_dataset_examples = 0

    for prediction in predictions:

        example_id = prediction.get("id")

        if example_id not in dataset_by_id:
            missing_dataset_examples += 1
            continue

        example = dataset_by_id[example_id]

        result = analyze_example(
            example,
            prediction,
        )

        analyzed_results.append(result)

    total = len(analyzed_results)

    # ------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------

    retrieval_success = 0
    retrieval_failure = 0

    generation_success = 0
    generation_failure = 0

    retrieval_success_generation_success = 0
    retrieval_success_generation_failure = 0
    retrieval_failure_generation_success = 0
    retrieval_failure_generation_failure = 0

    total_retrieval_fraction = 0.0

    retrieval_success_em = 0.0
    retrieval_failure_em = 0.0

    retrieval_success_f1 = 0.0
    retrieval_failure_f1 = 0.0

    for result in analyzed_results:

        rs = result["retrieval_success"]
        gs = result["generation_success"]

        total_retrieval_fraction += result["retrieval_fraction"]

        if rs:
            retrieval_success += 1

            retrieval_success_em += result["exact_match"]
            retrieval_success_f1 += result["f1"]

        else:
            retrieval_failure += 1

            retrieval_failure_em += result["exact_match"]
            retrieval_failure_f1 += result["f1"]

        if gs:
            generation_success += 1
        else:
            generation_failure += 1

        if rs and gs:
            retrieval_success_generation_success += 1

        elif rs and not gs:
            retrieval_success_generation_failure += 1

        elif not rs and gs:
            retrieval_failure_generation_success += 1

        else:
            retrieval_failure_generation_failure += 1

    # ------------------------------------------------------------
    # Overall statistics
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(f"Examples analyzed:          {total}")

    print(
        f"Generation correct:         "
        f"{generation_success}/{total} "
        f"({print_percentage(generation_success, total)})"
    )

    print(
        f"Generation incorrect:       "
        f"{generation_failure}/{total} "
        f"({print_percentage(generation_failure, total)})"
    )

    # ------------------------------------------------------------
    # Retrieval statistics
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVAL ANALYSIS")
    print("=" * 70)

    print(
        f"All supporting facts retrieved: "
        f"{retrieval_success}/{total} "
        f"({print_percentage(retrieval_success, total)})"
    )

    print(
        f"Not all supporting facts retrieved: "
        f"{retrieval_failure}/{total} "
        f"({print_percentage(retrieval_failure, total)})"
    )

    if total > 0:
        average_retrieval_fraction = (
            total_retrieval_fraction / total
        )
    else:
        average_retrieval_fraction = 0.0

    print(
        f"Average supporting-fact coverage: "
        f"{average_retrieval_fraction:.4f} "
        f"({average_retrieval_fraction * 100:.2f}%)"
    )

    # ------------------------------------------------------------
    # 2x2 diagnostic matrix
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVAL vs GENERATION")
    print("=" * 70)

    print(
        f"Retrieval SUCCESS + Generation SUCCESS: "
        f"{retrieval_success_generation_success}/{total} "
        f"({print_percentage(retrieval_success_generation_success, total)})"
    )

    print(
        f"Retrieval SUCCESS + Generation FAILURE: "
        f"{retrieval_success_generation_failure}/{total} "
        f"({print_percentage(retrieval_success_generation_failure, total)})"
    )

    print(
        f"Retrieval FAILURE + Generation SUCCESS: "
        f"{retrieval_failure_generation_success}/{total} "
        f"({print_percentage(retrieval_failure_generation_success, total)})"
    )

    print(
        f"Retrieval FAILURE + Generation FAILURE: "
        f"{retrieval_failure_generation_failure}/{total} "
        f"({print_percentage(retrieval_failure_generation_failure, total)})"
    )

    # ------------------------------------------------------------
    # Conditional generation performance
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATION PERFORMANCE CONDITIONED ON RETRIEVAL")
    print("=" * 70)

    if retrieval_success > 0:

        em_when_retrieval_success = (
            retrieval_success_em / retrieval_success
        )

        f1_when_retrieval_success = (
            retrieval_success_f1 / retrieval_success
        )

        print(
            f"When retrieval succeeds:"
        )

        print(
            f"  EM: {em_when_retrieval_success:.4f} "
            f"({em_when_retrieval_success * 100:.2f}%)"
        )

        print(
            f"  F1: {f1_when_retrieval_success:.4f}"
        )

    else:
        print("No examples with successful retrieval.")

    if retrieval_failure > 0:

        em_when_retrieval_failure = (
            retrieval_failure_em / retrieval_failure
        )

        f1_when_retrieval_failure = (
            retrieval_failure_f1 / retrieval_failure
        )

        print(
            f"\nWhen retrieval fails:"
        )

        print(
            f"  EM: {em_when_retrieval_failure:.4f} "
            f"({em_when_retrieval_failure * 100:.2f}%)"
        )

        print(
            f"  F1: {f1_when_retrieval_failure:.4f}"
        )

    else:
        print("\nNo examples with failed retrieval.")

    # ------------------------------------------------------------
    # Detailed failure categories
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if total > 0:

        retrieval_failure_rate = retrieval_failure / total

        generation_failure_given_retrieval_success = (
            retrieval_success_generation_failure / retrieval_success
            if retrieval_success > 0
            else 0.0
        )

        print(
            f"Retrieval failure rate: "
            f"{retrieval_failure_rate * 100:.2f}%"
        )

        print(
            f"Generation failure rate when retrieval succeeds: "
            f"{generation_failure_given_retrieval_success * 100:.2f}%"
        )

        if (
            retrieval_failure
            > retrieval_success_generation_failure
        ):
            print(
                "\nPrimary signal: retrieval appears to be the "
                "larger bottleneck."
            )

        elif (
            retrieval_success_generation_failure
            > retrieval_failure
        ):
            print(
                "\nPrimary signal: generation/reasoning appears "
                "to be the larger bottleneck."
            )

        else:
            print(
                "\nPrimary signal: retrieval and generation "
                "appear similarly important."
            )

    # ------------------------------------------------------------
    # Detailed examples
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVAL SUCCESS + GENERATION FAILURE")
    print("=" * 70)

    examples = [
        result
        for result in analyzed_results
        if result["retrieval_success"]
        and not result["generation_success"]
    ]

    for index, result in enumerate(examples[:10], start=1):

        print("\n" + "-" * 70)
        print(f"EXAMPLE {index}")

        print(f"Question:   {result['question']}")
        print(f"Gold:       {result['gold_answer']}")
        print(f"Prediction: {result['prediction']}")

        print(
            f"Supporting facts retrieved: "
            f"{result['matched_count']}/"
            f"{result['total_gold_facts']}"
        )

        print("Retrieved titles:")

        for document in result["retrieved_documents"][:10]:
            rank = document.get("rank", "?")
            title = document.get("title", "")

            print(
                f"  [{rank}] {title}"
            )

    # ------------------------------------------------------------
    # Retrieval failure + generation failure
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVAL FAILURE + GENERATION FAILURE")
    print("=" * 70)

    examples = [
        result
        for result in analyzed_results
        if not result["retrieval_success"]
        and not result["generation_success"]
    ]

    for index, result in enumerate(examples[:10], start=1):

        print("\n" + "-" * 70)
        print(f"EXAMPLE {index}")

        print(f"Question:   {result['question']}")
        print(f"Gold:       {result['gold_answer']}")
        print(f"Prediction: {result['prediction']}")

        print(
            f"Supporting facts retrieved: "
            f"{result['matched_count']}/"
            f"{result['total_gold_facts']}"
        )

        print(
            f"Coverage: "
            f"{result['retrieval_fraction'] * 100:.1f}%"
        )

        print("Gold supporting facts:")

        for title, sentence_id in result["supporting_facts"]:
            print(
                f"  - {title} | sentence {sentence_id}"
            )

        print("Retrieved titles:")

        for document in result["retrieved_documents"][:10]:
            rank = document.get("rank", "?")
            title = document.get("title", "")

            print(
                f"  [{rank}] {title}"
            )

    # ------------------------------------------------------------
    # Save detailed analysis
    # ------------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "experiments"
        / "generation"
        / "rag_error_analysis.json"
    )

    output = {
        "num_examples": total,
        "retrieval_success": retrieval_success,
        "retrieval_failure": retrieval_failure,
        "generation_success": generation_success,
        "generation_failure": generation_failure,
        "retrieval_success_generation_success":
            retrieval_success_generation_success,
        "retrieval_success_generation_failure":
            retrieval_success_generation_failure,
        "retrieval_failure_generation_success":
            retrieval_failure_generation_success,
        "retrieval_failure_generation_failure":
            retrieval_failure_generation_failure,
        "average_supporting_fact_coverage":
            average_retrieval_fraction,
        "results": analyzed_results,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Detailed results saved to:\n"
        f"{output_path}"
    )

    if missing_dataset_examples:
        print(
            f"\nWarning: {missing_dataset_examples} predictions "
            f"could not be matched to the dataset."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()