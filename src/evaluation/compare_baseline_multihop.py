import json
import re
from pathlib import Path

from src.retrieval.reranked_hybrid_retriever import RerankedHybridRetriever
from src.retrieval.multihop_retriever import MultiHopRetriever


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
    / "baseline_vs_multihop.json"
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
    title = re.sub(r"\s+", " ", title)
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
            title = normalize_title(fact[0])

            try:
                sentence_id = int(fact[1])
            except (TypeError, ValueError):
                continue

            facts.append(
                (title, sentence_id)
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

            facts.add(
                (title, sentence_id)
            )

    return facts


def evaluate_retrieval(
    gold_facts,
    documents,
):
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

    complete = (
        len(gold_facts) > 0
        and len(matched) == len(gold_facts)
    )

    if gold_facts:
        coverage = (
            len(matched)
            / len(gold_facts)
        )
    else:
        coverage = 0.0

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

    print("=" * 70)
    print("BASELINE VS MULTI-HOP COMPARISON")
    print("=" * 70)

    dataset = load_dataset(
        DATASET_PATH
    )

    examples = dataset[:50]

    print(
        f"Evaluating {len(examples)} examples"
    )

    # ---------------------------------------------------------
    # Initialize both retrievers
    # ---------------------------------------------------------

    print("\nInitializing baseline retriever...")

    baseline = RerankedHybridRetriever(
        candidate_k=20
    )

    print("\nInitializing multi-hop retriever...")

    multihop = MultiHopRetriever(
        candidate_k=20,
        initial_k=10,
        expansion_k=5,
        final_k=15,
    )

    # ---------------------------------------------------------
    # Counters
    # ---------------------------------------------------------

    baseline_correct = 0
    multihop_correct = 0

    fixed = []
    regressed = []
    both_correct = []
    both_wrong = []

    total_baseline_coverage = 0.0
    total_multihop_coverage = 0.0

    # ---------------------------------------------------------
    # Evaluate each example
    # ---------------------------------------------------------

    for index, example in enumerate(
        examples,
        start=1,
    ):

        question = example["question"]

        print(
            f"\n[{index}/{len(examples)}]"
        )

        print(question)

        gold_facts = get_supporting_facts(
            example
        )

        # -----------------------------------------------------
        # Baseline
        # -----------------------------------------------------

        baseline_documents = baseline.retrieve(
            question,
            k=15,
        )

        baseline_result = evaluate_retrieval(
            gold_facts,
            baseline_documents,
        )

        # -----------------------------------------------------
        # Multi-hop
        # -----------------------------------------------------

        multihop_documents = multihop.retrieve(
            question,
            k=15,
        )

        multihop_result = evaluate_retrieval(
            gold_facts,
            multihop_documents,
        )

        baseline_ok = baseline_result[
            "complete"
        ]

        multihop_ok = multihop_result[
            "complete"
        ]

        if baseline_ok:
            baseline_correct += 1

        if multihop_ok:
            multihop_correct += 1

        total_baseline_coverage += (
            baseline_result["coverage"]
        )

        total_multihop_coverage += (
            multihop_result["coverage"]
        )

        record = {
            "id": example["_id"],
            "question": question,
            "answer": example["answer"],
            "gold_facts": format_facts(
                gold_facts
            ),
            "baseline": {
                "complete": baseline_ok,
                "coverage": baseline_result[
                    "coverage"
                ],
                "matched_facts": format_facts(
                    baseline_result["matched"]
                ),
                "missed_facts": format_facts(
                    baseline_result["missed"]
                ),
                "retrieved_titles": get_titles(
                    baseline_documents
                ),
            },
            "multihop": {
                "complete": multihop_ok,
                "coverage": multihop_result[
                    "coverage"
                ],
                "matched_facts": format_facts(
                    multihop_result["matched"]
                ),
                "missed_facts": format_facts(
                    multihop_result["missed"]
                ),
                "retrieved_titles": get_titles(
                    multihop_documents
                ),
            },
        }

        # -----------------------------------------------------
        # Classification
        # -----------------------------------------------------

        if not baseline_ok and multihop_ok:

            fixed.append(record)

            print(
                "RESULT: FIXED BY MULTI-HOP"
            )

        elif baseline_ok and not multihop_ok:

            regressed.append(record)

            print(
                "RESULT: REGRESSION"
            )

        elif baseline_ok and multihop_ok:

            both_correct.append(record)

            print(
                "RESULT: BOTH CORRECT"
            )

        else:

            both_wrong.append(record)

            print(
                "RESULT: BOTH WRONG"
            )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    total = len(examples)

    baseline_rate = (
        baseline_correct / total
    )

    multihop_rate = (
        multihop_correct / total
    )

    baseline_coverage = (
        total_baseline_coverage / total
    )

    multihop_coverage = (
        total_multihop_coverage / total
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Baseline complete retrieval: "
        f"{baseline_correct}/{total} "
        f"({baseline_rate * 100:.2f}%)"
    )

    print(
        f"Multi-hop complete retrieval: "
        f"{multihop_correct}/{total} "
        f"({multihop_rate * 100:.2f}%)"
    )

    print(
        f"Baseline coverage: "
        f"{baseline_coverage * 100:.2f}%"
    )

    print(
        f"Multi-hop coverage: "
        f"{multihop_coverage * 100:.2f}%"
    )

    print("\n" + "=" * 70)
    print("QUESTION TRANSITIONS")
    print("=" * 70)

    print(
        f"Both correct : "
        f"{len(both_correct)}"
    )

    print(
        f"Fixed        : "
        f"{len(fixed)}"
    )

    print(
        f"Regressed    : "
        f"{len(regressed)}"
    )

    print(
        f"Both wrong   : "
        f"{len(both_wrong)}"
    )

    # ---------------------------------------------------------
    # Fixed questions
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIXED BY MULTI-HOP")
    print("=" * 70)

    for i, record in enumerate(
        fixed,
        start=1,
    ):

        print(
            f"\n{i}. {record['question']}"
        )

        print(
            f"Answer: {record['answer']}"
        )

        print("Previously missed:")

        for fact in record["baseline"][
            "missed_facts"
        ]:
            print(
                f"  - {fact['title']} "
                f"[{fact['sentence_id']}]"
            )

        print("Multi-hop retrieved:")

        for fact in record["multihop"][
            "matched_facts"
        ]:
            print(
                f"  + {fact['title']} "
                f"[{fact['sentence_id']}]"
            )

    # ---------------------------------------------------------
    # Regressions
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("REGRESSIONS")
    print("=" * 70)

    if not regressed:

        print(
            "No regressions found."
        )

    else:

        for i, record in enumerate(
            regressed,
            start=1,
        ):

            print(
                f"\n{i}. {record['question']}"
            )

            print(
                "Baseline had complete retrieval."
            )

            print("Multi-hop missed:")

            for fact in record["multihop"][
                "missed_facts"
            ]:
                print(
                    f"  - {fact['title']} "
                    f"[{fact['sentence_id']}]"
                )

    # ---------------------------------------------------------
    # Save complete analysis
    # ---------------------------------------------------------

    output = {
        "num_examples": total,
        "baseline": {
            "complete_count": baseline_correct,
            "complete_rate": baseline_rate,
            "average_coverage": baseline_coverage,
        },
        "multihop": {
            "complete_count": multihop_correct,
            "complete_rate": multihop_rate,
            "average_coverage": multihop_coverage,
        },
        "transition_counts": {
            "both_correct": len(both_correct),
            "fixed": len(fixed),
            "regressed": len(regressed),
            "both_wrong": len(both_wrong),
        },
        "fixed": fixed,
        "regressed": regressed,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
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

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()