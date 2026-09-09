import json
import re
from pathlib import Path

from src.retrieval.reranked_hybrid_retriever import RerankedHybridRetriever
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
    / "baseline_vs_recall_preserving_multihop.json"
)


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_title(title):
    title = title.lower().strip()
    return re.sub(r"\s+", " ", title)


def get_supporting_facts(example):
    facts = []

    for fact in example.get("supporting_facts", []):
        if isinstance(fact, (list, tuple)) and len(fact) >= 2:
            title = normalize_title(fact[0])

            try:
                sentence_id = int(fact[1])
            except (TypeError, ValueError):
                continue

            facts.append((title, sentence_id))

    return facts


def get_retrieved_facts(documents):
    facts = set()

    for document in documents:
        title = normalize_title(
            document.metadata.get("title", "")
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

            facts.add((title, sentence_id))

    return facts


def evaluate(gold_facts, documents):
    retrieved_facts = get_retrieved_facts(documents)

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

    coverage = (
        len(matched) / len(gold_facts)
        if gold_facts
        else 0.0
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
        document.metadata.get("title", "")
        for document in documents
    ]


def main():

    print("=" * 70)
    print("BASELINE VS RECALL-PRESERVING MULTI-HOP")
    print("=" * 70)

    dataset = load_dataset(DATASET_PATH)
    examples = dataset[:50]

    print(
        f"Evaluating {len(examples)} examples"
    )

    print("\nInitializing baseline...")

    baseline = RerankedHybridRetriever(
        candidate_k=20
    )

    print("\nInitializing recall-preserving multi-hop...")

    multihop = RecallPreservingMultiHopRetriever(
        candidate_k=20,
        initial_k=15,
        expansion_k=5,
        expansion_candidate_k=20,
        protected_k=5,
        final_k=15,
    )

    both_correct = []
    fixed = []
    regressed = []
    both_wrong = []

    baseline_coverage_total = 0.0
    multihop_coverage_total = 0.0

    for index, example in enumerate(
        examples,
        start=1,
    ):

        question = example["question"]

        print(
            f"\n[{index}/{len(examples)}] {question}"
        )

        gold_facts = get_supporting_facts(example)

        # -----------------------------------------------------
        # Baseline
        # -----------------------------------------------------

        baseline_documents = baseline.retrieve(
            question,
            k=15,
        )

        baseline_result = evaluate(
            gold_facts,
            baseline_documents,
        )

        # -----------------------------------------------------
        # Recall-preserving multi-hop
        # -----------------------------------------------------

        multihop_documents = multihop.retrieve(
            question,
            k=15,
        )

        multihop_result = evaluate(
            gold_facts,
            multihop_documents,
        )

        baseline_ok = baseline_result["complete"]
        multihop_ok = multihop_result["complete"]

        baseline_coverage_total += (
            baseline_result["coverage"]
        )

        multihop_coverage_total += (
            multihop_result["coverage"]
        )

        record = {
            "id": example["_id"],
            "question": question,
            "answer": example["answer"],
            "gold_facts": format_facts(gold_facts),
            "baseline": {
                "complete": baseline_ok,
                "coverage": baseline_result["coverage"],
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
            "recall_preserving_multihop": {
                "complete": multihop_ok,
                "coverage": multihop_result["coverage"],
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
        # Transition
        # -----------------------------------------------------

        if baseline_ok and multihop_ok:
            both_correct.append(record)
            print("RESULT: BOTH CORRECT")

        elif not baseline_ok and multihop_ok:
            fixed.append(record)
            print("RESULT: FIXED BY MULTI-HOP")

        elif baseline_ok and not multihop_ok:
            regressed.append(record)
            print("RESULT: REGRESSION")

        else:
            both_wrong.append(record)
            print("RESULT: BOTH WRONG")

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    total = len(examples)

    baseline_complete = (
        len(both_correct) + len(regressed)
    )

    multihop_complete = (
        len(both_correct) + len(fixed)
    )

    baseline_rate = baseline_complete / total
    multihop_rate = multihop_complete / total

    baseline_coverage = (
        baseline_coverage_total / total
    )

    multihop_coverage = (
        multihop_coverage_total / total
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Baseline complete retrieval: "
        f"{baseline_complete}/{total} "
        f"({baseline_rate * 100:.2f}%)"
    )

    print(
        f"Recall-preserving complete retrieval: "
        f"{multihop_complete}/{total} "
        f"({multihop_rate * 100:.2f}%)"
    )

    print(
        f"Baseline coverage: "
        f"{baseline_coverage * 100:.2f}%"
    )

    print(
        f"Recall-preserving coverage: "
        f"{multihop_coverage * 100:.2f}%"
    )

    print("\n" + "=" * 70)
    print("QUESTION TRANSITIONS")
    print("=" * 70)

    print(
        f"Both correct : {len(both_correct)}"
    )

    print(
        f"Fixed        : {len(fixed)}"
    )

    print(
        f"Regressed    : {len(regressed)}"
    )

    print(
        f"Both wrong   : {len(both_wrong)}"
    )

    # ---------------------------------------------------------
    # Fixed
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIXED BY RECALL-PRESERVING MULTI-HOP")
    print("=" * 70)

    for index, record in enumerate(
        fixed,
        start=1,
    ):

        print(
            f"\n{index}. {record['question']}"
        )

        print(
            f"Answer: {record['answer']}"
        )

        print("Previously missed:")

        for fact in record["baseline"]["missed_facts"]:
            print(
                f"  - {fact['title']} "
                f"[{fact['sentence_id']}]"
            )

        print("Now retrieved:")

        for fact in record[
            "recall_preserving_multihop"
        ]["matched_facts"]:

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

        print("No regressions found.")

    else:

        for index, record in enumerate(
            regressed,
            start=1,
        ):

            print(
                f"\n{index}. {record['question']}"
            )

            print("Multi-hop missed:")

            for fact in record[
                "recall_preserving_multihop"
            ]["missed_facts"]:

                print(
                    f"  - {fact['title']} "
                    f"[{fact['sentence_id']}]"
                )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = {
        "num_examples": total,
        "baseline": {
            "complete_count": baseline_complete,
            "complete_rate": baseline_rate,
            "average_coverage": baseline_coverage,
        },
        "recall_preserving_multihop": {
            "complete_count": multihop_complete,
            "complete_rate": multihop_rate,
            "average_coverage": multihop_coverage,
        },
        "transitions": {
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