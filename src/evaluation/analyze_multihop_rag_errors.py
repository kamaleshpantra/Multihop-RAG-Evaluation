import json
import re
from pathlib import Path

from src.data.loader import load_hotpotqa


DATASET_PATH = Path(
    "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"
)

MULTIHOP_RESULTS_PATH = Path(
    "experiments/generation/rag_multihop_predictions.json"
)

BASELINE_RESULTS_PATH = Path(
    "experiments/generation/rag_predictions.json"
)

OUTPUT_PATH = Path(
    "experiments/generation/rag_multihop_error_analysis.json"
)


def normalize_answer(answer):
    answer = str(answer).lower().strip()

    answer = re.sub(
        r"[^a-z0-9\s]",
        "",
        answer,
    )

    answer = re.sub(
        r"\s+",
        " ",
        answer,
    )

    return answer


def token_f1(prediction, gold):
    prediction_tokens = normalize_answer(
        prediction
    ).split()

    gold_tokens = normalize_answer(
        gold
    ).split()

    if not prediction_tokens or not gold_tokens:
        return 0.0

    common_tokens = set(prediction_tokens) & set(
        gold_tokens
    )

    if not common_tokens:
        return 0.0

    common_count = sum(
        min(
            prediction_tokens.count(token),
            gold_tokens.count(token),
        )
        for token in common_tokens
    )

    precision = (
        common_count / len(prediction_tokens)
    )

    recall = (
        common_count / len(gold_tokens)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def normalize_title(title):
    return str(title).strip().lower()


def fact_key(title, sentence_id):
    return (
        normalize_title(title),
        int(sentence_id),
    )


def get_gold_facts(example):
    facts = set()

    for fact in example.get(
        "supporting_facts",
        [],
    ):
        if isinstance(fact, (list, tuple)):
            if len(fact) >= 2:
                title = fact[0]
                sentence_id = fact[1]

                facts.add(
                    fact_key(
                        title,
                        sentence_id,
                    )
                )

        elif isinstance(fact, dict):
            title = fact.get("title")

            sentence_id = fact.get(
                "sent_id",
                fact.get(
                    "sentence_id"
                ),
            )

            if (
                title is not None
                and sentence_id is not None
            ):
                facts.add(
                    fact_key(
                        title,
                        sentence_id,
                    )
                )

    return facts


def get_retrieved_facts(prediction):
    facts = set()

    for document in prediction.get(
        "retrieved_documents",
        [],
    ):
        title = document.get(
            "title",
            "",
        )

        sentence_ids = document.get(
            "sentence_ids",
            [],
        )

        for sentence_id in sentence_ids:
            facts.add(
                fact_key(
                    title,
                    sentence_id,
                )
            )

    return facts


def analyze_example(
    example,
    prediction,
):
    gold_facts = get_gold_facts(
        example
    )

    retrieved_facts = get_retrieved_facts(
        prediction
    )

    retrieved_gold_facts = (
        gold_facts & retrieved_facts
    )

    missed_gold_facts = (
        gold_facts - retrieved_facts
    )

    total_gold_facts = len(gold_facts)

    if total_gold_facts == 0:
        coverage = 0.0
    else:
        coverage = (
            len(retrieved_gold_facts)
            / total_gold_facts
        )

    retrieval_complete = (
        len(missed_gold_facts) == 0
    )

    predicted_answer = prediction.get(
        "predicted_answer",
        prediction.get(
            "answer",
            "",
        ),
    )

    gold_answer = example["answer"]

    exact_match = (
        normalize_answer(
            predicted_answer
        )
        == normalize_answer(
            gold_answer
        )
    )

    f1 = token_f1(
        predicted_answer,
        gold_answer,
    )

    if retrieval_complete and exact_match:
        category = "retrieval_success_generation_success"

    elif retrieval_complete and not exact_match:
        category = "retrieval_success_generation_failure"

    elif not retrieval_complete and exact_match:
        category = "retrieval_failure_generation_success"

    else:
        category = "retrieval_failure_generation_failure"

    return {
        "id": example["_id"],
        "question": example["question"],
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        "exact_match": exact_match,
        "f1": f1,
        "retrieval_complete": retrieval_complete,
        "gold_fact_count": total_gold_facts,
        "retrieved_gold_fact_count": len(
            retrieved_gold_facts
        ),
        "coverage": coverage,
        "retrieved_gold_facts": [
            {
                "title": title,
                "sentence_id": sentence_id,
            }
            for title, sentence_id
            in sorted(retrieved_gold_facts)
        ],
        "missed_gold_facts": [
            {
                "title": title,
                "sentence_id": sentence_id,
            }
            for title, sentence_id
            in sorted(missed_gold_facts)
        ],
        "category": category,
    }


def load_predictions(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if isinstance(data, dict):
        predictions = data.get(
            "predictions",
            [],
        )
    else:
        predictions = data

    return predictions


def main():
    print("=" * 70)
    print("MULTI-HOP RAG ERROR ANALYSIS")
    print("=" * 70)

    print()
    print("Loading dataset...")

    dataset = load_hotpotqa(
        DATASET_PATH
    )

    dataset = dataset[:50]

    print(
        f"Loaded {len(dataset)} examples."
    )

    print()
    print("Loading multi-hop predictions...")

    if not MULTIHOP_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing results file:\n"
            f"{MULTIHOP_RESULTS_PATH}"
        )

    multihop_predictions = load_predictions(
        MULTIHOP_RESULTS_PATH
    )

    print(
        f"Loaded {len(multihop_predictions)} "
        "multi-hop predictions."
    )

    baseline_predictions = None

    if BASELINE_RESULTS_PATH.exists():
        print()
        print("Loading baseline predictions...")

        baseline_predictions = load_predictions(
            BASELINE_RESULTS_PATH
        )

        print(
            f"Loaded {len(baseline_predictions)} "
            "baseline predictions."
        )

    prediction_by_id = {
        prediction["id"]: prediction
        for prediction in multihop_predictions
        if "id" in prediction
    }

    baseline_by_id = {}

    if baseline_predictions is not None:
        baseline_by_id = {
            prediction["id"]: prediction
            for prediction in baseline_predictions
            if "id" in prediction
        }

    analyses = []

    for example in dataset:
        example_id = example["_id"]

        prediction = prediction_by_id.get(
            example_id
        )

        if prediction is None:
            print(
                f"WARNING: Missing prediction "
                f"for {example_id}"
            )
            continue

        analysis = analyze_example(
            example,
            prediction,
        )

        if example_id in baseline_by_id:
            baseline_prediction = (
                baseline_by_id[example_id]
            )

            baseline_exact_match = (
                normalize_answer(
                    baseline_prediction.get(
                        "predicted_answer",
                        "",
                    )
                )
                == normalize_answer(
                    example["answer"]
                )
            )

            analysis[
                "baseline_exact_match"
            ] = baseline_exact_match

            if (
                not baseline_exact_match
                and analysis["exact_match"]
            ):
                analysis[
                    "transition"
                ] = "fixed"

            elif (
                baseline_exact_match
                and not analysis["exact_match"]
            ):
                analysis[
                    "transition"
                ] = "regressed"

            elif (
                baseline_exact_match
                and analysis["exact_match"]
            ):
                analysis[
                    "transition"
                ] = "both_correct"

            else:
                analysis[
                    "transition"
                ] = "both_wrong"

        analyses.append(analysis)

    total = len(analyses)

    complete_retrieval = sum(
        a["retrieval_complete"]
        for a in analyses
    )

    retrieval_success_generation_success = sum(
        a["category"]
        == "retrieval_success_generation_success"
        for a in analyses
    )

    retrieval_success_generation_failure = sum(
        a["category"]
        == "retrieval_success_generation_failure"
        for a in analyses
    )

    retrieval_failure_generation_success = sum(
        a["category"]
        == "retrieval_failure_generation_success"
        for a in analyses
    )

    retrieval_failure_generation_failure = sum(
        a["category"]
        == "retrieval_failure_generation_failure"
        for a in analyses
    )

    exact_matches = sum(
        a["exact_match"]
        for a in analyses
    )

    average_f1 = (
        sum(a["f1"] for a in analyses)
        / total
        if total
        else 0.0
    )

    average_coverage = (
        sum(a["coverage"] for a in analyses)
        / total
        if total
        else 0.0
    )

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"Total examples: "
        f"{total}"
    )

    print(
        f"Exact Match: "
        f"{exact_matches}/{total} "
        f"({exact_matches / total * 100:.2f}%)"
    )

    print(
        f"Average F1: "
        f"{average_f1:.4f}"
    )

    print(
        f"Complete retrieval: "
        f"{complete_retrieval}/{total} "
        f"({complete_retrieval / total * 100:.2f}%)"
    )

    print(
        f"Average supporting-fact coverage: "
        f"{average_coverage * 100:.2f}%"
    )

    print()
    print("=" * 70)
    print("RETRIEVAL VS GENERATION")
    print("=" * 70)

    print(
        "Retrieval success + generation success: "
        f"{retrieval_success_generation_success}"
    )

    print(
        "Retrieval success + generation failure: "
        f"{retrieval_success_generation_failure}"
    )

    print(
        "Retrieval failure + generation success: "
        f"{retrieval_failure_generation_success}"
    )

    print(
        "Retrieval failure + generation failure: "
        f"{retrieval_failure_generation_failure}"
    )

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    generation_failures_with_evidence = (
        retrieval_success_generation_failure
    )

    retrieval_failures = (
        retrieval_failure_generation_failure
        + retrieval_failure_generation_success
    )

    print(
        f"Questions with retrieval failure: "
        f"{retrieval_failures}/{total}"
    )

    print(
        f"Questions with generation failure "
        f"despite complete retrieval: "
        f"{generation_failures_with_evidence}/{total}"
    )

    if total:
        print(
            f"Retrieval failure rate: "
            f"{retrieval_failures / total * 100:.2f}%"
        )

        print(
            "Generation failure despite complete "
            "retrieval rate: "
            f"{generation_failures_with_evidence / total * 100:.2f}%"
        )

    if baseline_predictions is not None:
        transitions = {
            "both_correct": 0,
            "fixed": 0,
            "regressed": 0,
            "both_wrong": 0,
        }

        for analysis in analyses:
            transition = analysis.get(
                "transition"
            )

            if transition in transitions:
                transitions[
                    transition
                ] += 1

        print()
        print("=" * 70)
        print("BASELINE → MULTI-HOP TRANSITIONS")
        print("=" * 70)

        print(
            f"Both correct : "
            f"{transitions['both_correct']}"
        )

        print(
            f"Fixed        : "
            f"{transitions['fixed']}"
        )

        print(
            f"Regressed    : "
            f"{transitions['regressed']}"
        )

        print(
            f"Both wrong   : "
            f"{transitions['both_wrong']}"
        )

    print()
    print("=" * 70)
    print("GENERATION FAILURES WITH COMPLETE RETRIEVAL")
    print("=" * 70)

    count = 0

    for analysis in analyses:
        if analysis["category"] != (
            "retrieval_success_generation_failure"
        ):
            continue

        count += 1

        print()
        print(
            f"{count}. "
            f"{analysis['question']}"
        )

        print(
            f"Gold: "
            f"{analysis['gold_answer']}"
        )

        print(
            f"Predicted: "
            f"{analysis['predicted_answer']}"
        )

        print(
            f"F1: "
            f"{analysis['f1']:.4f}"
        )

        print(
            "Supporting facts retrieved: "
            f"{analysis['retrieved_gold_fact_count']}/"
            f"{analysis['gold_fact_count']}"
        )

    print()
    print("=" * 70)
    print("RETRIEVAL FAILURES")
    print("=" * 70)

    count = 0

    for analysis in analyses:
        if analysis["retrieval_complete"]:
            continue

        count += 1

        print()
        print(
            f"{count}. "
            f"{analysis['question']}"
        )

        print(
            f"Gold: "
            f"{analysis['gold_answer']}"
        )

        print(
            f"Predicted: "
            f"{analysis['predicted_answer']}"
        )

        print(
            f"Supporting-fact coverage: "
            f"{analysis['coverage'] * 100:.2f}%"
        )

        print("Missed:")

        for fact in analysis[
            "missed_gold_facts"
        ]:
            print(
                f"  - "
                f"{fact['title']} "
                f"[{fact['sentence_id']}]"
            )

    if baseline_predictions is not None:
        print()
        print("=" * 70)
        print("FIXED BY MULTI-HOP")
        print("=" * 70)

        count = 0

        for analysis in analyses:
            if analysis.get(
                "transition"
            ) != "fixed":
                continue

            count += 1

            print()
            print(
                f"{count}. "
                f"{analysis['question']}"
            )

            print(
                f"Gold: "
                f"{analysis['gold_answer']}"
            )

            print(
                f"Predicted: "
                f"{analysis['predicted_answer']}"
            )

            print(
                f"Supporting-fact coverage: "
                f"{analysis['coverage'] * 100:.2f}%"
            )

        print()
        print("=" * 70)
        print("REGRESSED BY MULTI-HOP")
        print("=" * 70)

        count = 0

        for analysis in analyses:
            if analysis.get(
                "transition"
            ) != "regressed":
                continue

            count += 1

            print()
            print(
                f"{count}. "
                f"{analysis['question']}"
            )

            print(
                f"Gold: "
                f"{analysis['gold_answer']}"
            )

            print(
                f"Predicted: "
                f"{analysis['predicted_answer']}"
            )

            print(
                f"Supporting-fact coverage: "
                f"{analysis['coverage'] * 100:.2f}%"
            )

    output = {
        "configuration": {
            "dataset": str(
                DATASET_PATH
            ),
            "multihop_results": str(
                MULTIHOP_RESULTS_PATH
            ),
            "baseline_results": str(
                BASELINE_RESULTS_PATH
            ),
        },
        "summary": {
            "total": total,
            "exact_match": (
                exact_matches / total
                if total
                else 0.0
            ),
            "average_f1": average_f1,
            "complete_retrieval": (
                complete_retrieval / total
                if total
                else 0.0
            ),
            "average_coverage": (
                average_coverage
            ),
            "retrieval_success_generation_success":
                retrieval_success_generation_success,
            "retrieval_success_generation_failure":
                retrieval_success_generation_failure,
            "retrieval_failure_generation_success":
                retrieval_failure_generation_success,
            "retrieval_failure_generation_failure":
                retrieval_failure_generation_failure,
        },
        "examples": analyses,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()