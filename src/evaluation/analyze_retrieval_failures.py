import json
import re
from collections import Counter
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "retrieval"
    / "retrieval_failure_analysis.json"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gold_facts(example):
    """
    Returns:
        [
            ("Scott Derrickson", 0),
            ("Ed Wood", 0)
        ]
    """
    facts = []

    for title, sent_id in example["supporting_facts"]:
        facts.append((normalize(title), int(sent_id)))

    return facts


def get_retrieved_facts(documents):
    """
    Convert retrieved chunks into
    (title, sentence_id) pairs.
    """
    retrieved = set()

    for doc in documents:

        title = normalize(
            doc.metadata["title"]
        )

        sentence_ids = doc.metadata.get(
            "sentence_ids",
            [],
        )

        for sid in sentence_ids:
            retrieved.add((title, int(sid)))

    return retrieved


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("RETRIEVAL FAILURE ANALYSIS")
    print("=" * 70)

    dataset = load_dataset()

    examples = dataset[:50]

    print(f"Loaded {len(examples)} examples")

    print("\nInitializing retriever...")

    retriever = RerankedHybridRetriever(
        candidate_k=20
    )

    retrieval_success = 0
    retrieval_failure = 0

    missed_titles = Counter()

    failure_examples = []

    for i, example in enumerate(examples, start=1):

        question = example["question"]

        print(f"[{i}/50] {question}")

        docs = retriever.retrieve(
            question,
            k=15,
        )

        gold = get_gold_facts(example)

        retrieved = get_retrieved_facts(docs)

        matched = [
            fact for fact in gold
            if fact in retrieved
        ]

        missed = [
            fact for fact in gold
            if fact not in retrieved
        ]

        if len(missed) == 0:

            retrieval_success += 1

        else:

            retrieval_failure += 1

            for title, _ in missed:
                missed_titles[title] += 1

            failure_examples.append(
                {
                    "question": question,
                    "gold_answer": example["answer"],
                    "matched": matched,
                    "missed": missed,
                    "retrieved_titles": [
                        doc.metadata["title"]
                        for doc in docs
                    ],
                }
            )

    # -----------------------------------------------------
    # Print statistics
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Retrieval success : {retrieval_success}"
    )

    print(
        f"Retrieval failure : {retrieval_failure}"
    )

    print(
        f"Success rate      : "
        f"{100*retrieval_success/50:.2f}%"
    )

    print("\nMost frequently missed titles")

    print("-" * 70)

    for title, count in missed_titles.most_common(20):

        print(
            f"{title:45} {count}"
        )

    print("\n" + "=" * 70)
    print("EXAMPLE FAILURES")
    print("=" * 70)

    for idx, ex in enumerate(
        failure_examples[:10],
        start=1,
    ):

        print("\n" + "-" * 70)

        print(f"Example {idx}")

        print(
            f"Question : {ex['question']}"
        )

        print(
            f"Gold     : {ex['gold_answer']}"
        )

        print("\nMatched supporting facts")

        for title, sid in ex["matched"]:

            print(f"  ✓ {title} [{sid}]")

        print("\nMissed supporting facts")

        for title, sid in ex["missed"]:

            print(f"  ✗ {title} [{sid}]")

        print("\nRetrieved titles")

        for t in ex["retrieved_titles"]:
            print(f"  - {t}")

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

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
            {
                "retrieval_success": retrieval_success,
                "retrieval_failure": retrieval_failure,
                "missed_titles": dict(missed_titles),
                "failure_examples": failure_examples,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved analysis to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()