import json

from src.retrieval.bm25_retriever import BM25Retriever


DATASET_PATH = "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    data = load_dataset()

    example = data[0]

    question = example["question"]
    answer = example["answer"]
    supporting_facts = example["supporting_facts"]

    print("\nQuestion:")
    print(question)

    print("\nGold answer:")
    print(answer)

    print("\nGold supporting facts:")
    print(supporting_facts)

    retriever = BM25Retriever()

    results = retriever.retrieve(
        question,
        k=5
    )

    print("\n===== BM25 Results =====")

    for rank, document in enumerate(results, start=1):

        print(f"\nRank {rank}")

        print(
            "Title:",
            document.metadata["title"]
        )

        print(
            "Sentence IDs:",
            document.metadata["sentence_ids"]
        )

        print(
            "BM25 score:",
            round(
                document.metadata["bm25_score"],
                4
            )
        )

        print(
            "Text:",
            document.page_content[:300]
        )


if __name__ == "__main__":
    main()