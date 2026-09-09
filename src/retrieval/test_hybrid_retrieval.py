import json

from src.retrieval.hybrid_retriever import HybridRetriever


DATASET_PATH = (
    "data/raw/hotpotqa/"
    "hotpot_dev_distractor_v1.json"
)


def main():

    # Load one real HotpotQA example
    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    example = data[0]

    question = example["question"]
    answer = example["answer"]
    supporting_facts = example["supporting_facts"]

    print("\n==============================")
    print("QUESTION")
    print("==============================")

    print(question)

    print("\nGold answer:")
    print(answer)

    print("\nGold supporting facts:")
    print(supporting_facts)

    # Build hybrid retriever
    print("\n==============================")
    print("BUILDING HYBRID RETRIEVER")
    print("==============================")

    retriever = HybridRetriever(
        rrf_k=60
    )

    # Retrieve candidates from both systems
    results = retriever.retrieve(
        question,
        k=10,
        candidate_k=10
    )

    print("\n==============================")
    print("HYBRID RESULTS")
    print("==============================")

    for rank, document in enumerate(
        results,
        start=1
    ):

        print(
            f"\n---------- RANK {rank} ----------"
        )

        print(
            "Title:",
            document.metadata["title"]
        )

        print(
            "Sentence IDs:",
            document.metadata["sentence_ids"]
        )

        print("\nText:")
        print(document.page_content)

        print(
    "Dense rank:",
    document.metadata.get("dense_rank")
)

        print(
    "BM25 rank:",
    document.metadata.get("bm25_rank")
)

        print(
    "RRF score:",
    document.metadata.get("rrf_score")
)


if __name__ == "__main__":
    main()